/**
 * wopr-plugin-memory-semantic
 *
 * Semantic memory search with embeddings, auto-recall, and auto-capture for WOPR
 *
 * Features:
 * - Vector embeddings (OpenAI, Gemini, local via node-llama-cpp)
 * - Hybrid search (vector + keyword)
 * - Auto-recall: inject relevant memories before agent processing
 * - Auto-capture: extract and store important information from conversations
 */

import { mkdirSync } from "node:fs";
import { join } from "node:path";
import type { WOPRPlugin, WOPRPluginContext } from "@wopr-network/plugin-types";
import winston from "winston";
import { unregisterMemoryTools } from "./a2a-tools.js";
import { stopWatcher } from "./core-memory/watcher.js";
import { EmbeddingQueue } from "./embedding-queue.js";
import { handleFilesChanged, handleMemorySearch } from "./event-handlers.js";
import { handleAfterInject, handleBeforeInject } from "./hooks.js";
import { initialize, type PluginState } from "./init.js";
import { contentHash, memoryContextProvider, pluginConfigSchema, pluginManifest } from "./manifest.js";
import type { MemorySearchResult, SemanticMemoryConfig } from "./types.js";
import { DEFAULT_CONFIG } from "./types.js";

/**
 * Extended plugin context — adds the optional `memory` extension that
 * core exposes for keyword search fallback and `registerTool`.
 * Everything else (including `storage`) comes from @wopr-network/plugin-types.
 */
interface PluginContext extends WOPRPluginContext {
  memory?: {
    keywordSearch?(query: string, limit: number): Promise<any[]>;
  };
  registerTool?(tool: any): void;
}

let ctx: PluginContext | null = null;
const cleanups: Array<() => void> = [];

const logsDir = join(process.env.WOPR_HOME || "/tmp/wopr-test", "logs");
try {
  mkdirSync(logsDir, { recursive: true });
} catch {}

const log = winston.createLogger({
  level: "debug",
  format: winston.format.combine(winston.format.timestamp(), winston.format.json()),
  defaultMeta: { service: "semantic-memory" },
  transports: [
    new winston.transports.File({ filename: join(logsDir, "semantic-memory-error.log"), level: "error" }),
    new winston.transports.File({ filename: join(logsDir, "semantic-memory.log"), level: "debug" }),
    new winston.transports.Console({ level: "warn" }),
  ],
});

// =============================================================================
// Plugin State
// =============================================================================

const state: PluginState = {
  config: DEFAULT_CONFIG,
  embeddingProvider: null,
  searchManager: null,
  memoryManager: null,
  api: null,
  initialized: false,
  eventCleanup: [],
  instanceId: undefined,
};

const embeddingQueue = new EmbeddingQueue(log);

// =============================================================================
// Plugin Export
// =============================================================================

const plugin: WOPRPlugin = {
  name: "wopr-plugin-memory-semantic",
  version: "1.0.0",
  description: "Semantic memory search with embeddings, auto-recall, and auto-capture",
  manifest: pluginManifest,

  async init(api: WOPRPluginContext) {
    ctx = api as PluginContext;

    // Override ctx.log to use our file-backed winston logger
    ctx.log = {
      info: (msg: string) => log.info(msg),
      warn: (msg: string) => log.warn(msg),
      error: (msg: string) => log.error(msg),
      debug: (msg: string) => log.debug(msg),
    };
    ctx.log.info("[semantic-memory] init() called");

    // Clean up previous registrations if re-initialized
    for (let i = cleanups.length - 1; i >= 0; i--) {
      try {
        cleanups[i]();
      } catch {
        /* ignore */
      }
    }
    cleanups.length = 0;
    state.eventCleanup = [];

    // Register config schema
    ctx.registerConfigSchema("wopr-plugin-memory-semantic", pluginConfigSchema);
    cleanups.push(() => ctx?.unregisterConfigSchema("wopr-plugin-memory-semantic"));

    // Register context provider
    ctx.registerContextProvider(memoryContextProvider);
    cleanups.push(() => ctx?.unregisterContextProvider("memory-semantic"));

    // Read config from WOPR central config (set by onboard wizard)
    const storedConfig = ctx.getConfig?.() as Partial<SemanticMemoryConfig> | undefined;
    await initialize(ctx, state, embeddingQueue, log, storedConfig);

    if (!state.initialized) {
      ctx.log.error("[semantic-memory] Initialization failed — plugin will not activate");
      return;
    }

    // Register extension (public API for other plugins)
    const extensionApi = {
      search: async (query: string, maxResults?: number): Promise<MemorySearchResult[]> => {
        if (!state.searchManager) throw new Error("Semantic memory not initialized");
        return state.searchManager.search(query, maxResults, state.instanceId);
      },
      capture: async (text: string, source = "manual"): Promise<void> => {
        if (!state.searchManager) throw new Error("Semantic memory not initialized");
        const id = `man-${contentHash(text)}`;
        embeddingQueue.enqueue(
          [
            {
              entry: {
                id,
                path: source,
                startLine: 0,
                endLine: 0,
                source,
                snippet: text.slice(0, 500),
                content: text,
                instanceId: state.instanceId,
              },
              text,
              persist: true,
            },
          ],
          "manual-capture",
        );
      },
      getConfig: (): SemanticMemoryConfig => ({ ...state.config }),
    };
    if (ctx.registerExtension) {
      ctx.registerExtension("memory-semantic", extensionApi);
      cleanups.push(() => ctx?.unregisterExtension?.("memory-semantic"));
    }

    // Register hooks via the event bus — store cleanup functions for shutdown
    const unsubBeforeInject = ctx.events.on("session:beforeInject", (payload: any) =>
      handleBeforeInject(state, log, payload),
    );
    const unsubAfterInject = ctx.events.on("session:afterInject", (payload: any) =>
      handleAfterInject(state, log, embeddingQueue, payload),
    );

    // Subscribe to core's file change events for vector indexing
    const unsubFilesChanged = ctx.events.on("memory:filesChanged", (payload: any) =>
      handleFilesChanged(state, log, embeddingQueue, payload),
    );

    // Hook into memory:search to provide semantic results
    const unsubSearch = ctx.events.on("memory:search", (payload: any) => handleMemorySearch(state, ctx!.log, payload));

    cleanups.push(unsubBeforeInject, unsubAfterInject, unsubFilesChanged, unsubSearch);
    state.eventCleanup = [unsubBeforeInject, unsubAfterInject, unsubFilesChanged, unsubSearch];
    ctx.log.info("[semantic-memory] Plugin initialized - memory_search enhanced with semantic search");
  },

  async shutdown() {
    if (!ctx) return; // Idempotent

    // Stop the embedding queue first
    embeddingQueue.clear();

    // Stop file watcher
    await stopWatcher(ctx.log);

    // Run all cleanup functions in reverse order (event unsubs, extension, context provider, configSchema)
    for (let i = cleanups.length - 1; i >= 0; i--) {
      try {
        cleanups[i]();
      } catch {
        /* ignore */
      }
    }
    cleanups.length = 0;

    // Unregister A2A tools
    unregisterMemoryTools(ctx);

    // Close memory manager
    if (state.memoryManager) {
      await state.memoryManager.close();
      state.memoryManager = null;
    }

    if (state.searchManager) {
      await state.searchManager.close();
      state.searchManager = null;
    }

    state.embeddingProvider = null;
    state.initialized = false;
    state.api = null;
    state.eventCleanup = [];
    ctx = null;
  },
};

export default plugin;

// Re-export A2A tool unregister for shutdown cleanup
export { unregisterMemoryTools as unregisterA2AMemoryTools } from "./a2a-tools.js";
// Re-export types
export type { EmbeddingProvider, MemorySearchResult, SemanticMemoryConfig } from "./types.js";
export { DEFAULT_CONFIG } from "./types.js";
export type { AuthContext, WebMCPHandler, WebMCPRegistryLike, WebMCPTool, WebMCPToolDeclaration } from "./webmcp.js";
// Re-export WebMCP tools for browser-side registration
export { registerMemoryTools, unregisterMemoryTools, WEBMCP_MANIFEST } from "./webmcp.js";
