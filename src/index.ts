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

import type { WOPRPlugin, WOPRPluginContext } from "@wopr-network/plugin-types";
import { unregisterMemoryTools } from "./a2a-tools.js";
import { stopWatcher } from "./core-memory/watcher.js";
import { EmbeddingQueue } from "./embedding-queue.js";
import { handleFilesChanged, handleMemorySearch } from "./event-handlers.js";
import { handleAfterInject, handleBeforeInject } from "./hooks.js";
import { IDENTITY_TOOL_PERMISSION_MAP, unregisterIdentityTools } from "./identity-tools.js";
import { initialize, type PluginState } from "./init.js";
import { contentHash, memoryContextProvider, pluginConfigSchema, pluginManifest } from "./manifest.js";
import type { MemorySearchResult, SemanticMemoryConfig, SessionApi } from "./types.js";
import { DEFAULT_CONFIG } from "./types.js";
import type { WebMCPRegistryLike } from "./webmcp.js";
import {
  registerMemoryTools as registerWebMCPTools,
  unregisterMemoryTools as unregisterWebMCPTools,
} from "./webmcp.js";

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
  session?: SessionApi;
  webmcpRegistry?: WebMCPRegistryLike;
  registerPermission?(name: string): void;
  unregisterPermission?(name: string): void;
  registerToolPermission?(toolName: string, permission: string): void;
  unregisterToolPermission?(toolName: string): void;
}

let ctx: PluginContext | null = null;
const cleanups: Array<() => void> = [];

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

const embeddingQueue = new EmbeddingQueue({ info: console.info, error: console.error });

// =============================================================================
// Config mapping
// =============================================================================

/**
 * Translate flat wizard-config keys to the nested SemanticMemoryConfig shape
 * that initialize() and the runtime expect.
 *
 * The pluginConfigSchema exposes flat fields (autoRecallEnabled,
 * autoCaptureEnabled, searchMaxResults, searchHybridWeight) because config
 * UIs can only render flat key/value pairs. This function converts those to
 * the nested sub-objects that the runtime reads.
 */
export function mapFlatConfigToNested(raw: Record<string, unknown>): Partial<SemanticMemoryConfig> {
  const config: Partial<SemanticMemoryConfig> = {};

  // Pass-through flat keys that are already flat in SemanticMemoryConfig
  if (raw.provider !== undefined) config.provider = raw.provider as SemanticMemoryConfig["provider"];
  if (raw.model !== undefined) config.model = raw.model as string;
  if (raw.baseUrl !== undefined) config.baseUrl = raw.baseUrl as string;
  if (raw.maxWriteBytes !== undefined) config.maxWriteBytes = raw.maxWriteBytes as number;
  if (raw.instanceId !== undefined) config.instanceId = raw.instanceId as string;

  // Map flat wizard keys → nested sub-objects
  if (raw.autoRecallEnabled !== undefined) {
    config.autoRecall = { enabled: raw.autoRecallEnabled as boolean } as SemanticMemoryConfig["autoRecall"];
  }
  if (raw.autoCaptureEnabled !== undefined) {
    config.autoCapture = { enabled: raw.autoCaptureEnabled as boolean } as SemanticMemoryConfig["autoCapture"];
  }
  if (raw.searchMaxResults !== undefined) {
    config.search = { maxResults: Math.round(raw.searchMaxResults as number) } as SemanticMemoryConfig["search"];
  }
  if (raw.searchHybridWeight !== undefined) {
    const vectorWeight = Math.max(0, Math.min(1, raw.searchHybridWeight as number));
    config.hybrid = { vectorWeight, textWeight: 1 - vectorWeight } as SemanticMemoryConfig["hybrid"];
  }

  // Pass through any already-nested keys (e.g. programmatic callers using the full shape)
  for (const key of ["search", "hybrid", "autoRecall", "autoCapture", "store", "cache", "chunking", "sync"] as const) {
    if (raw[key] !== undefined && config[key] === undefined) {
      (config as Record<string, unknown>)[key] = raw[key];
    }
  }

  return config;
}

// =============================================================================
// Plugin Export
// =============================================================================

const plugin: WOPRPlugin & {
  id: string;
  search(query: string, maxResults?: number): Promise<MemorySearchResult[]>;
  capture(text: string, source?: string): Promise<void>;
  getConfig(): SemanticMemoryConfig;
} = {
  id: "memory-semantic",
  name: "Semantic Memory",
  version: "1.0.0",
  description: "Semantic memory search with embeddings, auto-recall, and auto-capture",
  manifest: pluginManifest,

  async search(query: string, maxResults?: number): Promise<MemorySearchResult[]> {
    if (!state.searchManager) throw new Error("Semantic memory not initialized");
    return state.searchManager.search(query, maxResults, state.instanceId);
  },

  async capture(text: string, source = "manual"): Promise<void> {
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

  getConfig(): SemanticMemoryConfig {
    return { ...state.config };
  },

  /** Initialize (or re-initialize) the plugin. Cleans up prior registrations before applying new ones. */
  async init(api: WOPRPluginContext) {
    // Use new api's log from the start, but don't overwrite ctx yet —
    // old cleanup closures capture the module-level ctx by reference.
    // Running cleanups before ctx = api ensures they operate on the old context.
    const log = api.log;
    embeddingQueue.setLogger(log);
    log.info("[semantic-memory] init() called");

    // Clean up previous registrations if re-initialized.
    // IMPORTANT: run BEFORE overwriting ctx so closures still see the old context.
    for (let i = cleanups.length - 1; i >= 0; i--) {
      try {
        cleanups[i]();
      } catch (e) {
        log.warn(`[semantic-memory] re-init cleanup[${i}] threw: ${e instanceof Error ? e.message : String(e)}`);
      }
    }
    cleanups.length = 0;
    state.eventCleanup = [];

    ctx = api as PluginContext;
    const cleanupCtx = ctx; // capture for cleanup closures — prevents re-init from unregistering from wrong context

    // Register config schema
    ctx.registerConfigSchema("wopr-plugin-memory-semantic", pluginConfigSchema);
    cleanups.push(() => cleanupCtx.unregisterConfigSchema("wopr-plugin-memory-semantic"));

    // Register security permissions
    ctx.registerPermission?.("memory.read");
    ctx.registerPermission?.("memory.write");
    cleanups.push(() => {
      cleanupCtx.unregisterPermission?.("memory.read");
      cleanupCtx.unregisterPermission?.("memory.write");
    });

    // Register tool -> permission mappings.
    // Identity tool entries are sourced from identity-tools.ts to keep names in sync.
    const TOOL_PERMISSION_MAP: Array<[string, string]> = [
      ["memory_read", "memory.read"],
      ["memory_write", "memory.write"],
      ["memory_search", "memory.read"],
      ["memory_get", "memory.read"],
      ["self_reflect", "memory.write"],
      ...IDENTITY_TOOL_PERMISSION_MAP,
    ];
    for (const [tool, perm] of TOOL_PERMISSION_MAP) {
      ctx.registerToolPermission?.(tool, perm);
    }
    cleanups.push(() => {
      for (const [tool] of TOOL_PERMISSION_MAP) {
        cleanupCtx.unregisterToolPermission?.(tool);
      }
    });

    // Register context provider
    ctx.registerContextProvider(memoryContextProvider);
    cleanups.push(() => cleanupCtx.unregisterContextProvider("memory-semantic"));

    // Read config from WOPR central config (set by onboard wizard).
    // The wizard stores flat keys (e.g. autoRecallEnabled) but initialize()
    // expects nested SemanticMemoryConfig — map them before passing.
    const rawConfig = ctx.getConfig?.() as Record<string, unknown> | undefined;
    if (rawConfig?.apiKey) {
      log.warn(
        "[semantic-memory] Deprecated: 'apiKey' in config is ignored. Set OPENAI_API_KEY (or GEMINI_API_KEY) environment variable instead.",
      );
    }
    const storedConfig = rawConfig ? mapFlatConfigToNested(rawConfig) : undefined;
    await initialize(ctx, state, embeddingQueue, log, storedConfig);

    // Capture any cleanup functions registered during initialize() (e.g. unsubSessionDestroy)
    // before state.eventCleanup is replaced below.
    cleanups.push(...state.eventCleanup);

    if (!state.initialized) {
      log.error("[semantic-memory] Initialization failed — plugin will not activate");
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
      cleanups.push(() => cleanupCtx.unregisterExtension?.("memory-semantic"));
    }

    // Register WebMCP browser-side tools if the platform exposes a registry
    if (ctx.webmcpRegistry) {
      const registry = ctx.webmcpRegistry;
      registerWebMCPTools(registry, "/api", state.instanceId, (query, limit, instId) => {
        if (!state.searchManager) throw new Error("Semantic memory not initialized");
        return state.searchManager.search(query, limit, instId);
      });
      cleanups.push(() => {
        unregisterWebMCPTools(registry);
      });
      log.info("[semantic-memory] Registered WebMCP browser tools");
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
    const unsubSearch = ctx.events.on("memory:search", (payload: any) => handleMemorySearch(state, log, payload));

    cleanups.push(unsubBeforeInject, unsubAfterInject, unsubFilesChanged, unsubSearch);
    state.eventCleanup = [unsubBeforeInject, unsubAfterInject, unsubFilesChanged, unsubSearch];
    log.info("[semantic-memory] Plugin initialized - memory_search enhanced with semantic search");
  },

  /** Shut down the plugin, stop background work, and unregister all hooks and providers. */
  async shutdown() {
    if (!ctx) return; // Idempotent
    const shutdownLog = ctx.log;

    // Stop the embedding queue first
    await embeddingQueue.clear();

    // Stop file watcher
    await stopWatcher(shutdownLog);

    // Run all cleanup functions in reverse order (event unsubs, extension, context provider, configSchema)
    for (let i = cleanups.length - 1; i >= 0; i--) {
      try {
        cleanups[i]();
      } catch (e) {
        shutdownLog.warn(`[semantic-memory] cleanup[${i}] threw: ${e instanceof Error ? e.message : String(e)}`);
      }
    }
    cleanups.length = 0;

    // Unregister A2A tools
    unregisterMemoryTools(ctx);
    unregisterIdentityTools(ctx);

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
