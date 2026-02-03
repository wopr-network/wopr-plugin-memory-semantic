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

import type { SemanticMemoryConfig, EmbeddingProvider, MemorySearchResult } from "./types.js";
import { DEFAULT_CONFIG } from "./types.js";
import { createEmbeddingProvider } from "./embeddings.js";
import { createSemanticSearchManager, type SemanticSearchManager } from "./search.js";
import { performAutoRecall, injectMemoriesIntoMessages } from "./recall.js";
import { shouldCapture, extractCaptureCandidate, extractFromConversation } from "./capture.js";

// =============================================================================
// Plugin State
// =============================================================================

interface PluginState {
  config: SemanticMemoryConfig;
  embeddingProvider: EmbeddingProvider | null;
  searchManager: SemanticSearchManager | null;
  initialized: boolean;
}

const state: PluginState = {
  config: DEFAULT_CONFIG,
  embeddingProvider: null,
  searchManager: null,
  initialized: false,
};

// =============================================================================
// Plugin API Types (minimal interface with WOPR)
// =============================================================================

interface WoprPluginApi {
  // Hook registration - handlers can return mutated payloads
  on(event: string, handler: (...args: any[]) => any): void;

  // Memory tools (provided by core)
  memory?: {
    keywordSearch?(query: string, limit: number): Promise<any[]>;
    getManager?(): any;
  };

  // A2A tool registration
  registerA2AServer?(config: {
    name: string;
    description: string;
    tools: Array<{
      name: string;
      description: string;
      inputSchema: Record<string, any>;
      handler: (args: any) => Promise<any>;
    }>;
  }): void;

  // Logging
  log: {
    info(msg: string): void;
    error(msg: string): void;
    debug(msg: string): void;
  };
}

// =============================================================================
// Initialization
// =============================================================================

async function initialize(api: WoprPluginApi, userConfig?: Partial<SemanticMemoryConfig>) {
  if (state.initialized) return;

  // Merge user config with defaults
  state.config = {
    ...DEFAULT_CONFIG,
    ...userConfig,
    search: { ...DEFAULT_CONFIG.search, ...userConfig?.search },
    hybrid: { ...DEFAULT_CONFIG.hybrid, ...userConfig?.hybrid },
    autoRecall: { ...DEFAULT_CONFIG.autoRecall, ...userConfig?.autoRecall },
    autoCapture: { ...DEFAULT_CONFIG.autoCapture, ...userConfig?.autoCapture },
    store: { ...DEFAULT_CONFIG.store, ...userConfig?.store },
    cache: { ...DEFAULT_CONFIG.cache, ...userConfig?.cache },
    chunking: { ...DEFAULT_CONFIG.chunking, ...userConfig?.chunking },
  };

  try {
    // Create embedding provider
    state.embeddingProvider = await createEmbeddingProvider(state.config);
    api.log.info(`[semantic-memory] Embedding provider: ${state.embeddingProvider.id}`);

    // Create search manager
    // Wire up keyword search from core if available
    const keywordSearchFn = api.memory?.keywordSearch
      ? async (query: string, limit: number) => {
          const results = await api.memory!.keywordSearch!(query, limit);
          return results.map((r: any) => ({
            id: r.id || `${r.path}:${r.startLine}`,
            path: r.path,
            startLine: r.startLine || 0,
            endLine: r.endLine || 0,
            source: r.source || "memory",
            snippet: r.snippet || r.content || "",
            textScore: r.score || 0,
          }));
        }
      : undefined;

    state.searchManager = await createSemanticSearchManager(
      state.config,
      state.embeddingProvider,
      keywordSearchFn
    );

    state.initialized = true;
    api.log.info("[semantic-memory] Initialized");
  } catch (err) {
    api.log.error(
      `[semantic-memory] Failed to initialize: ${err instanceof Error ? err.message : String(err)}`
    );
  }
}

// =============================================================================
// Hook Handlers
// =============================================================================

/**
 * Before inject hook - auto-recall relevant memories
 */
async function handleBeforeInject(
  api: WoprPluginApi,
  payload: {
    sessionName: string;
    messages: Array<{ role: string; content: string }>;
  }
): Promise<{ messages: Array<{ role: string; content: string }> } | void> {
  if (!state.initialized || !state.searchManager || !state.config.autoRecall.enabled) {
    return;
  }

  // Find the last user message
  const lastUserMessage = payload.messages.filter((m) => m.role === "user").pop();
  if (!lastUserMessage) {
    return;
  }

  try {
    const recall = await performAutoRecall(
      lastUserMessage.content,
      state.searchManager,
      state.config
    );

    if (recall && recall.memories.length > 0) {
      api.log.debug(
        `[semantic-memory] Recalled ${recall.memories.length} memories for: "${recall.query.slice(0, 50)}..."`
      );

      // Inject memories into messages
      const enhancedMessages = injectMemoriesIntoMessages(payload.messages, recall);
      return { messages: enhancedMessages };
    }
  } catch (err) {
    api.log.error(
      `[semantic-memory] Auto-recall failed: ${err instanceof Error ? err.message : String(err)}`
    );
  }
}

/**
 * After inject hook - auto-capture important information
 */
async function handleAfterInject(
  api: WoprPluginApi,
  payload: {
    sessionName: string;
    messages: Array<{ role: string; content: string }>;
    response?: string;
  }
): Promise<void> {
  if (!state.initialized || !state.searchManager || !state.config.autoCapture.enabled) {
    return;
  }

  try {
    // Extract capturable content from the conversation
    const candidates = extractFromConversation(payload.messages, state.config);

    if (candidates.length === 0) {
      return;
    }

    api.log.debug(`[semantic-memory] Found ${candidates.length} capture candidates`);

    // Store each candidate
    for (const candidate of candidates) {
      const id = `capture-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      await state.searchManager.addEntry(
        {
          id,
          path: `session:${payload.sessionName}`,
          startLine: 0,
          endLine: 0,
          source: "auto-capture",
          snippet: candidate.text.slice(0, 200),
        },
        candidate.text
      );
    }

    api.log.info(`[semantic-memory] Captured ${candidates.length} memories`);
  } catch (err) {
    api.log.error(
      `[semantic-memory] Auto-capture failed: ${err instanceof Error ? err.message : String(err)}`
    );
  }
}

// =============================================================================
// Plugin Export
// =============================================================================

export interface SemanticMemoryPlugin {
  id: string;
  name: string;
  description: string;
  version: string;

  register(api: WoprPluginApi, config?: Partial<SemanticMemoryConfig>): Promise<void>;
  unregister(): Promise<void>;

  // Public API
  search(query: string, maxResults?: number): Promise<MemorySearchResult[]>;
  capture(text: string, source?: string): Promise<void>;
  getConfig(): SemanticMemoryConfig;
}

const plugin: SemanticMemoryPlugin = {
  id: "memory-semantic",
  name: "Semantic Memory",
  description: "Semantic memory search with embeddings, auto-recall, and auto-capture",
  version: "1.0.0",

  async register(api: WoprPluginApi, config?: Partial<SemanticMemoryConfig>) {
    await initialize(api, config);

    // Register hooks
    api.on("session:beforeInject", (payload: any) => handleBeforeInject(api, payload));
    api.on("session:afterInject", (payload: any) => handleAfterInject(api, payload));

    // Register A2A tools
    if (api.registerA2AServer) {
      api.registerA2AServer({
        name: "semantic-memory",
        description: "Semantic memory search with vector embeddings",
        tools: [
          {
            name: "memory_search_semantic",
            description: "Search memory using semantic/vector similarity. More accurate than keyword search for finding conceptually related content.",
            inputSchema: {
              type: "object",
              properties: {
                query: {
                  type: "string",
                  description: "Search query - finds semantically similar content",
                },
                maxResults: {
                  type: "number",
                  description: "Maximum results (default: 10)",
                },
                minScore: {
                  type: "number",
                  description: "Minimum relevance score 0-1 (default: 0.35)",
                },
              },
              required: ["query"],
            },
            handler: async (args: { query: string; maxResults?: number; minScore?: number }) => {
              if (!state.searchManager) {
                return { content: [{ type: "text", text: "Semantic memory not initialized" }] };
              }

              const { query, maxResults = 10, minScore = 0.35 } = args;

              try {
                const results = await state.searchManager.search(query, maxResults);
                const filtered = results.filter((r) => r.score >= minScore);

                if (filtered.length === 0) {
                  return { content: [{ type: "text", text: `No semantic matches found for "${query}"` }] };
                }

                const formatted = filtered
                  .map(
                    (r, i) =>
                      `[${i + 1}] ${r.source}/${r.path}:${r.startLine}-${r.endLine} (score: ${r.score.toFixed(2)})\n${r.snippet}`
                  )
                  .join("\n\n---\n\n");

                return {
                  content: [{ type: "text", text: `Found ${filtered.length} semantic matches:\n\n${formatted}` }],
                };
              } catch (err) {
                const message = err instanceof Error ? err.message : String(err);
                return { content: [{ type: "text", text: `Semantic search failed: ${message}` }] };
              }
            },
          },
          {
            name: "memory_capture",
            description: "Manually capture text to semantic memory for later recall.",
            inputSchema: {
              type: "object",
              properties: {
                text: {
                  type: "string",
                  description: "Text to capture and store",
                },
                source: {
                  type: "string",
                  description: "Source identifier (default: 'manual')",
                },
              },
              required: ["text"],
            },
            handler: async (args: { text: string; source?: string }) => {
              if (!state.searchManager) {
                return { content: [{ type: "text", text: "Semantic memory not initialized" }] };
              }

              const { text, source = "manual" } = args;

              try {
                const id = `manual-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
                await state.searchManager.addEntry(
                  {
                    id,
                    path: source,
                    startLine: 0,
                    endLine: 0,
                    source,
                    snippet: text.slice(0, 200),
                  },
                  text
                );

                return { content: [{ type: "text", text: `Captured to semantic memory: "${text.slice(0, 100)}..."` }] };
              } catch (err) {
                const message = err instanceof Error ? err.message : String(err);
                return { content: [{ type: "text", text: `Capture failed: ${message}` }] };
              }
            },
          },
        ],
      });
      api.log.info("[semantic-memory] Registered A2A tools: memory_search_semantic, memory_capture");
    }

    api.log.info("[semantic-memory] Plugin registered");
  },

  async unregister() {
    if (state.searchManager) {
      await state.searchManager.close();
      state.searchManager = null;
    }
    state.embeddingProvider = null;
    state.initialized = false;
  },

  async search(query: string, maxResults?: number): Promise<MemorySearchResult[]> {
    if (!state.searchManager) {
      throw new Error("Semantic memory not initialized");
    }
    return state.searchManager.search(query, maxResults);
  },

  async capture(text: string, source = "manual"): Promise<void> {
    if (!state.searchManager) {
      throw new Error("Semantic memory not initialized");
    }

    const candidate = extractCaptureCandidate(text);
    const id = `manual-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

    await state.searchManager.addEntry(
      {
        id,
        path: source,
        startLine: 0,
        endLine: 0,
        source,
        snippet: text.slice(0, 200),
      },
      text
    );
  },

  getConfig(): SemanticMemoryConfig {
    return { ...state.config };
  },
};

export default plugin;

// Re-export types
export type { SemanticMemoryConfig, EmbeddingProvider, MemorySearchResult } from "./types.js";
export { DEFAULT_CONFIG } from "./types.js";
