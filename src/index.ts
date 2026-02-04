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
import { createSemanticSearchManager, type SemanticSearchManager, loadMtimeStore, saveMtimeStore } from "./search.js";
import { performAutoRecall, injectMemoriesIntoMessages } from "./recall.js";
import { shouldCapture, extractCaptureCandidate, extractFromConversation } from "./capture.js";
import { existsSync, readdirSync, readFileSync, statSync } from "fs";
import { join } from "path";
import { createHash } from "crypto";

/**
 * Produce a deterministic 12-hex identifier for a piece of text.
 *
 * @param text - The input content to hash
 * @returns A 12-character hexadecimal identifier derived from the MD5 hash of `text`
 */
function contentHash(text: string): string {
  return createHash('md5').update(text).digest('hex').slice(0, 12);
}

// =============================================================================
// Text Chunking
// =============================================================================

/**
 * Split input text into trimmed chunks while preserving contextual overlap.
 *
 * Prefers splitting at paragraph, sentence, or word boundaries when possible and falls back to fixed-size splits.
 *
 * @param text - The text to chunk
 * @param maxChars - Maximum characters per chunk (default 4000)
 * @param overlapChars - Number of characters to overlap between adjacent chunks (default 500)
 * @returns An array of non-empty, trimmed text chunks that include overlapping context between neighbors
 */
function chunkText(text: string, maxChars: number = 4000, overlapChars: number = 500): string[] {
  if (text.length <= maxChars) {
    return [text];
  }

  const chunks: string[] = [];
  let start = 0;

  while (start < text.length) {
    // Calculate end position
    let end = Math.min(start + maxChars, text.length);

    // If this is the last chunk, just take everything
    if (end >= text.length) {
      chunks.push(text.slice(start).trim());
      break;
    }

    // Try to find a good split point (paragraph, then sentence, then word)
    const searchStart = start + Math.floor(maxChars * 0.5);
    const searchText = text.slice(start, end);

    // Look for paragraph break (double newline) in the second half
    let splitAt = searchText.lastIndexOf("\n\n");
    if (splitAt > searchText.length * 0.5) {
      end = start + splitAt + 2;
    } else {
      // Look for single newline
      splitAt = searchText.lastIndexOf("\n");
      if (splitAt > searchText.length * 0.5) {
        end = start + splitAt + 1;
      } else {
        // Look for sentence end
        splitAt = searchText.lastIndexOf(". ");
        if (splitAt > searchText.length * 0.5) {
          end = start + splitAt + 2;
        } else {
          // Look for word boundary
          splitAt = searchText.lastIndexOf(" ");
          if (splitAt > searchText.length * 0.3) {
            end = start + splitAt + 1;
          }
          // Otherwise just split at maxChars
        }
      }
    }

    chunks.push(text.slice(start, end).trim());

    // Move start position back by overlap amount for context continuity
    start = end - overlapChars;
    if (start < 0) start = end; // Prevent negative start
  }

  return chunks.filter(c => c.length > 0);
}

// =============================================================================
// Plugin State
// =============================================================================

interface PluginState {
  config: SemanticMemoryConfig;
  embeddingProvider: EmbeddingProvider | null;
  searchManager: SemanticSearchManager | null;
  initialized: boolean;
  indexedFiles: Map<string, number>; // path -> mtime for incremental indexing
  lastIncrementalIndex: number; // timestamp of last incremental check
  startupIndexingComplete: boolean; // flag to skip incremental until startup done
}

const state: PluginState = {
  config: DEFAULT_CONFIG,
  embeddingProvider: null,
  searchManager: null,
  initialized: false,
  indexedFiles: new Map(),
  lastIncrementalIndex: 0,
  startupIndexingComplete: false,
};

// =============================================================================
// Plugin API Types (minimal interface with WOPR)
// =============================================================================

interface WoprPluginApi {
  // Hook registration - handlers can return mutated payloads (legacy)
  on(event: string, handler: (...args: any[]) => any): void;

  // Event bus for registering event handlers
  events: {
    on(event: string, handler: (...args: any[]) => any): () => void;
    off(event: string, handler: (...args: any[]) => any): void;
    emit(event: string, payload: any): Promise<void>;
  };

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
/**
 * Initialize the semantic memory plugin and prepare embedding and search providers.
 *
 * Merges `userConfig` with plugin defaults, creates an embedding provider and a
 * semantic search manager (optionally wiring core keyword search), loads persisted
 * file mtimes for incremental indexing, and sets internal initialization state.
 * If persisted vectors exist, marks startup indexing complete; otherwise kicks off
 * background startup indexing of existing memory files.
 *
 * @param userConfig - Partial configuration to override defaults; nested sections
 *   (`search`, `hybrid`, `autoRecall`, `autoCapture`, `store`, `cache`, `chunking`)
 *   are merged with their respective defaults.

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
            content: r.content || r.snippet || "",  // Use content if available, fallback to snippet
            textScore: r.score || 0,
          }));
        }
      : undefined;

    state.searchManager = await createSemanticSearchManager(
      state.config,
      state.embeddingProvider,
      keywordSearchFn
    );

    // Load persisted mtimes for incremental indexing
    state.indexedFiles = loadMtimeStore();
    api.log.info(`[semantic-memory] Loaded ${state.indexedFiles.size} file mtimes from persistent storage`);

    const loadedVectors = state.searchManager.getEntryCount();
    state.initialized = true;
    api.log.info(`[semantic-memory] Initialized with ${loadedVectors} persisted vectors`);

    // If we have persisted vectors, mark startup indexing as complete
    // Only run background indexing if we have no persisted data
    if (loadedVectors > 0) {
      state.startupIndexingComplete = true;
      state.lastIncrementalIndex = Date.now();
      api.log.info("[semantic-memory] Using persisted index, skipping startup indexing");
    } else {
      // Index existing memory files on startup (run in background to not block init)
      indexExistingMemoryFiles(api).catch((err) => {
        api.log.error(`[semantic-memory] Background indexing failed: ${err instanceof Error ? err.message : String(err)}`);
      });
    }
  } catch (err) {
    api.log.error(
      `[semantic-memory] Failed to initialize: ${err instanceof Error ? err.message : String(err)}`
    );
  }
}

// =============================================================================
// Startup Indexing
// =============================================================================

/**
 * Scans configured memory directories for markdown and session files, creates chunked memory entries, and batch-indexes them into the semantic search manager.
 *
 * Scanned files are recorded in plugin state for incremental indexing; the search manager is populated, state flags are updated, and file mtimes are persisted.
 *
 * @param api - Plugin API used for logging and access to core services
 */
async function indexExistingMemoryFiles(api: WoprPluginApi): Promise<void> {
  api.log.info("[semantic-memory] Starting startup indexing (BATCH MODE)...");

  if (!state.searchManager) {
    api.log.error("[semantic-memory] Startup indexing aborted: searchManager is null");
    return;
  }

  const memoryDirs = [
    "/data/identity",
    "/data/identity/memory",
    "/data/sessions",  // Session transcripts
  ];

  // Collect ALL entries first, then batch embed
  type PendingEntry = { entry: Omit<import("./search.js").VectorEntry, "embedding">; text: string };
  const allEntries: PendingEntry[] = [];
  const timestamp = Date.now();

  for (const dir of memoryDirs) {
    if (!existsSync(dir)) {
      api.log.info(`[semantic-memory] Directory does not exist, skipping: ${dir}`);
      continue;
    }

    try {
      const files = readdirSync(dir);
      const mdFiles = files.filter(f => f.endsWith(".md"));
      const jsonlFiles = files.filter(f => f.endsWith(".conversation.jsonl"));
      api.log.info(`[semantic-memory] Scanning ${dir}: ${mdFiles.length} .md, ${jsonlFiles.length} .jsonl`);

      for (const file of files) {
        const filePath = join(dir, file);
        let stat;
        try {
          stat = statSync(filePath);
        } catch {
          continue;
        }
        if (!stat.isFile()) continue;

        // Track mtime for incremental indexing
        state.indexedFiles.set(filePath, stat.mtimeMs);

        try {
          // Handle markdown files
          if (file.endsWith(".md")) {
            const content = readFileSync(filePath, "utf-8");
            if (content.length < 10) continue;

            const chunks = chunkText(content, 4000, 500);
            for (let i = 0; i < chunks.length; i++) {
              const chunk = chunks[i];
              allEntries.push({
                entry: {
                  id: `mem-${contentHash(chunk)}`,
                  path: filePath,
                  startLine: i * 100,
                  endLine: (i + 1) * 100,
                  source: "memory",
                  snippet: chunk.slice(0, 500),
                  content: chunk,
                },
                text: chunk,
              });
            }
          }
          // Handle JSONL conversation logs - index each message individually
          else if (file.endsWith(".conversation.jsonl")) {
            const content = readFileSync(filePath, "utf-8");
            const lines = content.split("\n").filter(l => l.trim());

            let msgNum = 0;
            for (const line of lines) {
              try {
                const entry = JSON.parse(line);
                if (entry.content && entry.content.trim().length > 0) {
                  allEntries.push({
                    entry: {
                      id: `msg-${contentHash(entry.content)}`,
                      path: filePath,
                      startLine: msgNum,
                      endLine: msgNum,
                      source: "sessions",
                      snippet: entry.content.slice(0, 500),
                      content: entry.content,
                    },
                    text: entry.content,
                  });
                  msgNum++;
                }
              } catch {}
            }
          }
        } catch (err) {
          api.log.error(`[semantic-memory] Failed to read ${filePath}: ${err}`);
        }
      }
    } catch (err) {
      api.log.error(`[semantic-memory] Failed to scan ${dir}: ${err}`);
    }
  }

  // Now batch embed all entries (up to 2000 per API call)
  api.log.info(`[semantic-memory] Collected ${allEntries.length} chunks, starting batch embedding...`);

  const startTime = Date.now();
  const indexedCount = await state.searchManager.addEntriesBatch(allEntries);
  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

  state.lastIncrementalIndex = Date.now();
  state.startupIndexingComplete = true;

  // Save mtimes to persistent storage
  saveMtimeStore(state.indexedFiles);

  if (indexedCount > 0) {
    api.log.info(`[semantic-memory] Indexed ${indexedCount} entries in ${elapsed}s (batch mode)`);
  }
  api.log.info(`[semantic-memory] Startup indexing complete, incremental indexing enabled`);
}

/**
 * Indexes new or changed memory and session files since the last check.
 *
 * Scans configured memory directories for modified Markdown and conversation JSONL files, adds any new or updated chunks/messages to the semantic search manager, and updates internal indexing state (mtimes and last-check timestamp). The function is a no-op if the search manager is unavailable or startup indexing has not completed, and it is rate-limited to avoid frequent scans.
 *
 * @param api - The WOPR plugin API used for logging and access to core services
 * @returns The number of newly indexed entries during this run
 */
async function incrementalIndexFiles(api: WoprPluginApi): Promise<number> {
  if (!state.searchManager) return 0;

  // Skip incremental indexing until startup indexing is complete
  // This prevents blocking searches while re-indexing all files on first search
  if (!state.startupIndexingComplete) {
    return 0;
  }

  // Throttle: don't check more than once per 5 seconds
  const now = Date.now();
  if (now - state.lastIncrementalIndex < 5000) {
    return 0;
  }

  const memoryDirs = [
    "/data/identity",
    "/data/identity/memory",
    "/data/sessions",
  ];

  let indexedCount = 0;

  for (const dir of memoryDirs) {
    if (!existsSync(dir)) continue;

    try {
      const files = readdirSync(dir);
      for (const file of files) {
        const filePath = join(dir, file);

        let stat;
        try {
          stat = statSync(filePath);
        } catch {
          continue; // File may have been deleted
        }

        if (!stat.isFile()) continue;

        const mtime = stat.mtimeMs;
        const lastMtime = state.indexedFiles.get(filePath);

        // Skip if file hasn't changed
        if (lastMtime !== undefined && mtime <= lastMtime) {
          continue;
        }

        // Log what file changed
        api.log.info(`[semantic-memory] File changed: ${file} (mtime: ${lastMtime} -> ${mtime})`);

        // Update tracked mtime
        state.indexedFiles.set(filePath, mtime);

        try {
          // Handle markdown files
          if (file.endsWith(".md")) {
            const content = readFileSync(filePath, "utf-8");
            if (content.length < 10) continue;

            const chunks = chunkText(content, 4000, 500);
            for (let i = 0; i < chunks.length; i++) {
              const chunk = chunks[i];
              const id = `mem-${contentHash(chunk)}`;
              await state.searchManager!.addEntry(
                {
                  id,
                  path: filePath,
                  startLine: i * 100,
                  endLine: (i + 1) * 100,
                  source: "memory",
                  snippet: chunk.slice(0, 500),
                  content: chunk,
                },
                chunk
              );
              indexedCount++;
            }
          }
          // Handle JSONL conversation logs - index each message individually for stability
          else if (file.endsWith(".conversation.jsonl")) {
            const content = readFileSync(filePath, "utf-8");
            const lines = content.split("\n").filter(l => l.trim());

            let msgNum = 0;
            for (const line of lines) {
              try {
                const entry = JSON.parse(line);
                if (entry.content && entry.content.trim().length > 0) {
                  // Index each message individually - stable IDs even as file grows
                  const id = `msg-${contentHash(entry.content)}`;
                  // Skip if already indexed (content hash ensures dedup)
                  if (state.searchManager!.hasEntry(id)) continue;

                  await state.searchManager!.addEntry(
                    {
                      id,
                      path: filePath,
                      startLine: msgNum,
                      endLine: msgNum,
                      source: "sessions",
                      snippet: entry.content.slice(0, 500),
                      content: entry.content,
                    },
                    entry.content
                  );
                  indexedCount++;
                  msgNum++;
                }
              } catch {}
            }
          }
        } catch (err) {
          api.log.error(`[semantic-memory] Failed to incremental index ${filePath}: ${err}`);
        }
      }
    } catch (err) {
      api.log.error(`[semantic-memory] Failed to scan ${dir} for incremental: ${err}`);
    }
  }

  state.lastIncrementalIndex = now;

  if (indexedCount > 0) {
    api.log.info(`[semantic-memory] Incremental indexed ${indexedCount} new entries`);
  }

  return indexedCount;
}

// =============================================================================
// Hook Handlers
// =============================================================================

/**
 * Initiates an automatic memory recall before a message is injected into a session.
 *
 * When the plugin is initialized and auto-recall is enabled, examines the provided
 * session inject payload for a user message, runs a semantic recall against the
 * configured search manager, and logs any recalled memories or errors.
 *
 * @param payload - The session inject event payload; expected to contain a `message` string (may also include `session`, `from`, and optional `channel`). If `message` is missing or empty, no action is taken.
 */
async function handleBeforeInject(
  api: WoprPluginApi,
  payload: any
): Promise<void> {
  if (!state.initialized || !state.searchManager || !state.config.autoRecall.enabled) {
    return;
  }

  // Payload is SessionInjectEvent: { session, message, from, channel? }
  // Not the expected messages array - skip for now until payload interface is resolved
  if (!payload || typeof payload.message !== "string" || !payload.message.trim()) {
    return;
  }

  const lastUserMessage = { role: "user", content: payload.message };

  try {
    const recall = await performAutoRecall(
      lastUserMessage.content,
      state.searchManager,
      state.config
    );

    if (recall && recall.memories.length > 0) {
      api.log.info(
        `[semantic-memory] Recalled ${recall.memories.length} memories for: "${recall.query.slice(0, 50)}..."`
      );
      // Note: Can't inject memories through event bus - would need mutable payload support
      // For now, just log that we found relevant memories
    }
  } catch (err) {
    api.log.error(
      `[semantic-memory] Auto-recall failed: ${err instanceof Error ? err.message : String(err)}`
    );
  }
}

/**
 * Indexes the current session's user message and assistant response into semantic memory and runs optional auto-capture.
 *
 * When invoked, this hook will chunk and add the session's user message and assistant response to the search manager,
 * logging the number of indexed chunks. If auto-capture is enabled it will also extract and store capture candidates
 * from the two-turn exchange. Short or empty responses are ignored.
 *
 * @param payload - SessionResponseEvent with shape `{ session?: string, message?: string, response: string, from?: string }`.
 *   Only `response` is required; `message` and `session` are used when present. Messages shorter than ~10 characters are skipped.
 */
async function handleAfterInject(
  api: WoprPluginApi,
  payload: any
): Promise<void> {
  if (!state.initialized || !state.searchManager) {
    return;
  }

  // Validate payload structure
  if (!payload || typeof payload.response !== "string" || !payload.response.trim()) {
    return;
  }

  const sessionName = payload.session || "unknown";
  let indexedCount = 0;

  try {
    // REAL-TIME INDEXING: Index ALL session content immediately with full text
    // Index user message if present
    if (payload.message && payload.message.trim().length > 10) {
      const userChunks = chunkText(payload.message, 4000, 500);
      for (let i = 0; i < userChunks.length; i++) {
        const chunk = userChunks[i];
        const id = `rt-${contentHash(chunk)}`;
        await state.searchManager.addEntry(
          {
            id,
            path: `session:${sessionName}`,
            startLine: 0,
            endLine: 0,
            source: "realtime-user",
            snippet: chunk.slice(0, 500),
            content: chunk,
          },
          chunk
        );
        indexedCount++;
      }
    }

    // Index assistant response
    if (payload.response.trim().length > 10) {
      const responseChunks = chunkText(payload.response, 4000, 500);
      for (let i = 0; i < responseChunks.length; i++) {
        const chunk = responseChunks[i];
        const id = `rt-${contentHash(chunk)}`;
        await state.searchManager.addEntry(
          {
            id,
            path: `session:${sessionName}`,
            startLine: 0,
            endLine: 0,
            source: "realtime-assistant",
            snippet: chunk.slice(0, 500),
            content: chunk,
          },
          chunk
        );
        indexedCount++;
      }
    }

    if (indexedCount > 0) {
      api.log.info(`[semantic-memory] Real-time indexed ${indexedCount} chunks from session ${sessionName}`);
    }

    // ALSO run capture analysis for important content (if enabled)
    if (state.config.autoCapture.enabled) {
      const messages = [
        { role: "user" as const, content: payload.message || "" },
        { role: "assistant" as const, content: payload.response },
      ];

      const candidates = extractFromConversation(messages, state.config);

      if (candidates.length > 0) {
        api.log.info(`[semantic-memory] Found ${candidates.length} capture candidates`);

        for (const candidate of candidates) {
          const id = `cap-${contentHash(candidate.text)}`;
          await state.searchManager.addEntry(
            {
              id,
              path: `session:${sessionName}`,
              startLine: 0,
              endLine: 0,
              source: "auto-capture",
              snippet: candidate.text.slice(0, 500),
              content: candidate.text,
            },
            candidate.text
          );
        }

        api.log.info(`[semantic-memory] Captured ${candidates.length} memories`);
      }
    }
  } catch (err) {
    api.log.error(
      `[semantic-memory] Real-time indexing failed: ${err instanceof Error ? err.message : String(err)}`
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

  init(api: WoprPluginApi, config?: Partial<SemanticMemoryConfig>): Promise<void>;
  shutdown(): Promise<void>;

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

  async init(api: WoprPluginApi, config?: Partial<SemanticMemoryConfig>) {
    api.log.info("[semantic-memory] init() called");
    await initialize(api, config);

    // Register hooks via the event bus
    api.events.on("session:beforeInject", (payload: any) => handleBeforeInject(api, payload));
    api.events.on("session:afterInject", (payload: any) => handleAfterInject(api, payload));

    // Hook into memory_search to provide semantic results
    api.events.on("memory:search", async (payload: {
      query: string;
      maxResults: number;
      minScore: number;
      sessionName: string;
      results: any[] | null;
    }) => {
      api.log.info(`[semantic-memory] memory:search handler called for: "${payload.query}"`);

      if (!state.initialized || !state.searchManager) {
        api.log.info(`[semantic-memory] Not initialized, skipping (initialized=${state.initialized})`);
        return;
      }

      try {
        // Incremental index check before search (catches file-based changes)
        // Real-time session content is already indexed via session:afterInject
        const newEntries = await incrementalIndexFiles(api);
        if (newEntries > 0) {
          api.log.info(`[semantic-memory] Pre-search incremental index added ${newEntries} entries`);
        }

        api.log.info(`[semantic-memory] Starting semantic search...`);
        const results = await state.searchManager.search(payload.query, payload.maxResults);
        api.log.info(`[semantic-memory] Raw results: ${results.length}`);
        payload.results = results.filter((r) => r.score >= payload.minScore);
        api.log.info(`[semantic-memory] After filter: ${payload.results.length} results (minScore: ${payload.minScore})`);
      } catch (err) {
        api.log.error(`[semantic-memory] Search failed: ${err instanceof Error ? err.message : String(err)}`);
      }
    });

    api.log.info("[semantic-memory] Plugin initialized - memory_search enhanced with semantic search");
  },

  async shutdown() {
    // Save mtimes before shutdown
    if (state.indexedFiles.size > 0) {
      saveMtimeStore(state.indexedFiles);
    }
    if (state.searchManager) {
      await state.searchManager.close();
      state.searchManager = null;
    }
    state.embeddingProvider = null;
    state.initialized = false;
    state.startupIndexingComplete = false;
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
    const id = `man-${contentHash(text)}`;

    await state.searchManager.addEntry(
      {
        id,
        path: source,
        startLine: 0,
        endLine: 0,
        source,
        snippet: text.slice(0, 500),
        content: text,
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