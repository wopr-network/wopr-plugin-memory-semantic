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
import { performAutoRecall } from "./recall.js";
import { extractFromConversation } from "./capture.js";
import { createHash } from "crypto";
import winston from "winston";

const log = winston.createLogger({
  level: "debug",
  format: winston.format.combine(
    winston.format.timestamp({ format: "HH:mm:ss.SSS" }),
    winston.format.printf(({ timestamp, level, message }) =>
      `${timestamp} [semantic-memory] ${level}: ${message}`
    ),
  ),
  transports: [new winston.transports.Console()],
});

// Generate deterministic ID from content to avoid duplicates
function contentHash(text: string): string {
  return createHash('sha256').update(text).digest('hex').slice(0, 24);
}

// =============================================================================
// Plugin State
// =============================================================================

interface PluginState {
  config: SemanticMemoryConfig;
  embeddingProvider: EmbeddingProvider | null;
  searchManager: SemanticSearchManager | null;
  api: WoprPluginApi | null;
  initialized: boolean;
  indexedFiles: Map<string, number>; // path -> mtime for incremental indexing
  lastIncrementalIndex: number; // timestamp of last incremental check
  startupIndexingComplete: boolean; // flag to skip incremental until startup done
}

const state: PluginState = {
  config: DEFAULT_CONFIG,
  embeddingProvider: null,
  searchManager: null,
  api: null,
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

  // Extension system - discover APIs from core and other plugins
  getExtension<T = unknown>(name: string): T | undefined;

  // Memory tools (provided by core — may be undefined)
  memory?: {
    keywordSearch?(query: string, limit: number): Promise<any[]>;
  };

  // Plugin config from WOPR central config (plugins.data[pluginName])
  getConfig?<T = any>(): T;
  saveConfig?<T = any>(config: T): Promise<void>;

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
// SQLite Embedding Persistence (via core's memory:db extension)
// =============================================================================

function getDb(api: WoprPluginApi): any | null {
  return api.getExtension<any>('memory:db') ?? null;
}

/** Ensure the embedding column exists on the chunks table (plugin owns this column) */
function ensureEmbeddingColumn(db: any): void {
  try {
    const cols = db.prepare(`PRAGMA table_info(chunks)`).all() as Array<{ name: string }>;
    if (!cols.some((c: any) => c.name === 'embedding')) {
      db.exec(`ALTER TABLE chunks ADD COLUMN embedding BLOB`);
    }
  } catch {}
}

/** Open a read-only DB handle — tries core extension first, falls back to direct file open. */
async function openDbForRead(api: WoprPluginApi): Promise<{ db: any; owned: boolean } | null> {
  const db = getDb(api);
  if (db) return { db, owned: false };

  const home = process.env.WOPR_HOME;
  const dbPath = home ? `${home}/memory/index.sqlite` : undefined;
  if (!dbPath) return null;
  try {
    const { DatabaseSync } = await import("node:sqlite");
    const handle = new DatabaseSync(dbPath, { readOnly: true });
    log.info(`Opened direct read-only DB: ${dbPath}`);
    return { db: handle, owned: true };
  } catch (err) {
    log.warn(`Cannot open DB directly: ${err instanceof Error ? err.message : err}`);
    return null;
  }
}

/** Load chunk metadata from SQLite — used to reconstruct the metadata Map
 *  when loading HNSW from disk. Embeddings are NOT in SQLite (they live in
 *  the HNSW binary), so we return a dummy empty embedding. */
async function loadChunkMetadata(api: WoprPluginApi): Promise<Map<string, import("./search.js").VectorEntry>> {
  const handle = await openDbForRead(api);
  if (!handle) return new Map();

  const { db, owned } = handle;
  const entries = new Map<string, import("./search.js").VectorEntry>();

  try {
    const stmt = db.prepare(
      `SELECT id, path, start_line, end_line, source, text FROM chunks`
    );
    for (const row of stmt.iterate()) {
      const r = row as any;
      const text = typeof r.text === 'string' ? r.text : '';
      const entry: import("./search.js").VectorEntry = {
        id: r.id,
        path: r.path,
        startLine: r.start_line,
        endLine: r.end_line,
        source: r.source,
        snippet: text.slice(0, 500),
        content: text,
        embedding: [], // placeholder — real vectors live in HNSW index
      };
      entries.set(r.id, entry);
    }
    log.info(`Loaded ${entries.size} chunk metadata rows from SQLite`);
  } catch (err) {
    log.error(`Failed to load chunk metadata: ${err}`);
  } finally {
    if (owned) { try { db.close(); } catch { /* ignore */ } }
  }
  return entries;
}

/** Derive path for the persisted HNSW binary.
 *  Tries the DB handle first, falls back to WOPR_HOME convention. */
function getHnswPath(api: WoprPluginApi): string | undefined {
  // Try DB handle (available after core memory init)
  const db = getDb(api);
  if (db) {
    const dbPath: unknown = typeof db.location === 'function' ? db.location() : db.name;
    if (typeof dbPath === 'string' && dbPath && dbPath !== ':memory:') {
      return dbPath + '.hnsw';
    }
  }
  // Fallback: WOPR_HOME convention (always available)
  const home = process.env.WOPR_HOME;
  if (home) return `${home}/memory/index.sqlite.hnsw`;
  return undefined;
}

function storeEmbeddingsToDb(
  api: WoprPluginApi,
  embeddings: Array<{ id: string; embedding: number[] }>
): void {
  const db = getDb(api);
  if (!db || embeddings.length === 0) return;
  ensureEmbeddingColumn(db);

  db.exec("BEGIN");
  try {
    const update = db.prepare(
      `UPDATE chunks SET embedding = ? WHERE id = ?`
    );
    for (const entry of embeddings) {
      const blob = Buffer.from(new Float32Array(entry.embedding).buffer);
      update.run(blob, entry.id);
    }
    db.exec("COMMIT");
  } catch (err) {
    db.exec("ROLLBACK");
    log.error(`Failed to store embeddings: ${err}`);
  }
}

/** INSERT a plugin-originated entry (real-time, capture) into chunks with its embedding */
function persistNewEntryToDb(api: WoprPluginApi, id: string): void {
  const entry = state.searchManager?.getEntry(id);
  if (!entry || !state.embeddingProvider) return;

  const db = getDb(api);
  if (!db) return;
  ensureEmbeddingColumn(db);

  try {
    const blob = Buffer.from(new Float32Array(entry.embedding).buffer);
    db.prepare(
      `INSERT OR IGNORE INTO chunks (id, path, source, start_line, end_line, hash, model, text, updated_at, embedding)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    ).run(
      entry.id, entry.path, entry.source, entry.startLine, entry.endLine,
      contentHash(entry.content), state.embeddingProvider.id, entry.content, Date.now(), blob
    );
  } catch {}
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
    chunking: {
      ...DEFAULT_CONFIG.chunking,
      ...userConfig?.chunking,
      multiScale: userConfig?.chunking?.multiScale ?? DEFAULT_CONFIG.chunking.multiScale,
    },
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

    // Load chunk metadata from SQLite (for HNSW metadata reconstruction).
    // Embeddings in SQLite are stale/empty — the HNSW binary is the vector source of truth.
    const chunkMetadata = await loadChunkMetadata(api);
    // Lazy resolver: DB extension may not be available at init
    const hnswPathFn = () => getHnswPath(api);

    state.searchManager = await createSemanticSearchManager(
      state.config,
      state.embeddingProvider,
      keywordSearchFn,
      chunkMetadata,
      hnswPathFn
    );

    const loadedVectors = state.searchManager.getEntryCount();
    state.api = api;
    state.initialized = true;
    api.log.info(`[semantic-memory] Initialized with ${loadedVectors} persisted vectors`);

    // Bootstrap: if HNSW is empty but we have chunk metadata, embed them now
    if (loadedVectors === 0 && chunkMetadata.size > 0) {
      api.log.info(`[semantic-memory] Bootstrap: ${chunkMetadata.size} chunks need embedding, starting async...`);
      bootstrapEmbeddings(chunkMetadata).catch((err) => {
        api.log.error(`[semantic-memory] Bootstrap failed: ${err instanceof Error ? err.message : err}`);
      });
    } else {
      api.log.info(`[semantic-memory] Initialized — waiting for memory:filesChanged events from core`);
    }
  } catch (err) {
    api.log.error(
      `[semantic-memory] Failed to initialize: ${err instanceof Error ? err.message : String(err)}`
    );
  }
}

// =============================================================================
// Bootstrap: embed existing chunks that have no HNSW vectors
// =============================================================================

async function bootstrapEmbeddings(
  chunkMetadata: Map<string, import("./search.js").VectorEntry>
): Promise<void> {
  const heapMB = () => Math.round(process.memoryUsage().heapUsed / 1024 / 1024);
  log.info(`Bootstrap start (heap=${heapMB()}MB)`);
  if (!state.searchManager || !state.embeddingProvider) return;

  const batch: PendingEntry[] = [];
  for (const [id, entry] of chunkMetadata) {
    if (state.searchManager.hasEntry(id)) continue;
    if (!entry.content || entry.content.length < 10) continue;
    batch.push({
      entry: {
        id: entry.id,
        path: entry.path,
        startLine: entry.startLine,
        endLine: entry.endLine,
        source: entry.source,
        snippet: entry.snippet,
        content: entry.content,
      },
      text: entry.content,
    });
  }

  if (batch.length === 0) {
    log.info("Bootstrap: all chunks already indexed");
    return;
  }

  log.info(`Bootstrap: embedding ${batch.length} chunks via ${state.embeddingProvider.id} (heap=${heapMB()}MB)`);
  const count = await state.searchManager.addEntriesBatch(batch);
  log.info(`Bootstrap complete: ${count} vectors added to HNSW (heap=${heapMB()}MB)`);
}

// =============================================================================
// Hook Handlers
// =============================================================================

/**
 * Before inject hook - auto-recall relevant memories
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
      // Prepend memory context to the mutable message payload
      // Core uses emitMutableIncoming → payload.message is mutable
      payload.message = `${recall.context}\n\n${payload.message}`;
    }
  } catch (err) {
    api.log.error(
      `[semantic-memory] Auto-recall failed: ${err instanceof Error ? err.message : String(err)}`
    );
  }
}

/**
 * After inject hook - real-time indexing of ALL session content
 * Payload is SessionResponseEvent: { session, message, response, from }
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
    const ms = state.config.chunking.multiScale;

    // Helper: index a text with optional multi-scale
    const indexText = async (text: string, baseId: string, source: string) => {
      if (ms?.enabled && ms.scales.length > 0) {
        const subChunks = multiScaleChunk(
          text, baseId,
          { path: `session:${sessionName}`, startLine: 0, endLine: 0, source },
          ms.scales
        );
        for (const sc of subChunks) {
          if (!state.searchManager!.hasEntry(sc.entry.id)) {
            await state.searchManager!.addEntry(sc.entry, sc.text);
            persistNewEntryToDb(api, sc.entry.id);
            indexedCount++;
          }
        }
      } else {
        await state.searchManager!.addEntry(
          { id: baseId, path: `session:${sessionName}`, startLine: 0, endLine: 0, source, snippet: text.slice(0, 500), content: text },
          text
        );
        persistNewEntryToDb(api, baseId);
        indexedCount++;
      }
    };

    // REAL-TIME INDEXING: Index session content immediately with full text
    if (payload.message && payload.message.trim().length > 10) {
      await indexText(payload.message, `rt-${contentHash(payload.message)}`, "realtime-user");
    }

    if (payload.response.trim().length > 10) {
      await indexText(payload.response, `rt-${contentHash(payload.response)}`, "realtime-assistant");
    }

    if (indexedCount > 0) {
      api.log.info(`[semantic-memory] Real-time indexed ${indexedCount} entries from session ${sessionName}`);
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
          persistNewEntryToDb(api, id);
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
// Multi-Scale Chunking
// =============================================================================

type PendingEntry = { entry: Omit<import("./search.js").VectorEntry, "embedding">; text: string };

/**
 * Re-chunk text at multiple granularities for multi-scale vector indexing.
 * Each scale produces independent vectors: small chunks for precision, large for context.
 */
function multiScaleChunk(
  text: string,
  baseId: string,
  meta: { path: string; startLine: number; endLine: number; source: string },
  scales: Array<{ tokens: number; overlap: number }>
): PendingEntry[] {
  const results: PendingEntry[] = [];
  for (const scale of scales) {
    const maxChars = scale.tokens * 4;
    let overlapChars = scale.overlap * 4;
    // Clamp overlap to prevent infinite loop if misconfigured
    if (overlapChars >= maxChars) overlapChars = Math.max(0, maxChars - 4);
    if (text.length <= maxChars) {
      // Text fits in one chunk at this scale
      results.push({
        entry: {
          id: `${baseId}-s${scale.tokens}`,
          path: meta.path,
          startLine: meta.startLine,
          endLine: meta.endLine,
          source: meta.source,
          snippet: text.slice(0, 500),
          content: text,
        },
        text,
      });
    } else {
      // Split into sub-chunks at this scale
      let start = 0;
      let subIdx = 0;
      while (start < text.length) {
        const end = Math.min(start + maxChars, text.length);
        const chunk = text.slice(start, end);
        if (chunk.trim().length >= 10) {
          results.push({
            entry: {
              id: `${baseId}-s${scale.tokens}-${subIdx}`,
              path: meta.path,
              startLine: meta.startLine,
              endLine: meta.endLine,
              source: meta.source,
              snippet: chunk.slice(0, 500),
              content: chunk,
            },
            text: chunk,
          });
        }
        if (end >= text.length) break; // Last chunk reached, stop
        start = end - overlapChars;
        subIdx++;
      }
    }
  }
  return results;
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
    // Read config from WOPR central config (set by onboard wizard)
    const storedConfig = api.getConfig?.() as Partial<SemanticMemoryConfig> | undefined;
    // Direct init arg overrides stored config (for programmatic use)
    const mergedConfig = { ...storedConfig, ...config };
    await initialize(api, mergedConfig);

    // Register hooks via the event bus
    // Note: WoprPluginApi.on() does not return cleanup functions.
    // Event handler cleanup on re-init is managed by the core event bus.
    api.events.on("session:beforeInject", (payload: any) => handleBeforeInject(api, payload));
    api.events.on("session:afterInject", (payload: any) => handleAfterInject(api, payload));

    // Subscribe to core's file change events for vector indexing
    api.events.on("memory:filesChanged", async (payload: any) => {
      const heapMB = () => Math.round(process.memoryUsage().heapUsed / 1024 / 1024);
      log.info(`filesChanged handler start (heap=${heapMB()}MB)`);
      if (!state.initialized || !state.searchManager) return;

      const changes = payload.changes || [];
      const batch: PendingEntry[] = [];
      const ms = state.config.chunking.multiScale;

      for (const change of changes) {
        if (change.action === "delete") continue;
        if (!change.chunks) continue;

        for (const chunk of change.chunks) {
          if (!chunk.text || chunk.text.trim().length < 10) continue;
          const id = chunk.id;

          if (ms?.enabled && ms.scales.length > 0) {
            // Multi-scale: produce vectors at each granularity
            const subChunks = multiScaleChunk(
              chunk.text, id,
              { path: change.absPath || change.path, startLine: chunk.startLine, endLine: chunk.endLine, source: change.source || "memory" },
              ms.scales
            );
            for (const sc of subChunks) {
              if (!state.searchManager.hasEntry(sc.entry.id)) batch.push(sc);
            }
          } else {
            // Single scale (original behavior)
            if (state.searchManager.hasEntry(id)) continue;
            batch.push({
              entry: {
                id,
                path: change.absPath || change.path,
                startLine: chunk.startLine,
                endLine: chunk.endLine,
                source: change.source || "memory",
                snippet: chunk.text.slice(0, 500),
                content: chunk.text,
              },
              text: chunk.text,
            });
          }
        }
      }

      log.info(`filesChanged: ${changes.length} changes -> ${batch.length} entries to embed (heap=${heapMB()}MB)`);

      if (batch.length > 0) {
        try {
          log.info(`filesChanged: calling addEntriesBatch (heap=${heapMB()}MB)`);
          const count = await state.searchManager.addEntriesBatch(batch);
          log.info(`filesChanged: addEntriesBatch done, ${count} indexed (heap=${heapMB()}MB)`);
          api.log.info(`[semantic-memory] Event-driven indexed ${count} chunks from memory:filesChanged`);

          // Persist embeddings to core's SQLite directly
          if (count > 0) {
            const embeddings: Array<{ id: string; embedding: number[] }> = [];
            for (const item of batch) {
              const stored = state.searchManager.getEntry(item.entry.id);
              if (stored) {
                embeddings.push({
                  id: stored.id,
                  embedding: stored.embedding,
                });
              }
            }
            log.info(`filesChanged: storing ${embeddings.length} embeddings to SQLite (heap=${heapMB()}MB)`);
            if (embeddings.length > 0) {
              storeEmbeddingsToDb(api, embeddings);
              api.log.info(`[semantic-memory] Stored ${embeddings.length} embeddings to SQLite`);
            }
          }
        } catch (err) {
          api.log.error(`[semantic-memory] Batch embedding failed: ${err instanceof Error ? err.message : String(err)}`);
        }
      }
      log.info(`filesChanged handler done (heap=${heapMB()}MB)`);
    });

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

    const id = `man-${contentHash(text)}`;

    await state.searchManager.addEntry(
      {
        id,
        path: source,
        startLine: 0,
        endLine: 0,
        source,
        snippet: text,
        content: text,
      },
      text
    );
    if (state.api) persistNewEntryToDb(state.api, id);
  },

  getConfig(): SemanticMemoryConfig {
    return { ...state.config };
  },
};

export default plugin;

// Re-export types
export type { SemanticMemoryConfig, EmbeddingProvider, MemorySearchResult } from "./types.js";
export { DEFAULT_CONFIG } from "./types.js";
