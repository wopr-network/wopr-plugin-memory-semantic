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

import { createHash } from "node:crypto";
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import type { WOPRPluginContext } from "@wopr-network/plugin-types";
import winston from "winston";
import { registerMemoryTools } from "./a2a-tools.js";
import { extractFromConversation } from "./capture.js";
import { MemoryIndexManager } from "./core-memory/manager.js";
import { createSessionDestroyHandler } from "./core-memory/session-hook.js";
import { startWatcher, stopWatcher } from "./core-memory/watcher.js";
import { createEmbeddingProvider } from "./embeddings.js";
import { memoryPluginSchema } from "./memory-schema.js";
import { performAutoRecall } from "./recall.js";
import { createSemanticSearchManager, type SemanticSearchManager } from "./search.js";
import type { EmbeddingProvider, MemorySearchResult, SemanticMemoryConfig } from "./types.js";
import { DEFAULT_CONFIG } from "./types.js";

/**
 * Extended plugin context — adds the optional `memory` extension that
 * core exposes for keyword search fallback.  Everything else comes from
 * the shared @wopr-network/plugin-types package.
 */
interface PluginContext extends WOPRPluginContext {
  memory?: {
    keywordSearch?(query: string, limit: number): Promise<any[]>;
  };
}

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

// Generate deterministic ID from content to avoid duplicates
function contentHash(text: string): string {
  return createHash("sha256").update(text).digest("hex");
}

// =============================================================================
// Plugin State
// =============================================================================

// =============================================================================
// Embedding Queue — serializes all embedding work through a single channel
// =============================================================================

class EmbeddingQueue {
  private queue: PendingEntry[] = [];
  private processing = false;
  private _bootstrapping = false;
  private searchManager: SemanticSearchManager | null = null;

  get bootstrapping(): boolean {
    return this._bootstrapping;
  }

  attach(sm: SemanticSearchManager): void {
    this.searchManager = sm;
  }

  /** Enqueue entries and start processing if idle. Returns immediately. */
  enqueue(entries: PendingEntry[], source: string): void {
    if (!this.searchManager) return;
    // Deduplicate against already-indexed AND against entries already in queue
    const queuedIds = new Set(this.queue.map((e) => e.entry.id));
    let added = 0;
    for (const entry of entries) {
      if (this.searchManager.hasEntry(entry.entry.id)) continue;
      if (queuedIds.has(entry.entry.id)) continue;
      this.queue.push(entry);
      queuedIds.add(entry.entry.id);
      added++;
    }
    log.info(`[queue] enqueued ${added} entries from ${source} (${this.queue.length} total pending)`);
    this.drain();
  }

  /** Run bootstrap: enqueue all chunks and process to completion before anything else. */
  async bootstrap(entries: PendingEntry[]): Promise<number> {
    this._bootstrapping = true;
    log.info(`[queue] bootstrap starting: ${entries.length} entries`);
    this.enqueue(entries, "bootstrap");
    // Wait for the queue to fully drain
    await this.waitForDrain();
    this._bootstrapping = false;
    const count = this.searchManager?.getEntryCount() ?? 0;
    log.info(`[queue] bootstrap complete: ${count} vectors in index`);
    return count;
  }

  /** Process the queue sequentially — only one batch at a time. */
  private async drain(): Promise<void> {
    if (this.processing || this.queue.length === 0 || !this.searchManager) return;
    this.processing = true;

    try {
      while (this.queue.length > 0) {
        // Take a batch from the front of the queue
        const batch = this.queue.splice(0, Math.min(this.queue.length, 500));
        log.info(`[queue] processing batch: ${batch.length} entries (${this.queue.length} remaining)`);
        try {
          await this.searchManager.addEntriesBatch(batch);
          // Persist plugin-originated entries (real-time, capture) to SQLite
          if (state.api) {
            for (const entry of batch) {
              if (entry.persist) persistNewEntryToDb(state.api, entry.entry.id);
            }
          }
        } catch (err) {
          log.error(`[queue] batch failed: ${err instanceof Error ? err.message : err}`);
        }
      }
    } finally {
      this.processing = false;
    }
  }

  private waitForDrain(): Promise<void> {
    return new Promise<void>((resolve) => {
      const check = () => {
        if (!this.processing && this.queue.length === 0) {
          resolve();
        } else {
          setTimeout(check, 500);
        }
      };
      check();
    });
  }

  clear(): void {
    this.queue = [];
    this.processing = false;
    this._bootstrapping = false;
    this.searchManager = null;
  }
}

const embeddingQueue = new EmbeddingQueue();

// =============================================================================
// Plugin State
// =============================================================================

interface PluginState {
  config: SemanticMemoryConfig;
  embeddingProvider: EmbeddingProvider | null;
  searchManager: SemanticSearchManager | null;
  memoryManager: MemoryIndexManager | null;
  api: PluginContext | null;
  initialized: boolean;
  eventCleanup: Array<() => void>; // event bus unsubscribe functions
}

const state: PluginState = {
  config: DEFAULT_CONFIG,
  embeddingProvider: null,
  searchManager: null,
  memoryManager: null,
  api: null,
  initialized: false,
  eventCleanup: [],
};

// PluginContext replaced by WOPRPluginContext from @wopr-network/plugin-types
// (extended locally as PluginContext for the optional memory property)

// =============================================================================
// SQLite Embedding Persistence (via core's memory:db extension)
// =============================================================================

function getDb(api: PluginContext): any | null {
  return api.getExtension<any>("memory:db") ?? null;
}

/** Ensure the embedding column exists on the chunks table (plugin owns this column) */
function ensureEmbeddingColumn(db: any): void {
  if (!db || typeof db.prepare !== "function" || typeof db.exec !== "function") return;
  try {
    const cols = db.prepare(`PRAGMA table_info(chunks)`).all() as Array<{ name: string }>;
    if (!cols.some((c: any) => c.name === "embedding")) {
      db.exec(`ALTER TABLE chunks ADD COLUMN embedding BLOB`);
    }
  } catch (err) {
    log.warn(`ensureEmbeddingColumn failed: ${err instanceof Error ? err.message : err}`);
  }
}

/** Open a read-only DB handle — tries core extension first, falls back to direct file open. */
async function openDbForRead(api: PluginContext): Promise<{ db: any; owned: boolean } | null> {
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

/** Load chunk metadata from SQLite — used at startup to seed the HNSW metadata
 *  map and to identify chunks that need (re-)embedding during bootstrap.
 *  Embeddings are NOT loaded from SQLite (the HNSW binary is the vector source
 *  of truth), so we return a dummy empty embedding. */
async function loadChunkMetadata(api: PluginContext): Promise<Map<string, import("./search.js").VectorEntry>> {
  const handle = await openDbForRead(api);
  if (!handle) return new Map();

  const { db, owned } = handle;
  const entries = new Map<string, import("./search.js").VectorEntry>();

  try {
    const stmt = db.prepare(`SELECT id, path, start_line, end_line, source, text FROM chunks`);
    for (const row of stmt.iterate()) {
      const r = row as any;
      const text = typeof r.text === "string" ? r.text : "";
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
    if (owned) {
      try {
        db.close();
      } catch {
        /* ignore */
      }
    }
  }
  return entries;
}

/** Derive path for the persisted HNSW binary.
 *  Tries the DB handle first, falls back to WOPR_HOME convention. */
function getHnswPath(api: PluginContext): string | undefined {
  // Try DB handle (available after core memory init)
  const db = getDb(api);
  if (db) {
    const dbPath: unknown = typeof db.location === "function" ? db.location() : db.name;
    if (typeof dbPath === "string" && dbPath && dbPath !== ":memory:") {
      return `${dbPath}.hnsw`;
    }
  }
  // Fallback: WOPR_HOME convention (always available)
  const home = process.env.WOPR_HOME;
  if (home) return `${home}/memory/index.sqlite.hnsw`;
  return undefined;
}

/** INSERT a plugin-originated entry (real-time, capture) into chunks with its embedding */
function persistNewEntryToDb(api: PluginContext, id: string): void {
  const entry = state.searchManager?.getEntry(id);
  if (!entry || !state.embeddingProvider) return;
  if (!entry.embedding || entry.embedding.length === 0) return;

  const db = getDb(api);
  if (!db) return;
  ensureEmbeddingColumn(db);

  try {
    const blob = Buffer.from(new Float32Array(entry.embedding).buffer);
    db.prepare(
      `INSERT INTO chunks (id, path, source, start_line, end_line, hash, model, text, updated_at, embedding)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET
         embedding = excluded.embedding,
         updated_at = excluded.updated_at,
         text = excluded.text,
         model = excluded.model`,
    ).run(
      entry.id,
      entry.path,
      entry.source,
      entry.startLine,
      entry.endLine,
      contentHash(entry.content),
      state.embeddingProvider.id,
      entry.content,
      Date.now(),
      blob,
    );
  } catch (err) {
    log.warn(`persistNewEntryToDb failed for ${id}: ${err instanceof Error ? err.message : err}`);
  }
}

// =============================================================================
// Initialization
// =============================================================================

let initInProgress = false;
async function initialize(api: PluginContext, userConfig?: Partial<SemanticMemoryConfig>) {
  if (state.initialized || initInProgress) return;
  initInProgress = true;

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
      multiScale: userConfig?.chunking?.multiScale
        ? {
            ...DEFAULT_CONFIG.chunking.multiScale,
            ...userConfig.chunking.multiScale,
            scales: userConfig.chunking.multiScale.scales ?? DEFAULT_CONFIG.chunking.multiScale?.scales ?? [],
          }
        : DEFAULT_CONFIG.chunking.multiScale,
    },
  };

  try {
    // 1. Register memory schema (creates tables in wopr.sqlite)
    await api.storage.register(memoryPluginSchema);
    api.log.info("[semantic-memory] Registered memory schema with Storage API");

    // 2. Create FTS5 virtual table via raw SQL
    await api.storage.raw(`
      CREATE VIRTUAL TABLE IF NOT EXISTS memory_chunks_fts USING fts5(
        text,
        id UNINDEXED,
        path UNINDEXED,
        source UNINDEXED,
        model UNINDEXED,
        start_line UNINDEXED,
        end_line UNINDEXED
      )
    `);
    api.log.info("[semantic-memory] Created FTS5 virtual table");

    // 3. Ensure embedding column on chunks table (plugin owns this)
    await api.storage
      .raw(`ALTER TABLE memory_chunks ADD COLUMN embedding BLOB`)
      .catch(() => {
        /* column may already exist */
      });

    // 4. Create MemoryIndexManager with Storage API
    const globalDir = process.env.WOPR_GLOBAL_IDENTITY || "/data/identity";
    const sessionsDir = join(process.env.WOPR_HOME || "", "sessions");
    const sessionDir = join(sessionsDir, "_boot");

    state.memoryManager = await MemoryIndexManager.create({
      globalDir,
      sessionDir,
      config: state.config as any, // MemoryConfig subset
      storage: api.storage,
      events: api.events,
      log: api.log,
    });
    api.log.info("[semantic-memory] MemoryIndexManager created");

    // 5. Register session:destroy handler (was initMemoryHooks in core)
    const sessionDestroyHandler = await createSessionDestroyHandler({
      sessionsDir,
      log: api.log,
    });
    const unsubSessionDestroy = api.events.on("session:destroy", async (payload: any) => {
      await sessionDestroyHandler(payload.session, payload.reason);
    });
    state.eventCleanup.push(unsubSessionDestroy);

    // 6. Start file watcher (was in core)
    if (state.config.sync?.watch !== false) {
      await startWatcher({
        dirs: [globalDir, sessionDir],
        debounceMs: state.config.sync?.watchDebounceMs ?? 1500,
        onSync: () => state.memoryManager!.sync(),
        log: api.log,
      });
    }

    // 7. Run initial sync
    await state.memoryManager.sync();
    api.log.info("[semantic-memory] Initial memory sync complete");

    // 8. Register A2A memory tools
    registerMemoryTools(api, state.memoryManager);

    // Create embedding provider
    state.embeddingProvider = await createEmbeddingProvider(state.config);
    api.log.info(`[semantic-memory] Embedding provider: ${state.embeddingProvider.id}`);

    // Create search manager
    // Wire up keyword search from memory manager
    const keywordSearchFn = state.memoryManager
      ? async (query: string, limit: number) => {
          const results = await state.memoryManager!.search(query, { maxResults: limit });
          return results.map((r: any) => ({
            id: r.id || `${r.path}:${r.startLine}`,
            path: r.path,
            startLine: r.startLine || 0,
            endLine: r.endLine || 0,
            source: r.source || "memory",
            snippet: r.snippet || r.content || "",
            content: r.content || r.snippet || "",
            textScore: r.score || 0,
          }));
        }
      : api.memory?.keywordSearch
      ? async (query: string, limit: number) => {
          const results = await api.memory!.keywordSearch!(query, limit);
          return results.map((r: any) => ({
            id: r.id || `${r.path}:${r.startLine}`,
            path: r.path,
            startLine: r.startLine || 0,
            endLine: r.endLine || 0,
            source: r.source || "memory",
            snippet: r.snippet || r.content || "",
            content: r.content || r.snippet || "", // Use content if available, fallback to snippet
            textScore: r.score || 0,
          }));
        }
      : undefined;

    // Load chunk metadata from SQLite for HNSW metadata seeding and bootstrap dedup.
    // Embeddings in SQLite are stale/empty — the HNSW binary is the vector source of truth.
    const chunkMetadata = await loadChunkMetadata(api);
    // Lazy resolver: DB extension may not be available at init
    const hnswPathFn = () => getHnswPath(api);

    state.searchManager = await createSemanticSearchManager(
      state.config,
      state.embeddingProvider,
      keywordSearchFn,
      hnswPathFn,
    );

    // Attach the queue to the search manager
    embeddingQueue.attach(state.searchManager);

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
    initInProgress = false;
    api.log.error(`[semantic-memory] Failed to initialize: ${err instanceof Error ? err.message : String(err)}`);
  }
}

// =============================================================================
// Bootstrap: embed existing chunks that have no HNSW vectors
// =============================================================================

async function bootstrapEmbeddings(chunkMetadata: Map<string, import("./search.js").VectorEntry>): Promise<void> {
  const heapMB = () => Math.round(process.memoryUsage().heapUsed / 1024 / 1024);
  log.info(`Bootstrap start (heap=${heapMB()}MB)`);
  if (!state.embeddingProvider) return;

  const entries: PendingEntry[] = [];
  for (const [, entry] of chunkMetadata) {
    if (!entry.content || entry.content.length < 10) continue;
    entries.push({
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

  if (entries.length === 0) {
    log.info("Bootstrap: all chunks already indexed");
    return;
  }

  log.info(`Bootstrap: ${entries.length} chunks via ${state.embeddingProvider.id} (heap=${heapMB()}MB)`);
  await embeddingQueue.bootstrap(entries);
  log.info(`Bootstrap complete (heap=${heapMB()}MB)`);
}

// =============================================================================
// Hook Handlers
// =============================================================================

/**
 * Before inject hook - auto-recall relevant memories
 */
async function handleBeforeInject(api: PluginContext, payload: any): Promise<void> {
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
    const recall = await performAutoRecall(lastUserMessage.content, state.searchManager, state.config);

    if (recall && recall.memories.length > 0) {
      api.log.info(`[semantic-memory] Recalled ${recall.memories.length} memories (queryLen=${recall.query.length})`);
      // Prepend memory context to the mutable message payload
      // Core uses emitMutableIncoming → payload.message is mutable
      payload.message = `${recall.context}\n\n${payload.message}`;
    }
  } catch (err) {
    api.log.error(`[semantic-memory] Auto-recall failed: ${err instanceof Error ? err.message : String(err)}`);
  }
}

/**
 * After inject hook - real-time indexing of ALL session content
 * Payload is SessionResponseEvent: { session, message, response, from }
 */
async function handleAfterInject(api: PluginContext, payload: any): Promise<void> {
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

    // Helper: index a text with optional multi-scale — enqueues through the serialized queue
    const indexText = (text: string, baseId: string, source: string) => {
      const entries: PendingEntry[] = [];
      if (ms?.enabled && ms.scales.length > 0) {
        const subChunks = multiScaleChunk(
          text,
          baseId,
          { path: `session:${sessionName}`, startLine: 0, endLine: 0, source },
          ms.scales,
        );
        for (const sc of subChunks) {
          entries.push({ ...sc, persist: true });
        }
      } else {
        entries.push({
          entry: {
            id: baseId,
            path: `session:${sessionName}`,
            startLine: 0,
            endLine: 0,
            source,
            snippet: text.slice(0, 500),
            content: text,
          },
          text,
          persist: true,
        });
      }
      if (entries.length > 0) {
        embeddingQueue.enqueue(entries, `realtime:${source}`);
        indexedCount += entries.length;
      }
    };

    // REAL-TIME INDEXING: Index session content immediately with full text
    // Include session name in hash to prevent cross-session collisions
    if (payload.message && payload.message.trim().length > 10) {
      indexText(payload.message, `rt-${contentHash(`${sessionName}:user:${payload.message}`)}`, "realtime-user");
    }

    if (payload.response.trim().length > 10) {
      indexText(
        payload.response,
        `rt-${contentHash(`${sessionName}:assistant:${payload.response}`)}`,
        "realtime-assistant",
      );
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

        const captureEntries: PendingEntry[] = candidates.map((candidate) => ({
          entry: {
            id: `cap-${contentHash(candidate.text)}`,
            path: `session:${sessionName}`,
            startLine: 0,
            endLine: 0,
            source: "auto-capture",
            snippet: candidate.text.slice(0, 500),
            content: candidate.text,
          },
          text: candidate.text,
          persist: true,
        }));
        embeddingQueue.enqueue(captureEntries, `auto-capture(${candidates.length})`);

        api.log.info(`[semantic-memory] Queued ${candidates.length} capture memories`);
      }
    }
  } catch (err) {
    api.log.error(`[semantic-memory] Real-time indexing failed: ${err instanceof Error ? err.message : String(err)}`);
  }
}

// =============================================================================
// Multi-Scale Chunking
// =============================================================================

type PendingEntry = { entry: Omit<import("./search.js").VectorEntry, "embedding">; text: string; persist?: boolean };

/**
 * Re-chunk text at multiple granularities for multi-scale vector indexing.
 * Each scale produces independent vectors: small chunks for precision, large for context.
 */
function multiScaleChunk(
  text: string,
  baseId: string,
  meta: { path: string; startLine: number; endLine: number; source: string },
  scales: Array<{ tokens: number; overlap: number }>,
): PendingEntry[] {
  const results: PendingEntry[] = [];

  // Always emit a canonical entry under baseId (smallest scale) so that
  // hasEntry(baseId) works for dedup on restart/bootstrap
  const smallest = scales.reduce((a, b) => (a.tokens <= b.tokens ? a : b), scales[0]);
  if (smallest && text.trim().length >= 10) {
    const maxChars = smallest.tokens * 4;
    results.push({
      entry: {
        id: baseId,
        path: meta.path,
        startLine: meta.startLine,
        endLine: meta.endLine,
        source: meta.source,
        snippet: text.slice(0, 500),
        content: text.length <= maxChars ? text : text.slice(0, maxChars),
      },
      text: text.length <= maxChars ? text : text.slice(0, maxChars),
    });
  }

  for (const scale of scales) {
    // Skip invalid scales to prevent infinite loops
    if (!Number.isFinite(scale.tokens) || scale.tokens <= 0) continue;
    const maxChars = scale.tokens * 4;
    let overlapChars = (Number.isFinite(scale.overlap) ? scale.overlap : 0) * 4;
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

  init(api: PluginContext, config?: Partial<SemanticMemoryConfig>): Promise<void>;
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

  async init(api: PluginContext, config?: Partial<SemanticMemoryConfig>) {
    // Override api.log to use our file-backed winston logger
    api.log = {
      info: (msg: string) => log.info(msg),
      warn: (msg: string) => log.warn(msg),
      error: (msg: string) => log.error(msg),
      debug: (msg: string) => log.debug(msg),
    };
    api.log.info("[semantic-memory] init() called");

    // Clean up previous subscriptions if re-initialized
    for (const unsub of state.eventCleanup) {
      try {
        unsub();
      } catch {}
    }
    state.eventCleanup = [];

    // Read config from WOPR central config (set by onboard wizard)
    const storedConfig = api.getConfig?.() as Partial<SemanticMemoryConfig> | undefined;
    // Direct init arg overrides stored config (for programmatic use)
    const mergedConfig = { ...storedConfig, ...config };
    await initialize(api, mergedConfig);

    if (!state.initialized) {
      api.log.error("[semantic-memory] Initialization failed — plugin will not activate");
      return;
    }

    // Register hooks via the event bus — store cleanup functions for shutdown
    const unsubBeforeInject = api.events.on("session:beforeInject", (payload: any) => handleBeforeInject(api, payload));
    const unsubAfterInject = api.events.on("session:afterInject", (payload: any) => handleAfterInject(api, payload));

    // Subscribe to core's file change events for vector indexing
    const unsubFilesChanged = api.events.on("memory:filesChanged", async (payload: any) => {
      if (!state.initialized || !state.searchManager) return;
      if (embeddingQueue.bootstrapping) {
        log.info(`filesChanged: skipped (bootstrap in progress)`);
        return;
      }

      const changes = payload.changes || [];
      const entries: PendingEntry[] = [];
      const ms = state.config.chunking.multiScale;

      for (const change of changes) {
        if (change.action === "delete") continue;
        if (!change.chunks) continue;

        for (const chunk of change.chunks) {
          if (!chunk.text || chunk.text.trim().length < 10) continue;
          const id = chunk.id;

          if (ms?.enabled && ms.scales.length > 0) {
            const subChunks = multiScaleChunk(
              chunk.text,
              id,
              {
                path: change.absPath || change.path,
                startLine: chunk.startLine,
                endLine: chunk.endLine,
                source: change.source || "memory",
              },
              ms.scales,
            );
            for (const sc of subChunks) entries.push(sc);
          } else {
            entries.push({
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

      if (entries.length > 0) {
        embeddingQueue.enqueue(entries, `filesChanged(${changes.length} files)`);
      }
    });

    // Hook into memory_search to provide semantic results
    const unsubSearch = api.events.on(
      "memory:search",
      async (payload: {
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
          api.log.info(
            `[semantic-memory] After filter: ${payload.results.length} results (minScore: ${payload.minScore})`,
          );
        } catch (err) {
          api.log.error(`[semantic-memory] Search failed: ${err instanceof Error ? err.message : String(err)}`);
        }
      },
    );

    state.eventCleanup = [unsubBeforeInject, unsubAfterInject, unsubFilesChanged, unsubSearch];
    api.log.info("[semantic-memory] Plugin initialized - memory_search enhanced with semantic search");
  },

  async shutdown() {
    // Stop the embedding queue first
    embeddingQueue.clear();

    // Stop file watcher
    if (state.api) {
      await stopWatcher(state.api.log);
    }

    // Unsubscribe all event handlers
    for (const unsub of state.eventCleanup) {
      try {
        unsub();
      } catch {}
    }
    state.eventCleanup = [];

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
          },
          text,
          persist: true,
        },
      ],
      `manual-capture`,
    );
  },

  getConfig(): SemanticMemoryConfig {
    return { ...state.config };
  },
};

export default plugin;

// Re-export types
export type { EmbeddingProvider, MemorySearchResult, SemanticMemoryConfig } from "./types.js";
export { DEFAULT_CONFIG } from "./types.js";
export type { AuthContext, WebMCPHandler, WebMCPRegistryLike, WebMCPTool, WebMCPToolDeclaration } from "./webmcp.js";
// Re-export WebMCP tools for browser-side registration
export { registerMemoryTools, unregisterMemoryTools, WEBMCP_MANIFEST } from "./webmcp.js";
