import { join } from "node:path";
import type { WOPRPluginContext } from "@wopr-network/plugin-types";
import { registerMemoryTools } from "./a2a-tools.js";
import { MemoryIndexManager } from "./core-memory/manager.js";
import { createSessionDestroyHandler } from "./core-memory/session-hook.js";
import { startWatcher } from "./core-memory/watcher.js";
import type { EmbeddingQueue, PendingEntry } from "./embedding-queue.js";
import { createEmbeddingProvider } from "./embeddings.js";
import { registerIdentityTools } from "./identity-tools.js";
import { createMemoryPluginSchema } from "./memory-schema.js";
import { getHnswPath, loadChunkMetadata, persistNewEntryToDb } from "./persistence.js";
import { createSemanticSearchManager, type SemanticSearchManager } from "./search.js";
import type { EmbeddingProvider, SemanticMemoryConfig, SessionApi } from "./types.js";
import { DEFAULT_CONFIG } from "./types.js";

interface PluginContext extends WOPRPluginContext {
  memory?: {
    keywordSearch?(query: string, limit: number): Promise<any[]>;
  };
  session?: SessionApi;
}

export interface PluginState {
  config: SemanticMemoryConfig;
  embeddingProvider: EmbeddingProvider | null;
  searchManager: SemanticSearchManager | null;
  memoryManager: MemoryIndexManager | null;
  api: PluginContext | null;
  initialized: boolean;
  eventCleanup: Array<() => void>;
  instanceId: string | undefined;
}

export interface InitLogger {
  info(msg: string): void;
  warn(msg: string): void;
  error(msg: string): void;
}

let initInProgress = false;

export async function initialize(
  api: PluginContext,
  state: PluginState,
  embeddingQueue: EmbeddingQueue,
  log: InitLogger,
  userConfig?: Partial<SemanticMemoryConfig>,
): Promise<void> {
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

  // Capture instanceId: config > env var > undefined (single-instance mode)
  state.instanceId = state.config.instanceId || process.env.WOPR_INSTANCE_ID || undefined;

  if (!state.instanceId) {
    api.log.warn(
      "[semantic-memory] No instanceId configured — multi-tenant isolation DISABLED. " +
        "Set instanceId in plugin config or WOPR_INSTANCE_ID env var.",
    );
  } else {
    api.log.info(`[semantic-memory] Instance isolation enabled: instanceId=${state.instanceId}`);
  }

  try {
    // 1. Register memory schema (creates tables in wopr.sqlite).
    // Pass the resolved instanceId so the v1→v2 migration tags rows with the
    // correct value (config.instanceId || env var) rather than reading the env
    // var directly inside the migration, which would mismatch when they differ.
    await api.storage.register(createMemoryPluginSchema(state.instanceId));
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
    await api.storage.raw(`ALTER TABLE memory_chunks ADD COLUMN embedding BLOB`).catch(() => {
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
      sessionApi: api.session,
    });
    api.log.info("[semantic-memory] MemoryIndexManager created");

    // 5. Register session:destroy handler (was initMemoryHooks in core)
    const sessionDestroyHandler = await createSessionDestroyHandler({
      sessionsDir,
      log: api.log,
      sessionApi: api.session,
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
    registerMemoryTools(api, state.memoryManager, state.instanceId);

    // 8b. Register identity tools (moved from core in WOP-1773)
    registerIdentityTools(api);

    // Create embedding provider
    state.embeddingProvider = await createEmbeddingProvider(state.config);
    api.log.info(`[semantic-memory] Embedding provider: ${state.embeddingProvider.id}`);

    // Create search manager
    // Wire up keyword search from memory manager
    const keywordSearchFn = state.memoryManager
      ? async (query: string, limit: number, instanceId?: string) => {
          const results = await state.memoryManager!.search(query, {
            maxResults: limit,
            instanceId,
            excludeLegacyEntries: !!(instanceId && state.config.search.excludeLegacyEntries),
          });
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
        ? async (query: string, limit: number, _instanceId?: string) => {
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
    const chunkMetadata = await loadChunkMetadata(api, log);
    // Lazy resolver: DB extension may not be available at init
    const hnswPathFn = () => getHnswPath(api);

    state.searchManager = await createSemanticSearchManager(
      state.config,
      state.embeddingProvider,
      keywordSearchFn,
      hnswPathFn,
    );

    // Attach the queue to the search manager with persistence callback
    embeddingQueue.attach(state.searchManager, (id) =>
      persistNewEntryToDb(api, id, state.searchManager!, state.embeddingProvider, state.instanceId, log),
    );

    const loadedVectors = state.searchManager.getEntryCount();
    state.api = api;
    state.initialized = true;
    api.log.info(`[semantic-memory] Initialized with ${loadedVectors} persisted vectors`);

    // Bootstrap: if HNSW is empty but we have chunk metadata, embed them now
    if (loadedVectors === 0 && chunkMetadata.size > 0) {
      api.log.info(`[semantic-memory] Bootstrap: ${chunkMetadata.size} chunks need embedding, starting async...`);
      bootstrapEmbeddings(chunkMetadata, state, embeddingQueue, log).catch((err) => {
        api.log.error(`[semantic-memory] Bootstrap failed: ${err instanceof Error ? err.message : err}`);
      });
    } else {
      api.log.info(`[semantic-memory] Initialized — waiting for memory:filesChanged events from core`);
    }
  } catch (err) {
    api.log.error(`[semantic-memory] Failed to initialize: ${err instanceof Error ? err.message : String(err)}`);
  } finally {
    initInProgress = false;
  }
}

export async function bootstrapEmbeddings(
  chunkMetadata: Map<string, import("./search.js").VectorEntry>,
  state: PluginState,
  embeddingQueue: EmbeddingQueue,
  log: InitLogger,
): Promise<void> {
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
        // Preserve existing instanceId from SQLite; only tag with current instance if unset
        instanceId: entry.instanceId ?? state.instanceId,
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
