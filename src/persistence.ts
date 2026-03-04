import { contentHash } from "./manifest.js";
import type { VectorEntry } from "./search.js";

export interface PersistenceLogger {
  info(msg: string): void;
  warn(msg: string): void;
  error(msg: string): void;
}

interface PluginContextLike {
  getExtension<T>(name: string): T | undefined;
}

export function getDb(api: PluginContextLike): any | null {
  return api.getExtension<any>("memory:db") ?? null;
}

/** Ensure the embedding column exists on the chunks table (plugin owns this column) */
export function ensureEmbeddingColumn(db: any, log: PersistenceLogger): void {
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
export async function openDbForRead(
  api: PluginContextLike,
  log: PersistenceLogger,
): Promise<{ db: any; owned: boolean } | null> {
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
export async function loadChunkMetadata(
  api: PluginContextLike,
  log: PersistenceLogger,
): Promise<Map<string, VectorEntry>> {
  const handle = await openDbForRead(api, log);
  if (!handle) return new Map();

  const { db, owned } = handle;
  const entries = new Map<string, VectorEntry>();

  try {
    const stmt = db.prepare(`SELECT id, path, start_line, end_line, source, text, instance_id FROM chunks`);
    for (const row of stmt.iterate()) {
      const r = row as any;
      const text = typeof r.text === "string" ? r.text : "";
      const entry: VectorEntry = {
        id: r.id,
        path: r.path,
        startLine: r.start_line,
        endLine: r.end_line,
        source: r.source,
        snippet: text.slice(0, 500),
        content: text,
        instanceId: r.instance_id ?? undefined,
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
export function getHnswPath(api: PluginContextLike): string | undefined {
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
export function persistNewEntryToDb(
  api: PluginContextLike,
  id: string,
  searchManager: { getEntry(id: string): VectorEntry | undefined },
  embeddingProvider: { id: string } | null,
  instanceId: string | undefined,
  log: PersistenceLogger,
): void {
  const entry = searchManager.getEntry(id);
  if (!entry || !embeddingProvider) return;
  if (!entry.embedding || entry.embedding.length === 0) return;

  const db = getDb(api);
  if (!db) return;
  ensureEmbeddingColumn(db, log);

  try {
    const blob = Buffer.from(new Float32Array(entry.embedding).buffer);
    db.prepare(
      `INSERT INTO chunks (id, path, source, start_line, end_line, hash, model, text, updated_at, embedding, instance_id)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET
         embedding = excluded.embedding,
         updated_at = excluded.updated_at,
         text = excluded.text,
         model = excluded.model,
         instance_id = excluded.instance_id`,
    ).run(
      entry.id,
      entry.path,
      entry.source,
      entry.startLine,
      entry.endLine,
      contentHash(entry.content),
      embeddingProvider.id,
      entry.content,
      Date.now(),
      blob,
      entry.instanceId ?? instanceId ?? null,
    );
  } catch (err) {
    log.warn(`persistNewEntryToDb failed for ${id}: ${err instanceof Error ? err.message : err}`);
  }
}

/**
 * Backfill instanceId on legacy entries (those with NULL instance_id).
 * Returns the number of rows updated.
 *
 * **Important:** This function only updates the SQLite `chunks` table. The HNSW index
 * persists `instanceId` in its `.map.json` file and loads it at startup. To make the
 * backfill take full effect in semantic search queries, delete the HNSW `.bin` and
 * `.map.json` files and restart the process to force an index rebuild.
 */
export function backfillLegacyInstanceId(api: PluginContextLike, instanceId: string, log: PersistenceLogger): number {
  if (!instanceId || instanceId.trim() === "") {
    throw new Error("instanceId must be a non-empty string");
  }
  const db = getDb(api);
  if (!db) {
    log.warn("backfillLegacyInstanceId: no database handle available");
    return 0;
  }

  try {
    const result = db.prepare(`UPDATE chunks SET instance_id = ? WHERE instance_id IS NULL`).run(instanceId);
    const count = typeof result.changes === "number" ? result.changes : 0;
    log.info(`Backfilled ${count} legacy entries with instanceId=${instanceId}`);
    return count;
  } catch (err) {
    log.error(`backfillLegacyInstanceId failed: ${err instanceof Error ? err.message : err}`);
    return 0;
  }
}
