// MemoryIndexManager - FTS5 keyword search only
// Vector/semantic search available via wopr-plugin-memory-semantic

import fs from "node:fs/promises";
import path from "node:path";
import type { PluginLogger, StorageApi, WOPREventBus } from "@wopr-network/plugin-types";
import { buildFileEntry, chunkMarkdown, ensureDir, hashText, listMemoryFiles } from "./internal.js";
import { syncSessionFiles } from "./sync-sessions.js";
import {
  DEFAULT_MEMORY_CONFIG,
  type MemoryConfig,
  type MemorySearchResult,
  type MemorySource,
  type TemporalFilter,
} from "./types.js";

const META_KEY = "memory_index_meta_v1";
const SNIPPET_MAX_CHARS = 700;
const FTS_TABLE = "memory_chunks_fts";

type MemoryIndexMeta = {
  chunkTokens: number;
  chunkOverlap: number;
};

type MemoryFileChange = {
  action: "upsert" | "delete";
  path: string;
  absPath?: string;
  source: MemorySource;
  chunks?: Array<{
    id: string;
    text: string;
    hash: string;
    startLine: number;
    endLine: number;
  }>;
};

/**
 * Build FTS5 query from raw search string
 */
function buildFtsQuery(raw: string): string | null {
  const tokens =
    raw
      .match(/[A-Za-z0-9_]+/g)
      ?.map((t) => t.trim())
      .filter(Boolean) ?? [];
  if (tokens.length === 0) {
    return null;
  }
  const quoted = tokens.map((t) => `"${t.replaceAll('"', "")}"`);
  return quoted.join(" AND ");
}

/**
 * Convert BM25 rank to normalized score (0-1)
 */
function bm25RankToScore(rank: number): number {
  const normalized = Number.isFinite(rank) ? Math.max(0, rank) : 999;
  return 1 / (1 + normalized);
}

/**
 * Build SQL WHERE clause for temporal filtering
 * Uses the chunks.updated_at column (ms since epoch)
 */
function buildTemporalFilter(temporal: TemporalFilter | undefined, alias?: string): { sql: string; params: number[] } {
  if (!temporal) {
    return { sql: "", params: [] };
  }

  const column = alias ? `${alias}.updated_at` : "updated_at";
  const clauses: string[] = [];
  const params: number[] = [];

  if (temporal.after !== undefined) {
    clauses.push(`${column} >= ?`);
    params.push(temporal.after);
  }

  if (temporal.before !== undefined) {
    clauses.push(`${column} <= ?`);
    params.push(temporal.before);
  }

  if (clauses.length === 0) {
    return { sql: "", params: [] };
  }

  return { sql: ` AND ${clauses.join(" AND ")}`, params };
}

/**
 * Scan /data/sessions/{id}/ directories that have a memory/ subdirectory.
 * Returns the session ROOT dirs (not the memory/ subdirs) because
 * listMemoryFiles() expects a workspace dir and looks for memory/ inside it.
 * Exported so other modules (e.g. a2a-mcp) can discover these dirs independently.
 */
export async function discoverSessionMemoryDirs(): Promise<string[]> {
  const sessionsBase = "/data/sessions";
  const dirs: string[] = [];
  try {
    const entries = await fs.readdir(sessionsBase, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      const sessionDir = path.join(sessionsBase, entry.name);
      const memDir = path.join(sessionDir, "memory");
      try {
        const stat = await fs.stat(memDir);
        if (stat.isDirectory()) dirs.push(sessionDir);
      } catch {}
    }
  } catch {}
  return dirs;
}

export class MemoryIndexManager {
  private readonly globalDir: string;
  private readonly sessionDir: string;
  private readonly sessionsDir: string;
  private readonly config: MemoryConfig;
  private readonly storage: StorageApi;
  private readonly events: WOPREventBus;
  private readonly log: PluginLogger;
  private readonly sources: Set<MemorySource>;
  private readonly fts: {
    enabled: boolean;
    available: boolean;
    loadError?: string;
  };
  private closed = false;
  private dirty = false;
  private syncing: Promise<void> | null = null;

  static async create(params: {
    globalDir: string;
    sessionDir: string;
    config?: Partial<MemoryConfig>;
    storage: StorageApi;
    events: WOPREventBus;
    log: PluginLogger;
  }): Promise<MemoryIndexManager> {
    const config = { ...DEFAULT_MEMORY_CONFIG, ...params.config };

    return new MemoryIndexManager({
      globalDir: params.globalDir,
      sessionDir: params.sessionDir,
      config,
      storage: params.storage,
      events: params.events,
      log: params.log,
    });
  }

  private constructor(params: {
    globalDir: string;
    sessionDir: string;
    config: MemoryConfig;
    storage: StorageApi;
    events: WOPREventBus;
    log: PluginLogger;
  }) {
    this.globalDir = params.globalDir;
    this.sessionDir = params.sessionDir;
    this.sessionsDir = path.join(process.env.WOPR_HOME || "", "sessions");
    this.config = params.config;
    this.storage = params.storage;
    this.events = params.events;
    this.log = params.log;
    this.sources = new Set(["global", "session", "sessions"] as MemorySource[]);
    this.fts = { enabled: true, available: true }; // FTS5 already created by plugin init

    // Subscribe FTS5 indexing as handler for memory:filesChanged
    this.events.on("memory:filesChanged", (event) => {
      return this.handleFilesChanged(event);
    });

    this.dirty = true;
  }

  async search(
    query: string,
    opts?: {
      maxResults?: number;
      minScore?: number;
      temporal?: TemporalFilter;
      instanceId?: string;
    },
  ): Promise<MemorySearchResult[]> {
    if (this.config.sync.onSearch && this.dirty) {
      await this.sync().catch((err) => {
        this.log.warn(`memory sync failed (search): ${String(err)}`);
      });
    }
    const cleaned = query.trim();
    if (!cleaned) {
      return [];
    }
    const minScore = opts?.minScore ?? this.config.query.minScore;
    const maxResults = opts?.maxResults ?? this.config.query.maxResults;
    const temporal = opts?.temporal;

    const results = await this.searchKeyword(cleaned, maxResults * 2, temporal, opts?.instanceId);
    return results.filter((entry) => entry.score >= minScore).slice(0, maxResults);
  }

  private async searchKeyword(
    query: string,
    limit: number,
    temporal?: TemporalFilter,
    instanceId?: string,
  ): Promise<MemorySearchResult[]> {
    if (!this.fts.available) {
      return [];
    }

    const ftsQuery = buildFtsQuery(query);
    if (!ftsQuery) {
      return [];
    }

    const sourceFilter = this.buildSourceFilter();
    const temporalFilter = buildTemporalFilter(temporal, "c");
    const hasTemporal = temporalFilter.sql.length > 0;

    // If instanceId is set, filter to matching rows (or NULL — legacy/global visible to all).
    // Must JOIN with memory_chunks since FTS5 virtual tables don't have instance_id.
    const instanceFilter: { sql: string; params: (string | number)[] } = instanceId
      ? { sql: ` AND (c.instance_id = ? OR c.instance_id IS NULL)`, params: [instanceId] }
      : { sql: "", params: [] };
    const needsJoin = hasTemporal || instanceId;

    // If temporal or instanceId filter is set, join with chunks table
    // FTS5 virtual tables don't have updated_at or instance_id columns
    let sql: string;
    let params: (string | number)[];

    if (needsJoin) {
      sql = `
        SELECT
          f.id,
          f.path,
          f.source,
          f.start_line,
          f.end_line,
          snippet(${FTS_TABLE}, 0, '', '', '...', 64) AS snippet,
          bm25(${FTS_TABLE}) AS rank
        FROM ${FTS_TABLE} f
        JOIN memory_chunks c ON c.id = f.id
        WHERE ${FTS_TABLE} MATCH ?
          ${sourceFilter.sql}
          ${temporalFilter.sql}
          ${instanceFilter.sql}
        ORDER BY rank
        LIMIT ?
      `;
      params = [ftsQuery, ...sourceFilter.params, ...temporalFilter.params, ...instanceFilter.params, limit];
    } else {
      sql = `
        SELECT
          f.id,
          f.path,
          f.source,
          f.start_line,
          f.end_line,
          snippet(${FTS_TABLE}, 0, '', '', '...', 64) AS snippet,
          bm25(${FTS_TABLE}) AS rank
        FROM ${FTS_TABLE} f
        WHERE ${FTS_TABLE} MATCH ?
          ${sourceFilter.sql}
        ORDER BY rank
        LIMIT ?
      `;
      params = [ftsQuery, ...sourceFilter.params, limit];
    }

    try {
      const rows = (await this.storage.raw(sql, params)) as Array<{
        id: string;
        path: string;
        source: MemorySource;
        start_line: number;
        end_line: number;
        snippet: string;
        rank: number;
      }>;

      return rows.map((row) => ({
        path: row.path,
        startLine: row.start_line,
        endLine: row.end_line,
        score: bm25RankToScore(row.rank),
        snippet: row.snippet?.substring(0, SNIPPET_MAX_CHARS) ?? "",
        source: row.source,
      }));
    } catch (err) {
      this.log.warn(`FTS search failed: ${err instanceof Error ? err.message : String(err)}`);
      return [];
    }
  }

  async sync(params?: { force?: boolean }): Promise<void> {
    if (this.syncing) {
      return this.syncing;
    }
    this.syncing = this.runSync(params).finally(() => {
      this.syncing = null;
    });
    return this.syncing;
  }

  private async runSync(_params?: { force?: boolean }): Promise<void> {
    const heapMB = () => Math.round(process.memoryUsage().heapUsed / 1024 / 1024);
    const needsFullReindex = await this.checkNeedsFullReindex();
    this.log.info(`[memory-sync] start (heap: ${heapMB()}MB, fullReindex: ${needsFullReindex})`);

    // Emit per-source to keep memory bounded (don't accumulate all changes at once)

    // Global memory files
    const globalChanges = await this.scanMemoryFiles({
      dirs: [this.globalDir],
      source: "global",
      needsFullReindex,
    });
    this.log.info(`[memory-sync] global: ${globalChanges.length} changes (heap: ${heapMB()}MB)`);
    if (globalChanges.length > 0) {
      await this.events.emit("memory:filesChanged", { changes: globalChanges });
      this.log.info(`[memory-sync] global emitted (heap: ${heapMB()}MB)`);
    }

    // Session memory files (current session + all session memory dirs)
    const sessionMemoryDirs = await discoverSessionMemoryDirs();
    this.log.info(`[memory-sync] session dirs: ${sessionMemoryDirs.length} (heap: ${heapMB()}MB)`);
    const sessionChanges = await this.scanMemoryFiles({
      dirs: [this.sessionDir, ...sessionMemoryDirs],
      source: "session",
      needsFullReindex,
    });
    this.log.info(`[memory-sync] session: ${sessionChanges.length} changes (heap: ${heapMB()}MB)`);
    if (sessionChanges.length > 0) {
      await this.events.emit("memory:filesChanged", { changes: sessionChanges });
      this.log.info(`[memory-sync] session emitted (heap: ${heapMB()}MB)`);
    }

    // Session transcripts — index one file at a time to avoid OOM
    if (this.config.sync.indexSessions !== false) {
      this.log.info(`[memory-sync] starting transcript streaming (heap: ${heapMB()}MB)`);
      await this.syncSessionTranscriptsStreaming(needsFullReindex);
      this.log.info(`[memory-sync] transcripts done (heap: ${heapMB()}MB)`);
    }

    await this.writeMeta();
    this.dirty = false;
    this.log.info(`[memory-sync] complete (heap: ${heapMB()}MB)`);
  }

  /**
   * Stream session transcripts one file at a time — emit per-file so each
   * can be processed and GC'd before loading the next. Prevents OOM from
   * accumulating all 50MB+ of session JSONL in memory at once.
   */
  private async syncSessionTranscriptsStreaming(needsFullReindex: boolean): Promise<void> {
    await syncSessionFiles({
      storage: this.storage,
      sessionsDir: this.sessionsDir,
      needsFullReindex,
      ftsTable: FTS_TABLE,
      ftsEnabled: this.fts.enabled,
      ftsAvailable: this.fts.available,
      model: "fts5",
      dirtyFiles: new Set(),
      runWithConcurrency: (tasks, concurrency) => this.runWithConcurrency(tasks, concurrency),
      indexSessionFile: async (entry) => {
        const chunks = chunkMarkdown(entry.content, this.config.chunking);
        if (chunks.length === 0) return;
        // Emit per-file — handlers process and release before next file
        await this.events.emit("memory:filesChanged", {
          changes: [
            {
              action: "upsert" as const,
              path: entry.path,
              absPath: entry.absPath,
              source: "sessions" as MemorySource,
              chunks: chunks.map((chunk) => ({
                id: hashText(`sessions:${entry.path}:${chunk.startLine}:${chunk.endLine}:${chunk.hash}`),
                text: chunk.text,
                hash: chunk.hash,
                startLine: chunk.startLine,
                endLine: chunk.endLine,
              })),
            },
          ],
        });
      },
      concurrency: 1, // One at a time to keep memory bounded
      log: this.log,
    });
  }

  private async scanMemoryFiles(params: {
    dirs: string[];
    source: MemorySource;
    needsFullReindex: boolean;
  }): Promise<MemoryFileChange[]> {
    const changes: MemoryFileChange[] = [];
    const activePaths = new Set<string>();

    for (const dir of params.dirs) {
      const files = await listMemoryFiles(dir);
      const fileEntries = await Promise.all(files.map(async (file) => buildFileEntry(file, dir)));

      for (const entry of fileEntries) {
        activePaths.add(entry.path);
        const records = (await this.storage.raw(`SELECT hash FROM memory_files WHERE path = ? AND source = ?`, [
          entry.path,
          params.source,
        ])) as Array<{ hash: string }>;
        const record = records[0];
        if (!params.needsFullReindex && record?.hash === entry.hash) {
          continue;
        }
        const content = await fs.readFile(entry.absPath, "utf-8");
        const chunks = chunkMarkdown(content, this.config.chunking);
        changes.push({
          action: "upsert",
          path: entry.path,
          absPath: entry.absPath,
          source: params.source,
          chunks: chunks.map((chunk) => ({
            id: hashText(`${params.source}:${entry.path}:${chunk.startLine}:${chunk.endLine}:${chunk.hash}`),
            text: chunk.text,
            hash: chunk.hash,
            startLine: chunk.startLine,
            endLine: chunk.endLine,
          })),
        });
      }
    }

    // Detect stale entries across ALL dirs for this source
    const staleRows = (await this.storage.raw(`SELECT path FROM memory_files WHERE source = ?`, [
      params.source,
    ])) as Array<{
      path: string;
    }>;
    for (const stale of staleRows) {
      if (!activePaths.has(stale.path)) {
        changes.push({
          action: "delete",
          path: stale.path,
          source: params.source,
        });
      }
    }

    return changes;
  }

  async handleFilesChanged(event: { changes: MemoryFileChange[] }): Promise<void> {
    const heapMB = () => Math.round(process.memoryUsage().heapUsed / 1024 / 1024);
    const totalChunks = event.changes.reduce((sum, c) => sum + (c.chunks?.length || 0), 0);
    this.log.debug(
      `[handleFilesChanged] ${event.changes.length} changes, ${totalChunks} total chunks (heap=${heapMB()}MB)`,
    );

    // Use storage transaction for atomic updates
    await this.storage.transaction(async () => {
      for (const change of event.changes) {
        if (change.action === "delete") {
          await this.storage.raw(`DELETE FROM memory_files WHERE path = ? AND source = ?`, [
            change.path,
            change.source,
          ]);
          await this.storage.raw(`DELETE FROM memory_chunks WHERE path = ? AND source = ?`, [
            change.path,
            change.source,
          ]);
          if (this.fts.available) {
            try {
              await this.storage.raw(`DELETE FROM ${FTS_TABLE} WHERE path = ? AND source = ?`, [
                change.path,
                change.source,
              ]);
            } catch (err) {
              this.log.warn(`[memory] FTS delete failed for ${change.path}: ${err}`);
            }
          }
          continue;
        }

        // Upsert
        if (!change.chunks || change.chunks.length === 0) continue;

        // Delete existing chunks for this file
        await this.storage.raw(`DELETE FROM memory_chunks WHERE path = ? AND source = ?`, [change.path, change.source]);
        if (this.fts.available) {
          try {
            await this.storage.raw(`DELETE FROM ${FTS_TABLE} WHERE path = ? AND source = ?`, [
              change.path,
              change.source,
            ]);
          } catch (err) {
            this.log.warn(`[memory] FTS delete failed for ${change.path}: ${err}`);
          }
        }

        // Insert chunks
        for (const chunk of change.chunks) {
          const chunkId = hashText(`${change.source}:${change.path}:${chunk.startLine}:${chunk.endLine}:${chunk.text}`);
          const now = Date.now();

          await this.storage.raw(
            `INSERT OR REPLACE INTO memory_chunks (id, path, source, start_line, end_line, hash, model, text, updated_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
            [chunkId, change.path, change.source, chunk.startLine, chunk.endLine, chunk.hash, "fts5", chunk.text, now],
          );

          if (this.fts.available) {
            await this.storage.raw(
              `INSERT OR REPLACE INTO ${FTS_TABLE} (text, id, path, source, model, start_line, end_line)
               VALUES (?, ?, ?, ?, ?, ?, ?)`,
              [chunk.text, chunkId, change.path, change.source, "fts5", chunk.startLine, chunk.endLine],
            );
          }
        }

        // Compute file-level hash from chunk hashes for the files table
        const combinedHash = change.chunks.map((c) => c.hash).join("");
        const fileId = hashText(`${change.path}:${change.source}`);
        await this.storage.raw(
          `INSERT OR REPLACE INTO memory_files (id, path, source, hash, mtime, size)
           VALUES (?, ?, ?, ?, ?, ?)`,
          [fileId, change.path, change.source, combinedHash, Date.now(), 0],
        );
      }
    });
  }

  private async checkNeedsFullReindex(): Promise<boolean> {
    const meta = await this.readMeta();
    if (!meta) {
      return true;
    }
    if (meta.chunkTokens !== this.config.chunking.tokens) {
      return true;
    }
    if (meta.chunkOverlap !== this.config.chunking.overlap) {
      return true;
    }
    return false;
  }

  private async readMeta(): Promise<MemoryIndexMeta | null> {
    const rows = (await this.storage.raw(`SELECT value FROM memory_meta WHERE key = ?`, [META_KEY])) as Array<{
      value: string;
    }>;
    const row = rows[0];
    if (!row?.value) {
      return null;
    }
    try {
      return JSON.parse(row.value);
    } catch {
      return null;
    }
  }

  private async writeMeta(): Promise<void> {
    const meta: MemoryIndexMeta = {
      chunkTokens: this.config.chunking.tokens,
      chunkOverlap: this.config.chunking.overlap,
    };
    await this.storage.raw(`INSERT OR REPLACE INTO memory_meta (key, value) VALUES (?, ?)`, [
      META_KEY,
      JSON.stringify(meta),
    ]);
  }

  private buildSourceFilter(alias?: string): { sql: string; params: MemorySource[] } {
    const sources = Array.from(this.sources);
    if (sources.length === 0) {
      return { sql: "", params: [] };
    }
    const column = alias ? `${alias}.source` : "source";
    const placeholders = sources.map(() => "?").join(", ");
    return { sql: ` AND ${column} IN (${placeholders})`, params: sources };
  }

  private async runWithConcurrency<T>(tasks: Array<() => Promise<T>>, concurrency: number): Promise<T[]> {
    const results: T[] = [];
    const executing: Set<Promise<void>> = new Set();

    for (const task of tasks) {
      const p = task()
        .then((result) => {
          results.push(result);
        })
        .finally(() => {
          executing.delete(p);
        });
      executing.add(p);

      if (executing.size >= concurrency) {
        await Promise.race(executing);
      }
    }

    await Promise.all(executing);
    return results;
  }

  async status(): Promise<{
    files: number;
    chunks: number;
    dirty: boolean;
    fts: { enabled: boolean; available: boolean };
  }> {
    const filesRows = (await this.storage.raw(`SELECT COUNT(*) as c FROM memory_files`)) as Array<{ c: number }>;
    const chunksRows = (await this.storage.raw(`SELECT COUNT(*) as c FROM memory_chunks`)) as Array<{ c: number }>;
    return {
      files: filesRows[0]?.c ?? 0,
      chunks: chunksRows[0]?.c ?? 0,
      dirty: this.dirty,
      fts: { enabled: this.fts.enabled, available: this.fts.available },
    };
  }

  async close(): Promise<void> {
    if (this.closed) {
      return;
    }
    this.closed = true;
    // Storage API connection is owned by the plugin context, not us
  }
}
