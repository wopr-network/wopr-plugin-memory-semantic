// Session file sync - indexes session transcripts for search
// Adapted from OpenClaw for WOPR
import type { StorageApi } from "@wopr-network/plugin-types";
import type { PluginLogger } from "@wopr-network/plugin-types";
import { buildSessionEntry, listSessionFiles, type SessionFileEntry, sessionPathForFile } from "./session-files.js";

export async function syncSessionFiles(params: {
  storage: StorageApi;
  sessionsDir: string;
  needsFullReindex: boolean;
  ftsTable: string;
  ftsEnabled: boolean;
  ftsAvailable: boolean;
  model: string;
  dirtyFiles: Set<string>;
  runWithConcurrency: <T>(tasks: Array<() => Promise<T>>, concurrency: number) => Promise<T[]>;
  indexSessionFile: (entry: SessionFileEntry) => Promise<void>;
  concurrency: number;
  log: PluginLogger;
}): Promise<void> {
  const files = await listSessionFiles(params.sessionsDir);
  const activePaths = new Set(files.map((file) => sessionPathForFile(file)));
  const indexAll = params.needsFullReindex || params.dirtyFiles.size === 0;

  const tasks = files.map((absPath) => async () => {
    if (!indexAll && !params.dirtyFiles.has(absPath)) {
      return;
    }
    const entry = await buildSessionEntry(absPath);
    if (!entry) {
      return;
    }
    const records = (await params.storage.raw(
      `SELECT hash FROM memory_files WHERE path = ? AND source = ?`,
      [entry.path, "sessions"],
    )) as Array<{ hash: string }>;
    const record = records[0];
    if (!params.needsFullReindex && record?.hash === entry.hash) {
      return;
    }
    await params.indexSessionFile(entry);
  });

  await params.runWithConcurrency(tasks, params.concurrency);

  // Remove stale session entries
  const staleRows = (await params.storage.raw(`SELECT path FROM memory_files WHERE source = ?`, [
    "sessions",
  ])) as Array<{
    path: string;
  }>;
  for (const stale of staleRows) {
    if (activePaths.has(stale.path)) {
      continue;
    }
    await params.storage.raw(`DELETE FROM memory_files WHERE path = ? AND source = ?`, [stale.path, "sessions"]);
    await params.storage.raw(`DELETE FROM memory_chunks WHERE path = ? AND source = ?`, [stale.path, "sessions"]);
    if (params.ftsEnabled && params.ftsAvailable) {
      try {
        await params.storage.raw(`DELETE FROM ${params.ftsTable} WHERE path = ? AND source = ? AND model = ?`, [
          stale.path,
          "sessions",
          params.model,
        ]);
      } catch (err) {
        params.log.warn(`[sync-sessions] FTS delete failed for ${stale.path}: ${err}`);
      }
    }
  }
}
