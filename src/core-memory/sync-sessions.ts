// Session file sync - indexes session transcripts for search
// Adapted from OpenClaw for WOPR
import type { PluginLogger, StorageApi } from "@wopr-network/plugin-types";
import type { SessionApi } from "../types.js";
import { buildSessionEntryFromSql, listSessionNames, type SessionFileEntry } from "./session-files.js";

export async function syncSessionFiles(params: {
  storage: StorageApi;
  sessionsDir: string;
  needsFullReindex: boolean;
  ftsTable: string;
  ftsEnabled: boolean;
  ftsAvailable: boolean;
  model: string;
  dirtyFiles: Set<string>;
  runWithConcurrency: <T>(
    tasks: Array<() => Promise<T>>,
    concurrency: number,
  ) => Promise<{ results: T[]; hadErrors: boolean }>;
  indexSessionFile: (entry: SessionFileEntry) => Promise<void>;
  concurrency: number;
  log: PluginLogger;
  sessionApi?: SessionApi;
}): Promise<void> {
  if (!params.sessionApi) {
    params.log.warn("[sync-sessions] ctx.session not available — skipping session sync");
    return;
  }

  const sessionNames = await listSessionNames(params.storage, params.log);
  const activePaths = new Set(sessionNames.map((name) => `sessions/${name}`));
  const indexAll = params.needsFullReindex || params.dirtyFiles.size === 0;

  const tasks = sessionNames.map((sessionName) => async () => {
    const entryPath = `sessions/${sessionName}`;
    if (!indexAll && !params.dirtyFiles.has(entryPath)) {
      return;
    }

    const entry = await buildSessionEntryFromSql(sessionName, params.sessionApi!, params.log);
    if (!entry) {
      return;
    }

    const records = (await params.storage.raw(`SELECT hash FROM memory_files WHERE path = ? AND source = ?`, [
      entry.path,
      "sessions",
    ])) as Array<{ hash: string }>;
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
