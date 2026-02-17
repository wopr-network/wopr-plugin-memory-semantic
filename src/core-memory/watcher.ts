// File watcher for auto-sync - uses chokidar for cross-platform watching
import type { PluginLogger } from "@wopr-network/plugin-types";

type FSWatcher = { close(): Promise<void>; on(event: string, handler: (...args: unknown[]) => void): FSWatcher };

let watcher: FSWatcher | null = null;
let watcherPromise: Promise<void> | null = null;
let debounceTimer: NodeJS.Timeout | null = null;

export type WatcherCallback = () => Promise<void>;

/**
 * Start file watching for auto-sync
 */
export async function startWatcher(params: {
  dirs: string[];
  debounceMs: number;
  onSync: WatcherCallback;
  log: PluginLogger;
}): Promise<void> {
  if (watcher) {
    return; // Already watching
  }

  try {
    // Dynamic import to avoid loading chokidar unless needed
    interface ChokidarWatchOptions {
      ignored?: RegExp | ((path: string) => boolean);
      persistent?: boolean;
      ignoreInitial?: boolean;
      awaitWriteFinish?: {
        stabilityThreshold: number;
        pollInterval: number;
      };
    }
    const chokidar = await (Function('return import("chokidar")')() as Promise<{
      watch: (paths: string[], options: ChokidarWatchOptions) => FSWatcher;
    }>);

    watcher = chokidar.watch(params.dirs, {
      ignored: /(^|[/\\])\../, // Ignore dotfiles
      persistent: true,
      ignoreInitial: true,
      awaitWriteFinish: {
        stabilityThreshold: 500,
        pollInterval: 100,
      },
    });

    const triggerSync = () => {
      if (debounceTimer) {
        clearTimeout(debounceTimer);
      }
      debounceTimer = setTimeout(async () => {
        try {
          await params.onSync();
        } catch (err) {
          params.log.warn(`[memory-watcher] Sync failed: ${err instanceof Error ? err.message : String(err)}`);
        }
      }, params.debounceMs);
    };

    watcher.on("add", triggerSync);
    watcher.on("change", triggerSync);
    watcher.on("unlink", triggerSync);

    // Wait for watcher to be ready
    watcherPromise = new Promise<void>((resolve, reject) => {
      watcher?.on("ready", () => resolve());
      watcher?.on("error", (err: unknown) => reject(err));
    });

    await watcherPromise;
    params.log.info(`[memory-watcher] Watching: ${params.dirs.join(", ")}`);
  } catch (err) {
    params.log.warn(`[memory-watcher] Failed to start: ${err instanceof Error ? err.message : String(err)}`);
    watcher = null;
  }
}

/**
 * Stop file watching
 */
export async function stopWatcher(log: PluginLogger): Promise<void> {
  if (debounceTimer) {
    clearTimeout(debounceTimer);
    debounceTimer = null;
  }
  if (watcher) {
    await watcher.close();
    watcher = null;
    watcherPromise = null;
    log.info("[memory-watcher] Stopped");
  }
}

/**
 * Check if watcher is running
 */
export function isWatching(): boolean {
  return watcher !== null;
}
