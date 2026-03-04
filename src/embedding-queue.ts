import type { SemanticSearchManager, VectorEntry } from "./search.js";

const MAX_RETRIES = 3;
const BASE_BACKOFF_MS = 1000;

export type PendingEntry = { entry: Omit<VectorEntry, "embedding">; text: string; persist?: boolean };

type QueuedEntry = PendingEntry & { _retryCount?: number };

export type PersistFn = (id: string) => void;

export interface EmbeddingQueueLogger {
  info(msg: string): void;
  error(msg: string): void;
}

export class EmbeddingQueue {
  private queue: QueuedEntry[] = [];
  private processing = false;
  private _bootstrapping = false;
  private stopped = false;
  private backoffTimer: ReturnType<typeof setTimeout> | null = null;
  private backoffResolve: (() => void) | null = null;
  private drainPromise: Promise<void> | null = null;
  private searchManager: SemanticSearchManager | null = null;
  private persistFn: PersistFn | null = null;
  private log: EmbeddingQueueLogger;

  constructor(log: EmbeddingQueueLogger) {
    this.log = log;
  }

  setLogger(log: EmbeddingQueueLogger): void {
    this.log = log;
  }

  get bootstrapping(): boolean {
    return this._bootstrapping;
  }

  attach(sm: SemanticSearchManager, persistFn?: PersistFn): void {
    this.stopped = false;
    this.searchManager = sm;
    this.persistFn = persistFn ?? null;
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
    this.log.info(`[queue] enqueued ${added} entries from ${source} (${this.queue.length} total pending)`);
    if (!this.drainPromise) {
      const p = this.drain().finally(() => {
        if (this.drainPromise === p) this.drainPromise = null;
      });
      this.drainPromise = p;
    }
  }

  /** Run bootstrap: enqueue all chunks and process to completion before anything else. */
  async bootstrap(entries: PendingEntry[]): Promise<number> {
    this._bootstrapping = true;
    this.log.info(`[queue] bootstrap starting: ${entries.length} entries`);
    this.enqueue(entries, "bootstrap");
    // Wait for the queue to fully drain
    await this.waitForDrain();
    this._bootstrapping = false;
    const count = this.searchManager?.getEntryCount() ?? 0;
    this.log.info(`[queue] bootstrap complete: ${count} vectors in index`);
    return count;
  }

  /** Process the queue sequentially — only one batch at a time. */
  private async drain(): Promise<void> {
    if (this.processing || this.queue.length === 0 || !this.searchManager) return;
    this.processing = true;

    try {
      while (this.queue.length > 0 && !this.stopped) {
        // Take a batch from the front of the queue
        const batch = this.queue.splice(0, Math.min(this.queue.length, 500));
        this.log.info(`[queue] processing batch: ${batch.length} entries (${this.queue.length} remaining)`);
        try {
          await this.searchManager.addEntriesBatch(batch);
          if (this.stopped) break;
          // Persist plugin-originated entries (real-time, capture) to SQLite
          if (this.persistFn) {
            for (const entry of batch) {
              if (entry.persist) this.persistFn(entry.entry.id);
            }
          }
        } catch (err) {
          this.log.error(`[queue] batch failed: ${err instanceof Error ? err.message : err}`);
          if (this.stopped) break;
          // Re-queue entries that haven't exceeded retry limit
          const retriable: QueuedEntry[] = [];
          const dropped: QueuedEntry[] = [];
          for (const entry of batch) {
            const retryCount = (entry._retryCount ?? 0) + 1;
            entry._retryCount = retryCount;
            if (retryCount <= MAX_RETRIES) {
              retriable.push(entry);
            } else {
              dropped.push(entry);
            }
          }
          if (dropped.length > 0) {
            this.log.error(
              `[queue] permanently dropping ${dropped.length} entries after ${MAX_RETRIES} retries: ${dropped.map((e) => e.entry.id).join(", ")}`,
            );
          }
          if (retriable.length > 0) {
            // Push to back of queue so new entries aren't starved by repeated failures
            this.queue.push(...retriable);
            const maxRetry = Math.max(...retriable.map((e) => e._retryCount ?? 1));
            const backoffMs = BASE_BACKOFF_MS * 2 ** (maxRetry - 1);
            this.log.info(`[queue] re-queued ${retriable.length} entries, backoff ${backoffMs}ms`);
            await new Promise<void>((resolve) => {
              this.backoffResolve = resolve;
              this.backoffTimer = setTimeout(() => {
                this.backoffTimer = null;
                this.backoffResolve = null;
                resolve();
              }, backoffMs);
            });
            if (this.stopped) break;
          }
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

  async clear(): Promise<void> {
    this.stopped = true;
    if (this.backoffTimer !== null) {
      clearTimeout(this.backoffTimer);
      this.backoffTimer = null;
    }
    if (this.backoffResolve !== null) {
      this.backoffResolve();
      this.backoffResolve = null;
    }
    // Wait for any in-flight drain to finish (it will exit because stopped=true)
    if (this.drainPromise) {
      await Promise.race([
        this.drainPromise.catch((err) => {
          this.log.error(`[queue] drain error during clear: ${err instanceof Error ? err.message : err}`);
        }),
        new Promise<void>((resolve) => setTimeout(resolve, 5000)),
      ]);
      this.drainPromise = null;
    }
    this.queue = [];
    this.processing = false;
    this._bootstrapping = false;
    this.searchManager = null;
    this.persistFn = null;
  }
}
