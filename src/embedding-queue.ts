import type { SemanticSearchManager, VectorEntry } from "./search.js";

export type PendingEntry = { entry: Omit<VectorEntry, "embedding">; text: string; persist?: boolean };

export type PersistFn = (id: string) => void;

export interface EmbeddingQueueLogger {
  info(msg: string): void;
  error(msg: string): void;
}

export class EmbeddingQueue {
  private queue: PendingEntry[] = [];
  private processing = false;
  private _bootstrapping = false;
  private searchManager: SemanticSearchManager | null = null;
  private persistFn: PersistFn | null = null;
  private log: EmbeddingQueueLogger;
  private drainResolvers: Array<() => void> = [];

  constructor(log: EmbeddingQueueLogger) {
    this.log = log;
  }

  get bootstrapping(): boolean {
    return this._bootstrapping;
  }

  attach(sm: SemanticSearchManager, persistFn?: PersistFn): void {
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
    this.drain();
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
      while (this.queue.length > 0) {
        // Take a batch from the front of the queue
        const batch = this.queue.splice(0, Math.min(this.queue.length, 500));
        this.log.info(`[queue] processing batch: ${batch.length} entries (${this.queue.length} remaining)`);
        try {
          await this.searchManager.addEntriesBatch(batch);
          // Persist plugin-originated entries (real-time, capture) to SQLite
          if (this.persistFn) {
            for (const entry of batch) {
              if (entry.persist) this.persistFn(entry.entry.id);
            }
          }
        } catch (err) {
          this.log.error(`[queue] batch failed: ${err instanceof Error ? err.message : err}`);
        }
      }
    } finally {
      this.processing = false;
      // Notify all waiters that the drain cycle finished.
      const resolvers = this.drainResolvers.splice(0);
      for (const resolve of resolvers) resolve();
    }
  }

  private waitForDrain(): Promise<void> {
    if (!this.processing && this.queue.length === 0) return Promise.resolve();
    return new Promise<void>((resolve) => {
      this.drainResolvers.push(resolve);
    });
  }

  clear(): void {
    this.queue = [];
    this.processing = false;
    this._bootstrapping = false;
    this.searchManager = null;
    this.persistFn = null;
    // Resolve any pending drain waiters so they don't hang after shutdown.
    const resolvers = this.drainResolvers.splice(0);
    for (const resolve of resolvers) resolve();
  }
}
