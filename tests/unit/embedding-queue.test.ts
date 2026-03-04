import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { EmbeddingQueue, type PendingEntry } from "../../src/embedding-queue.js";

function makeEntry(id: string, persist = false): PendingEntry {
  return {
    entry: { id, path: "test.md", startLine: 0, endLine: 1, source: "test", snippet: id, content: id },
    text: id,
    persist,
  };
}

function makeLogger() {
  return { info: vi.fn(), error: vi.fn() };
}

function makeSearchManager(opts: { failCount?: number; existingIds?: string[] } = {}) {
  let callCount = 0;
  const existingIds = new Set(opts.existingIds ?? []);
  return {
    hasEntry: vi.fn((id: string) => existingIds.has(id)),
    getEntryCount: vi.fn(() => 0),
    addEntriesBatch: vi.fn(async (batch: PendingEntry[]) => {
      callCount++;
      if (callCount <= (opts.failCount ?? 0)) {
        throw new Error("transient network error");
      }
      for (const e of batch) existingIds.add(e.entry.id);
      return batch.length;
    }),
  };
}

describe("EmbeddingQueue", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  // =========================================================================
  // Batch accumulation and drain ordering
  // =========================================================================

  describe("batch accumulation and drain ordering", () => {
    it("should process all enqueued entries through addEntriesBatch", async () => {
      const log = makeLogger();
      const queue = new EmbeddingQueue(log);
      const sm = makeSearchManager();
      queue.attach(sm as any);

      queue.enqueue([makeEntry("a"), makeEntry("b"), makeEntry("c")], "test");
      await vi.runAllTimersAsync();

      expect(sm.addEntriesBatch).toHaveBeenCalledTimes(1);
      const batch = sm.addEntriesBatch.mock.calls[0][0] as PendingEntry[];
      expect(batch).toHaveLength(3);
    });

    it("should process entries in FIFO order", async () => {
      const log = makeLogger();
      const queue = new EmbeddingQueue(log);
      const sm = makeSearchManager();
      queue.attach(sm as any);

      queue.enqueue([makeEntry("first"), makeEntry("second"), makeEntry("third")], "test");
      await vi.runAllTimersAsync();

      const batch = sm.addEntriesBatch.mock.calls[0][0] as PendingEntry[];
      expect(batch.map((e) => e.entry.id)).toEqual(["first", "second", "third"]);
    });

    it("should split into multiple batches when queue exceeds 500 entries", async () => {
      const log = makeLogger();
      const queue = new EmbeddingQueue(log);
      const sm = makeSearchManager();
      queue.attach(sm as any);

      const entries = Array.from({ length: 600 }, (_, i) => makeEntry(`e${i}`));
      queue.enqueue(entries, "test");
      await vi.runAllTimersAsync();

      expect(sm.addEntriesBatch).toHaveBeenCalledTimes(2);
      const first = sm.addEntriesBatch.mock.calls[0][0] as PendingEntry[];
      const second = sm.addEntriesBatch.mock.calls[1][0] as PendingEntry[];
      expect(first).toHaveLength(500);
      expect(second).toHaveLength(100);
      expect(first[0].entry.id).toBe("e0");
      expect(second[0].entry.id).toBe("e500");
    });
  });

  // =========================================================================
  // Deduplication
  // =========================================================================

  describe("deduplication", () => {
    it("should skip entries already in the search index", async () => {
      const log = makeLogger();
      const queue = new EmbeddingQueue(log);
      const sm = makeSearchManager({ existingIds: ["already-indexed"] });
      queue.attach(sm as any);

      queue.enqueue([makeEntry("already-indexed"), makeEntry("new")], "test");
      await vi.runAllTimersAsync();

      const batch = sm.addEntriesBatch.mock.calls[0][0] as PendingEntry[];
      expect(batch).toHaveLength(1);
      expect(batch[0].entry.id).toBe("new");
    });

    it("should deduplicate within a single enqueue call", async () => {
      const log = makeLogger();
      const queue = new EmbeddingQueue(log);
      const sm = makeSearchManager();
      queue.attach(sm as any);

      queue.enqueue([makeEntry("dup"), makeEntry("dup"), makeEntry("unique")], "test");
      await vi.runAllTimersAsync();

      const batch = sm.addEntriesBatch.mock.calls[0][0] as PendingEntry[];
      expect(batch).toHaveLength(2);
      expect(batch.map((e) => e.entry.id)).toEqual(["dup", "unique"]);
    });

    it("should deduplicate across consecutive enqueue calls", async () => {
      const log = makeLogger();
      const queue = new EmbeddingQueue(log);
      const sm = makeSearchManager();
      queue.attach(sm as any);

      queue.enqueue([makeEntry("x")], "first");
      queue.enqueue([makeEntry("x"), makeEntry("y")], "second");
      await vi.runAllTimersAsync();

      const allIds = sm.addEntriesBatch.mock.calls
        .flatMap((c) => c[0] as PendingEntry[])
        .map((e) => e.entry.id);
      expect(allIds.filter((id) => id === "x")).toHaveLength(1);
      expect(allIds.filter((id) => id === "y")).toHaveLength(1);
    });

    it("should be a no-op when searchManager is not attached", () => {
      const log = makeLogger();
      const queue = new EmbeddingQueue(log);
      // No attach() call
      expect(() => queue.enqueue([makeEntry("z")], "test")).not.toThrow();
    });
  });

  // =========================================================================
  // Retry on failure
  // =========================================================================

  describe("drain retry on failure", () => {
    it("should re-queue batch entries on transient error and succeed on retry", async () => {
      const log = makeLogger();
      const queue = new EmbeddingQueue(log);
      const sm = makeSearchManager({ failCount: 1 }); // fail first call, succeed second
      queue.attach(sm as any);

      const entry = makeEntry("a");
      queue.enqueue([entry], "test");

      await vi.runAllTimersAsync();

      expect(sm.addEntriesBatch).toHaveBeenCalledTimes(2); // 1 fail + 1 success
    });

    it("should drop entries after MAX_RETRIES (3) failures and log error", async () => {
      const log = makeLogger();
      const queue = new EmbeddingQueue(log);
      const sm = makeSearchManager({ failCount: 999 }); // always fail
      queue.attach(sm as any);

      const entry = makeEntry("b");
      queue.enqueue([entry], "test");

      await vi.runAllTimersAsync();

      expect(sm.addEntriesBatch).toHaveBeenCalledTimes(4); // 1 initial + 3 retries

      // Should have logged the permanent drop
      const dropLogs = log.error.mock.calls.filter(
        (c: string[]) => c[0].includes("permanently dropping")
      );
      expect(dropLogs.length).toBeGreaterThan(0);
    });

    it("should not lose entries on single transient failure during bootstrap", async () => {
      const log = makeLogger();
      const queue = new EmbeddingQueue(log);
      const sm = makeSearchManager({ failCount: 1 });
      sm.getEntryCount.mockReturnValue(1);
      queue.attach(sm as any);

      const bootstrapPromise = queue.bootstrap([makeEntry("c")]);
      await vi.runAllTimersAsync();
      const count = await bootstrapPromise;

      expect(sm.addEntriesBatch).toHaveBeenCalledTimes(2);
      expect(count).toBe(1);
    });

    it("should stop retrying when clear() is called during backoff", async () => {
      const log = makeLogger();
      const queue = new EmbeddingQueue(log);
      const sm = makeSearchManager({ failCount: 999 }); // always fail
      queue.attach(sm as any);

      queue.enqueue([makeEntry("d")], "test");

      // Flush microtasks so drain() reaches the backoff await after first failure
      await vi.advanceTimersByTimeAsync(0);
      expect(sm.addEntriesBatch).toHaveBeenCalledTimes(1);

      // Clear during the backoff — should cancel the timer and resolve the promise
      queue.clear();

      // Run any remaining timers; drain() should exit without retrying
      await vi.runAllTimersAsync();

      expect(sm.addEntriesBatch).toHaveBeenCalledTimes(1);
    });
  });

  // =========================================================================
  // Persist callback
  // =========================================================================

  describe("persist callback", () => {
    it("should call persistFn for entries with persist=true", async () => {
      const log = makeLogger();
      const queue = new EmbeddingQueue(log);
      const sm = makeSearchManager();
      const persistFn = vi.fn();
      queue.attach(sm as any, persistFn);

      queue.enqueue([makeEntry("a", true), makeEntry("b", false), makeEntry("c", true)], "test");
      await vi.runAllTimersAsync();

      expect(persistFn).toHaveBeenCalledTimes(2);
      expect(persistFn).toHaveBeenCalledWith("a");
      expect(persistFn).toHaveBeenCalledWith("c");
    });

    it("should not call persistFn for entries with persist=false or unset", async () => {
      const log = makeLogger();
      const queue = new EmbeddingQueue(log);
      const sm = makeSearchManager();
      const persistFn = vi.fn();
      queue.attach(sm as any, persistFn);

      queue.enqueue([makeEntry("a", false), makeEntry("b")], "test");
      await vi.runAllTimersAsync();

      expect(persistFn).not.toHaveBeenCalled();
    });
  });

  // =========================================================================
  // Bootstrap
  // =========================================================================

  describe("bootstrap", () => {
    it("should return entry count from searchManager after drain completes", async () => {
      const log = makeLogger();
      const queue = new EmbeddingQueue(log);
      const sm = makeSearchManager();
      sm.getEntryCount.mockReturnValue(42);
      queue.attach(sm as any);

      const count = await (async () => {
        const p = queue.bootstrap([makeEntry("a"), makeEntry("b")]);
        await vi.runAllTimersAsync();
        return p;
      })();

      expect(count).toBe(42);
    });

    it("should set bootstrapping=true during processing and false after", async () => {
      const log = makeLogger();
      const queue = new EmbeddingQueue(log);
      const sm = makeSearchManager();
      sm.getEntryCount.mockReturnValue(1);
      let duringProcess = false;
      sm.addEntriesBatch.mockImplementationOnce(async (batch: PendingEntry[]) => {
        duringProcess = queue.bootstrapping;
        return batch.length;
      });
      queue.attach(sm as any);

      expect(queue.bootstrapping).toBe(false);
      const p = queue.bootstrap([makeEntry("a")]);
      await vi.runAllTimersAsync();
      await p;

      expect(duringProcess).toBe(true);
      expect(queue.bootstrapping).toBe(false);
    });

    it("should process all bootstrap entries before resolving", async () => {
      const log = makeLogger();
      const queue = new EmbeddingQueue(log);
      const sm = makeSearchManager();
      sm.getEntryCount.mockReturnValue(3);
      const processed: string[] = [];
      sm.addEntriesBatch.mockImplementation(async (batch: PendingEntry[]) => {
        for (const e of batch) processed.push(e.entry.id);
        return batch.length;
      });
      queue.attach(sm as any);

      const p = queue.bootstrap([makeEntry("x"), makeEntry("y"), makeEntry("z")]);
      await vi.runAllTimersAsync();
      await p;

      expect(processed).toEqual(["x", "y", "z"]);
    });
  });

  // =========================================================================
  // clear()
  // =========================================================================

  describe("clear", () => {
    it("should prevent further processing after clear", async () => {
      const log = makeLogger();
      const queue = new EmbeddingQueue(log);
      const sm = makeSearchManager();
      queue.attach(sm as any);

      queue.enqueue([makeEntry("a")], "test");
      await vi.runAllTimersAsync();

      queue.clear();

      // After clear, enqueue should be no-op (no searchManager)
      queue.enqueue([makeEntry("b")], "after-clear");
      await vi.runAllTimersAsync();

      // Only the first entry was processed
      const allIds = sm.addEntriesBatch.mock.calls
        .flatMap((c) => c[0] as PendingEntry[])
        .map((e) => e.entry.id);
      expect(allIds).not.toContain("b");
    });

    it("should reset bootstrapping flag", () => {
      const log = makeLogger();
      const queue = new EmbeddingQueue(log);
      queue.clear();
      expect(queue.bootstrapping).toBe(false);
    });

    it("should not crash when clear() is called while drain() is mid-await", async () => {
      const log = makeLogger();
      const queue = new EmbeddingQueue(log);
      const sm = makeSearchManager();

      // Make addEntriesBatch hang until we resolve it
      let resolveAdd!: () => void;
      sm.addEntriesBatch.mockImplementationOnce(
        () => new Promise<number>((resolve) => {
          resolveAdd = () => resolve(1);
        }),
      );

      queue.attach(sm as any);
      queue.enqueue([makeEntry("a")], "test");

      // drain() is now awaiting addEntriesBatch
      await vi.advanceTimersByTimeAsync(0);
      expect(sm.addEntriesBatch).toHaveBeenCalledTimes(1);

      // clear() while drain is mid-await — should not crash
      const clearPromise = queue.clear();

      // Resolve the hanging addEntriesBatch — drain resumes with null searchManager
      resolveAdd();

      // Flush everything
      await clearPromise;
      await vi.runAllTimersAsync();

      // No crash — verify clear worked
      expect(log.error.mock.calls.filter((c: string[]) => c[0].includes("crash"))).toHaveLength(0);
    });

    it("should resolve via timeout when drainPromise never settles", async () => {
      const log = makeLogger();
      const queue = new EmbeddingQueue(log);
      const sm = makeSearchManager();

      // Make addEntriesBatch hang forever (never resolves or rejects)
      sm.addEntriesBatch.mockImplementationOnce(() => new Promise<number>(() => {}));

      queue.attach(sm as any);
      queue.enqueue([makeEntry("a")], "test");

      // Let drain() start and get stuck in addEntriesBatch
      await vi.advanceTimersByTimeAsync(0);
      expect(sm.addEntriesBatch).toHaveBeenCalledTimes(1);

      // clear() — should resolve after the 5000ms timeout, not hang forever
      const clearPromise = queue.clear();

      // Advance past the 5000ms drain timeout in clear()
      await vi.advanceTimersByTimeAsync(5001);

      await clearPromise;

      // Queue should be fully reset
      expect(queue.bootstrapping).toBe(false);
    });
  });
});
