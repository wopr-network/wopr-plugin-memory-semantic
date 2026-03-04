import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { EmbeddingQueue, type PendingEntry } from "../../src/embedding-queue.js";

function makeEntry(id: string): PendingEntry {
  return {
    entry: { id, path: "test.md", startLine: 0, endLine: 1, source: "test", snippet: id, content: id },
    text: id,
  };
}

function makeLogger() {
  return { info: vi.fn(), error: vi.fn() };
}

function makeSearchManager(opts: { failCount?: number } = {}) {
  let callCount = 0;
  return {
    hasEntry: vi.fn(() => false),
    getEntryCount: vi.fn(() => 0),
    addEntriesBatch: vi.fn(async () => {
      callCount++;
      if (callCount <= (opts.failCount ?? 0)) {
        throw new Error("transient network error");
      }
      return 1;
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
});
