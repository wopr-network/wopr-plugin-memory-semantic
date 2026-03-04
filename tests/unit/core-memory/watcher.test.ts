import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { mockLogger } from "./helpers.js";

// Mock chokidar so tests are deterministic and don't start real FS watchers.
// watcher.ts uses await import("chokidar") which vi.mock() can intercept.
vi.mock("chokidar", () => ({
  watch: vi.fn(),
}));

import { startWatcher, stopWatcher, isWatching } from "../../../src/core-memory/watcher.js";
import { watch as mockChokidarWatch } from "chokidar";

/** Returns a fake FSWatcher that immediately resolves the "ready" event. */
function makeFakeWatcher() {
  const fakeWatcher = {
    on: vi.fn().mockImplementation((event: string, handler: (...args: unknown[]) => void) => {
      if (event === "ready") Promise.resolve().then(() => handler());
      return fakeWatcher;
    }),
    close: vi.fn().mockResolvedValue(undefined),
  };
  return fakeWatcher;
}

beforeEach(async () => {
  // Reset watcher singleton state between tests
  if (isWatching()) await stopWatcher(mockLogger());
  vi.clearAllMocks();
  // Default: chokidar works and emits ready
  vi.mocked(mockChokidarWatch).mockImplementation(makeFakeWatcher as never);
});

afterEach(async () => {
  // Ensure no open watcher handles leak between tests
  if (isWatching()) await stopWatcher(mockLogger());
});

describe("watcher - initial state", () => {
  it("isWatching returns false before any startWatcher call", () => {
    expect(isWatching()).toBe(false);
  });
});

describe("watcher - stopWatcher when not watching", () => {
  it("resolves without error when not watching", async () => {
    await expect(stopWatcher(mockLogger())).resolves.not.toThrow();
  });

  it("does not log info when not watching", async () => {
    const log = mockLogger();
    await stopWatcher(log);
    expect(vi.mocked(log.info)).not.toHaveBeenCalled();
  });
});

describe("watcher - startWatcher failure path", () => {
  it("warns and leaves isWatching=false when chokidar throws", async () => {
    vi.mocked(mockChokidarWatch).mockImplementationOnce(() => {
      throw new Error("chokidar unavailable");
    });
    const log = mockLogger();
    await startWatcher({ dirs: ["/workspace"], debounceMs: 100, onSync: vi.fn(), log });
    expect(isWatching()).toBe(false);
    expect(vi.mocked(log.warn)).toHaveBeenCalledWith(
      expect.stringContaining("[memory-watcher]"),
    );
  });
});

describe("watcher - isWatching type", () => {
  it("returns a boolean", () => {
    expect(typeof isWatching()).toBe("boolean");
  });
});

describe("watcher - startWatcher called twice returns early", () => {
  it("does not throw when called multiple times", async () => {
    const log = mockLogger();
    const onSync = vi.fn();
    await expect(startWatcher({ dirs: ["/workspace"], debounceMs: 100, onSync, log })).resolves.not.toThrow();
    await expect(startWatcher({ dirs: ["/workspace"], debounceMs: 100, onSync, log })).resolves.not.toThrow();
    // Second call should be a no-op (watcher already running)
    expect(vi.mocked(mockChokidarWatch)).toHaveBeenCalledTimes(1);
  });
});

describe("watcher - WatcherCallback type", () => {
  it("accepts an async callback and resolves", async () => {
    const log = mockLogger();
    const onSync = vi.fn().mockResolvedValue(undefined);
    await expect(
      startWatcher({ dirs: ["/workspace"], debounceMs: 50, onSync, log }),
    ).resolves.not.toThrow();
    expect(isWatching()).toBe(true);
  });
});
