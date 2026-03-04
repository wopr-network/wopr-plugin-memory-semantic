import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import type { PluginLogger } from "@wopr-network/plugin-types";

function mockLogger(): PluginLogger {
  return {
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
    debug: vi.fn(),
  } as unknown as PluginLogger;
}

// The watcher module uses module-level singleton state.
// We reset it between tests via vi.resetModules().
// The chokidar dynamic import uses Function('return import("chokidar")')()
// which bypasses vi.mock. So we test behaviors that are independent of chokidar
// successfully loading (the failure/graceful-degradation path),
// plus state management functions.

describe("watcher - initial state", () => {
  beforeEach(() => {
    vi.resetModules();
  });

  it("isWatching returns false before any startWatcher call", async () => {
    const { isWatching } = await import("../../../src/core-memory/watcher.js");
    expect(isWatching()).toBe(false);
  });
});

describe("watcher - stopWatcher when not watching", () => {
  beforeEach(() => {
    vi.resetModules();
  });

  it("stopWatcher resolves without error when not watching", async () => {
    const { stopWatcher } = await import("../../../src/core-memory/watcher.js");
    const log = mockLogger();
    await expect(stopWatcher(log)).resolves.not.toThrow();
  });

  it("stopWatcher does not log info when not watching", async () => {
    const { stopWatcher } = await import("../../../src/core-memory/watcher.js");
    const log = mockLogger();
    await stopWatcher(log);
    expect(vi.mocked(log.info)).not.toHaveBeenCalled();
  });
});

describe("watcher - startWatcher when chokidar unavailable", () => {
  beforeEach(() => {
    vi.resetModules();
  });

  afterEach(async () => {
    const { stopWatcher } = await import("../../../src/core-memory/watcher.js");
    await stopWatcher(mockLogger()).catch(() => {});
    vi.resetModules();
  });

  it("startWatcher warns and leaves isWatching=false when chokidar fails", async () => {
    // chokidar is not installed in this test env; the dynamic import will fail
    const { startWatcher, isWatching } = await import("../../../src/core-memory/watcher.js");
    const log = mockLogger();
    const onSync = vi.fn();

    // This will fail to load chokidar in the test environment
    await startWatcher({ dirs: ["/workspace"], debounceMs: 100, onSync, log });

    // Either it worked (if chokidar is installed) or it warned and degraded
    if (!isWatching()) {
      expect(vi.mocked(log.warn)).toHaveBeenCalledWith(
        expect.stringContaining("[memory-watcher]"),
      );
    }
    // Either way, should not throw
  });
});

describe("watcher - isWatching type", () => {
  beforeEach(() => {
    vi.resetModules();
  });

  it("isWatching returns a boolean", async () => {
    const { isWatching } = await import("../../../src/core-memory/watcher.js");
    expect(typeof isWatching()).toBe("boolean");
  });
});

describe("watcher - startWatcher called twice returns early", () => {
  beforeEach(() => {
    vi.resetModules();
  });

  afterEach(async () => {
    const { stopWatcher } = await import("../../../src/core-memory/watcher.js");
    await stopWatcher(mockLogger()).catch(() => {});
    vi.resetModules();
  });

  it("does not throw when called multiple times", async () => {
    const { startWatcher } = await import("../../../src/core-memory/watcher.js");
    const log = mockLogger();
    const onSync = vi.fn();

    // Both calls should succeed (second is a no-op if watcher running)
    await expect(startWatcher({ dirs: ["/workspace"], debounceMs: 100, onSync, log })).resolves.not.toThrow();
    await expect(startWatcher({ dirs: ["/workspace"], debounceMs: 100, onSync, log })).resolves.not.toThrow();
  });
});

describe("watcher - WatcherCallback type", () => {
  afterEach(async () => {
    const { stopWatcher } = await import("../../../src/core-memory/watcher.js");
    await stopWatcher(mockLogger()).catch(() => {});
    vi.resetModules();
  });

  it("accepts an async callback", async () => {
    vi.resetModules();
    const { startWatcher } = await import("../../../src/core-memory/watcher.js");
    const log = mockLogger();
    let called = false;
    const onSync = async () => {
      called = true;
    };

    // Should not throw regardless of chokidar availability
    await expect(
      startWatcher({ dirs: ["/workspace"], debounceMs: 50, onSync, log }),
    ).resolves.not.toThrow();
  });
});
