/**
 * Unit tests for src/event-handlers.ts (WOP-1582)
 *
 * Tests handleFilesChanged and handleMemorySearch covering all code paths:
 * early-return guards, bootstrapping skip, delete-action skip, chunk filtering,
 * multi-scale chunking, plain enqueue, search delegation, and error isolation.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

vi.mock("../../src/chunking.js", () => ({
  multiScaleChunk: vi.fn(),
}));

import { handleFilesChanged, handleMemorySearch } from "../../src/event-handlers.js";
import { multiScaleChunk } from "../../src/chunking.js";

function makeLog() {
  return { info: vi.fn(), error: vi.fn() };
}

function makeSearchManager() {
  return {
    search: vi.fn(),
    hasEntry: vi.fn(() => false),
    getEntryCount: vi.fn(() => 0),
  };
}

function makeQueue(overrides: Record<string, any> = {}) {
  return {
    bootstrapping: false,
    enqueue: vi.fn(),
    ...overrides,
  };
}

function makeFilesChangedState(overrides: Record<string, any> = {}) {
  return {
    initialized: true,
    searchManager: makeSearchManager(),
    config: {
      chunking: { multiScale: { enabled: false, scales: [] } },
    },
    instanceId: "test-instance",
    ...overrides,
  };
}

function makeSearchState(overrides: Record<string, any> = {}) {
  return {
    initialized: true,
    searchManager: makeSearchManager(),
    instanceId: "test-instance",
    ...overrides,
  };
}

// -------------------------------------------------------------------
// handleFilesChanged
// -------------------------------------------------------------------
describe("handleFilesChanged", () => {
  beforeEach(() => vi.clearAllMocks());

  it("returns early when state.initialized is false", async () => {
    const state = makeFilesChangedState({ initialized: false });
    const queue = makeQueue();
    await handleFilesChanged(state as any, makeLog(), queue as any, { changes: [] });
    expect(queue.enqueue).not.toHaveBeenCalled();
  });

  it("returns early when searchManager is null", async () => {
    const state = makeFilesChangedState({ searchManager: null });
    const queue = makeQueue();
    await handleFilesChanged(state as any, makeLog(), queue as any, { changes: [] });
    expect(queue.enqueue).not.toHaveBeenCalled();
  });

  it("skips and logs when queue is bootstrapping", async () => {
    const state = makeFilesChangedState();
    const queue = makeQueue({ bootstrapping: true });
    const log = makeLog();
    await handleFilesChanged(state as any, log, queue as any, { changes: [] });
    expect(queue.enqueue).not.toHaveBeenCalled();
    expect(log.info).toHaveBeenCalledWith(expect.stringContaining("bootstrap"));
  });

  it("skips changes with action=delete", async () => {
    const state = makeFilesChangedState();
    const queue = makeQueue();
    const payload = {
      changes: [{ action: "delete", path: "foo.ts", chunks: [{ id: "c1", text: "some text here enough" }] }],
    };
    await handleFilesChanged(state as any, makeLog(), queue as any, payload);
    expect(queue.enqueue).not.toHaveBeenCalled();
  });

  it("skips changes without chunks", async () => {
    const state = makeFilesChangedState();
    const queue = makeQueue();
    await handleFilesChanged(state as any, makeLog(), queue as any, {
      changes: [{ action: "update", path: "foo.ts" }],
    });
    expect(queue.enqueue).not.toHaveBeenCalled();
  });

  it("skips chunks with text shorter than 10 chars", async () => {
    const state = makeFilesChangedState();
    const queue = makeQueue();
    await handleFilesChanged(state as any, makeLog(), queue as any, {
      changes: [{ action: "update", path: "foo.ts", chunks: [{ id: "c1", text: "short" }] }],
    });
    expect(queue.enqueue).not.toHaveBeenCalled();
  });

  it("skips chunks with no text", async () => {
    const state = makeFilesChangedState();
    const queue = makeQueue();
    await handleFilesChanged(state as any, makeLog(), queue as any, {
      changes: [{ action: "update", path: "foo.ts", chunks: [{ id: "c1" }] }],
    });
    expect(queue.enqueue).not.toHaveBeenCalled();
  });

  it("enqueues plain entry for valid chunk (no multi-scale)", async () => {
    const state = makeFilesChangedState();
    const queue = makeQueue();
    const payload = {
      changes: [
        {
          action: "update",
          absPath: "/abs/foo.ts",
          source: "git",
          chunks: [
            { id: "chunk-1", text: "This is a valid chunk with enough text", startLine: 1, endLine: 5 },
          ],
        },
      ],
    };

    await handleFilesChanged(state as any, makeLog(), queue as any, payload);

    expect(queue.enqueue).toHaveBeenCalledTimes(1);
    const [entries, label] = queue.enqueue.mock.calls[0];
    expect(label).toMatch(/filesChanged/);
    expect(entries).toHaveLength(1);
    expect(entries[0].entry.id).toBe("chunk-1");
    expect(entries[0].entry.path).toBe("/abs/foo.ts");
    expect(entries[0].entry.source).toBe("git");
    expect(entries[0].entry.instanceId).toBe("test-instance");
    expect(entries[0].text).toBe("This is a valid chunk with enough text");
  });

  it("falls back to change.path when absPath is missing", async () => {
    const state = makeFilesChangedState();
    const queue = makeQueue();
    await handleFilesChanged(state as any, makeLog(), queue as any, {
      changes: [
        {
          action: "update",
          path: "relative/bar.ts",
          chunks: [{ id: "c1", text: "Chunk text long enough to pass", startLine: 0, endLine: 2 }],
        },
      ],
    });
    const [entries] = queue.enqueue.mock.calls[0];
    expect(entries[0].entry.path).toBe("relative/bar.ts");
  });

  it("defaults source to 'memory' when change.source is missing", async () => {
    const state = makeFilesChangedState();
    const queue = makeQueue();
    await handleFilesChanged(state as any, makeLog(), queue as any, {
      changes: [
        {
          action: "update",
          path: "foo.ts",
          chunks: [{ id: "c1", text: "Long enough chunk text here", startLine: 0, endLine: 1 }],
        },
      ],
    });
    const [entries] = queue.enqueue.mock.calls[0];
    expect(entries[0].entry.source).toBe("memory");
  });

  it("uses multiScaleChunk when multi-scale is enabled", async () => {
    const state = makeFilesChangedState();
    (state.config as any).chunking.multiScale = { enabled: true, scales: [{ chunkSize: 100, overlap: 20 }] };
    const queue = makeQueue();
    vi.mocked(multiScaleChunk).mockReturnValue([
      { entry: { id: "ms-1", path: "p", startLine: 0, endLine: 0, source: "ms", snippet: "s", content: "c" }, text: "ms chunk" },
    ] as any);

    await handleFilesChanged(state as any, makeLog(), queue as any, {
      changes: [
        {
          action: "update",
          absPath: "/abs/baz.ts",
          chunks: [{ id: "base-1", text: "Long enough text for multi-scale chunking here", startLine: 0, endLine: 10 }],
        },
      ],
    });

    expect(multiScaleChunk).toHaveBeenCalled();
    const [entries] = queue.enqueue.mock.calls[0];
    expect(entries[0].entry.id).toBe("ms-1");
  });

  it("does not enqueue when all changes are empty/invalid", async () => {
    const state = makeFilesChangedState();
    const queue = makeQueue();
    await handleFilesChanged(state as any, makeLog(), queue as any, { changes: [] });
    expect(queue.enqueue).not.toHaveBeenCalled();
  });

  it("handles missing payload.changes gracefully", async () => {
    const state = makeFilesChangedState();
    const queue = makeQueue();
    await handleFilesChanged(state as any, makeLog(), queue as any, {});
    expect(queue.enqueue).not.toHaveBeenCalled();
  });
});

// -------------------------------------------------------------------
// handleMemorySearch
// -------------------------------------------------------------------
describe("handleMemorySearch", () => {
  beforeEach(() => vi.clearAllMocks());

  it("logs the query on entry", async () => {
    const state = makeSearchState({ initialized: false });
    const log = makeLog();
    const payload = { query: "my query", maxResults: 5, minScore: 0.5, sessionName: "s", results: null };
    await handleMemorySearch(state as any, log, payload);
    expect(log.info).toHaveBeenCalledWith(expect.stringContaining("my query"));
  });

  it("returns early when state.initialized is false", async () => {
    const state = makeSearchState({ initialized: false });
    const log = makeLog();
    const payload = { query: "test", maxResults: 5, minScore: 0.5, sessionName: "s", results: null };
    await handleMemorySearch(state as any, log, payload);
    expect(state.searchManager.search).not.toHaveBeenCalled();
    expect(payload.results).toBeNull();
  });

  it("returns early when searchManager is null", async () => {
    const state = makeSearchState({ searchManager: null });
    const log = makeLog();
    const payload = { query: "test", maxResults: 5, minScore: 0.5, sessionName: "s", results: null };
    await handleMemorySearch(state as any, log, payload);
    expect(payload.results).toBeNull();
  });

  it("sets payload.results to filtered search results", async () => {
    const state = makeSearchState();
    const log = makeLog();
    state.searchManager.search.mockResolvedValue([
      { id: "r1", score: 0.9, snippet: "high score" },
      { id: "r2", score: 0.3, snippet: "low score" },
      { id: "r3", score: 0.7, snippet: "medium score" },
    ]);
    const payload = { query: "test query", maxResults: 10, minScore: 0.5, sessionName: "s", results: null };

    await handleMemorySearch(state as any, log, payload);

    expect(state.searchManager.search).toHaveBeenCalledWith("test query", 10, "test-instance");
    expect(payload.results).toHaveLength(2);
    expect(payload.results![0].id).toBe("r1");
    expect(payload.results![1].id).toBe("r3");
  });

  it("returns empty array when all results are below minScore", async () => {
    const state = makeSearchState();
    state.searchManager.search.mockResolvedValue([
      { id: "r1", score: 0.2 },
      { id: "r2", score: 0.1 },
    ]);
    const payload = { query: "test", maxResults: 5, minScore: 0.5, sessionName: "s", results: null };

    await handleMemorySearch(state as any, makeLog(), payload);

    expect(payload.results).toEqual([]);
  });

  it("passes instanceId to search", async () => {
    const state = makeSearchState({ instanceId: "my-instance-id" });
    state.searchManager.search.mockResolvedValue([]);
    const payload = { query: "q", maxResults: 3, minScore: 0.0, sessionName: "s", results: null };

    await handleMemorySearch(state as any, makeLog(), payload);

    expect(state.searchManager.search).toHaveBeenCalledWith("q", 3, "my-instance-id");
  });

  it("passes undefined instanceId when not set", async () => {
    const state = makeSearchState({ instanceId: undefined });
    state.searchManager.search.mockResolvedValue([]);
    const payload = { query: "q", maxResults: 3, minScore: 0.0, sessionName: "s", results: null };

    await handleMemorySearch(state as any, makeLog(), payload);

    expect(state.searchManager.search).toHaveBeenCalledWith("q", 3, undefined);
  });

  it("catches search errors and logs without throwing", async () => {
    const state = makeSearchState();
    const log = makeLog();
    state.searchManager.search.mockRejectedValue(new Error("db connection failed"));
    const payload = { query: "test", maxResults: 5, minScore: 0.5, sessionName: "s", results: null };

    await handleMemorySearch(state as any, log, payload);

    expect(log.error).toHaveBeenCalledWith(expect.stringContaining("db connection failed"));
    // results should remain null (not set on error)
    expect(payload.results).toBeNull();
  });

  it("catches non-Error exceptions and logs string form", async () => {
    const state = makeSearchState();
    const log = makeLog();
    state.searchManager.search.mockRejectedValue("string error");
    const payload = { query: "q", maxResults: 5, minScore: 0.0, sessionName: "s", results: null };

    await handleMemorySearch(state as any, log, payload);

    expect(log.error).toHaveBeenCalledWith(expect.stringContaining("string error"));
  });
});
