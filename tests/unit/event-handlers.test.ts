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
  return { info: vi.fn(), error: vi.fn(), debug: vi.fn() };
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
  beforeEach(() => vi.resetAllMocks());

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

  it("filters out invalid chunks across multiple changes, enqueuing only valid ones", async () => {
    const state = makeFilesChangedState();
    const queue = makeQueue();
    await handleFilesChanged(state as any, makeLog(), queue as any, {
      changes: [
        // valid change with one valid and one too-short chunk
        {
          action: "update",
          absPath: "/abs/a.ts",
          source: "git",
          chunks: [
            { id: "valid-1", text: "This chunk is long enough to pass the filter", startLine: 0, endLine: 5 },
            { id: "short-1", text: "tiny", startLine: 6, endLine: 6 },
          ],
        },
        // delete action — all chunks skipped
        {
          action: "delete",
          absPath: "/abs/b.ts",
          chunks: [{ id: "del-1", text: "Would be valid but action is delete", startLine: 0, endLine: 2 }],
        },
        // change with no chunks at all
        { action: "update", absPath: "/abs/c.ts" },
        // change with a chunk that has no text
        {
          action: "update",
          absPath: "/abs/d.ts",
          chunks: [{ id: "notext-1", startLine: 0, endLine: 1 }],
        },
        // valid change with one valid chunk
        {
          action: "update",
          absPath: "/abs/e.ts",
          source: "editor",
          chunks: [
            { id: "valid-2", text: "Another chunk that is long enough to pass", startLine: 0, endLine: 3 },
          ],
        },
      ],
    });

    expect(queue.enqueue).toHaveBeenCalledTimes(1);
    const [entries] = queue.enqueue.mock.calls[0];
    expect(entries).toHaveLength(2);
    expect(entries.map((e: any) => e.entry.id)).toEqual(["valid-1", "valid-2"]);
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
    expect(queue.enqueue).toHaveBeenCalledTimes(1);
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
    expect(queue.enqueue).toHaveBeenCalledTimes(1);
    const [entries] = queue.enqueue.mock.calls[0];
    expect(entries[0].entry.source).toBe("memory");
  });

  it("uses multiScaleChunk when multi-scale is enabled", async () => {
    const state = makeFilesChangedState();
    (state.config as any).chunking.multiScale = { enabled: true, scales: [{ tokens: 100, overlap: 20 }] };
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
  beforeEach(() => vi.resetAllMocks());

  it("logs the query at debug level (not info) on entry", async () => {
    const state = makeSearchState({ initialized: false });
    const log = makeLog();
    const payload = { query: "my query", maxResults: 5, minScore: 0.5, sessionName: "s", results: null };
    await handleMemorySearch(state as any, log, payload);
    expect(log.debug).toHaveBeenCalledWith(expect.stringContaining("my query"));
    for (const call of log.info.mock.calls) {
      expect(call[0]).not.toContain("my query");
    }
  });

  it("truncates long queries in debug log to limit PII exposure", async () => {
    const state = makeSearchState({ initialized: false });
    const log = makeLog();
    const longQuery = "a".repeat(80);
    const payload = { query: longQuery, maxResults: 5, minScore: 0.5, sessionName: "s", results: null };
    await handleMemorySearch(state as any, log, payload);
    expect(log.debug).toHaveBeenCalledTimes(1);
    const debugMsg: string = log.debug.mock.calls[0][0];
    // Should contain truncated preview (60 chars + ellipsis), not the full query
    expect(debugMsg).toContain("a".repeat(60) + "…");
    expect(debugMsg).not.toContain("a".repeat(61));
    // Should include the full query length for diagnostics
    expect(debugMsg).toContain(`length=${longQuery.length}`);
  });

  it("does not truncate short queries in debug log", async () => {
    const state = makeSearchState({ initialized: false });
    const log = makeLog();
    const shortQuery = "find relevant docs";
    const payload = { query: shortQuery, maxResults: 5, minScore: 0.5, sessionName: "s", results: null };
    await handleMemorySearch(state as any, log, payload);
    expect(log.debug).toHaveBeenCalledTimes(1);
    const debugMsg: string = log.debug.mock.calls[0][0];
    expect(debugMsg).toContain(shortQuery);
    expect(debugMsg).not.toContain("…");
  });

  it("works when log has no debug method (optional interface)", async () => {
    const state = makeSearchState({ initialized: false });
    const log = { info: vi.fn(), error: vi.fn() }; // no debug
    const payload = { query: "test", maxResults: 5, minScore: 0.5, sessionName: "s", results: null };
    // Should not throw even without a debug method
    await expect(handleMemorySearch(state as any, log, payload)).resolves.toBeUndefined();
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
