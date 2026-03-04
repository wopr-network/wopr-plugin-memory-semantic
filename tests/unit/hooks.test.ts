/**
 * Hook edge case tests (WOP-1591)
 *
 * Tests handleBeforeInject and handleAfterInject covering early-return guards,
 * error isolation, real-time indexing, multi-scale chunking, and auto-capture.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock all dependencies before importing
vi.mock("../../src/recall.js", () => ({
  performAutoRecall: vi.fn(),
}));
vi.mock("../../src/chunking.js", () => ({
  multiScaleChunk: vi.fn(),
}));
vi.mock("../../src/capture.js", () => ({
  extractFromConversation: vi.fn(),
}));
vi.mock("../../src/manifest.js", () => ({
  contentHash: vi.fn((s: string) => `hash-${s.slice(0, 10)}`),
}));

import { handleBeforeInject, handleAfterInject } from "../../src/hooks.js";
import { performAutoRecall } from "../../src/recall.js";
import { multiScaleChunk } from "../../src/chunking.js";
import { extractFromConversation } from "../../src/capture.js";
import { contentHash } from "../../src/manifest.js";

function makeLog() {
  return { info: vi.fn(), error: vi.fn() };
}

function makeState(overrides: Record<string, any> = {}) {
  return {
    initialized: true,
    searchManager: { hasEntry: vi.fn(() => false) },
    config: {
      autoRecall: { enabled: true },
      autoCapture: { enabled: false },
      chunking: { multiScale: { enabled: false, scales: [] } },
    },
    instanceId: "test-instance",
    ...overrides,
  };
}

function makeQueue() {
  return { enqueue: vi.fn() };
}

describe("handleBeforeInject", () => {
  beforeEach(() => vi.clearAllMocks());

  it("returns early when state.initialized is false", async () => {
    const state = makeState({ initialized: false });
    await handleBeforeInject(state as any, makeLog(), { message: "hello world" });
    expect(performAutoRecall).not.toHaveBeenCalled();
  });

  it("returns early when searchManager is null", async () => {
    const state = makeState({ searchManager: null });
    await handleBeforeInject(state as any, makeLog(), { message: "hello world" });
    expect(performAutoRecall).not.toHaveBeenCalled();
  });

  it("returns early when autoRecall is disabled", async () => {
    const state = makeState();
    state.config.autoRecall.enabled = false;
    await handleBeforeInject(state as any, makeLog(), { message: "hello world" });
    expect(performAutoRecall).not.toHaveBeenCalled();
  });

  it("returns early when payload is null", async () => {
    await handleBeforeInject(makeState() as any, makeLog(), null);
    expect(performAutoRecall).not.toHaveBeenCalled();
  });

  it("returns early when payload.message is empty string", async () => {
    await handleBeforeInject(makeState() as any, makeLog(), { message: "   " });
    expect(performAutoRecall).not.toHaveBeenCalled();
  });

  it("returns early when payload.message is not a string", async () => {
    await handleBeforeInject(makeState() as any, makeLog(), { message: 42 });
    expect(performAutoRecall).not.toHaveBeenCalled();
  });

  it("calls performAutoRecall and prepends context to payload.message", async () => {
    const state = makeState();
    const payload = { message: "what is the project about?" };
    vi.mocked(performAutoRecall).mockResolvedValue({
      query: "project about",
      memories: [{ id: "m1", snippet: "WOPR is a network", score: 0.9 }],
      context: "[Memory: WOPR is a network]",
    } as any);

    await handleBeforeInject(state as any, makeLog(), payload);

    expect(performAutoRecall).toHaveBeenCalledWith(
      "what is the project about?",
      state.searchManager,
      state.config,
      "test-instance",
    );
    expect(payload.message).toBe("[Memory: WOPR is a network]\n\nwhat is the project about?");
  });

  it("does not modify payload when recall returns empty memories", async () => {
    const state = makeState();
    const payload = { message: "hello there" };
    vi.mocked(performAutoRecall).mockResolvedValue({
      query: "hello",
      memories: [],
      context: "",
    } as any);

    await handleBeforeInject(state as any, makeLog(), payload);
    expect(payload.message).toBe("hello there");
  });

  it("catches performAutoRecall errors and logs them without throwing", async () => {
    const state = makeState();
    const log = makeLog();
    vi.mocked(performAutoRecall).mockRejectedValue(new Error("embedding timeout"));

    await handleBeforeInject(state as any, log, { message: "test query" });

    expect(log.error).toHaveBeenCalledWith(
      expect.stringContaining("embedding timeout"),
    );
  });
});

describe("handleAfterInject", () => {
  beforeEach(() => vi.clearAllMocks());

  it("returns early when state.initialized is false", async () => {
    const state = makeState({ initialized: false });
    const queue = makeQueue();
    await handleAfterInject(state as any, makeLog(), queue as any, { response: "hello world resp" });
    expect(queue.enqueue).not.toHaveBeenCalled();
  });

  it("returns early when searchManager is null", async () => {
    const state = makeState({ searchManager: null });
    const queue = makeQueue();
    await handleAfterInject(state as any, makeLog(), queue as any, { response: "hello world resp" });
    expect(queue.enqueue).not.toHaveBeenCalled();
  });

  it("returns early when payload is null", async () => {
    const queue = makeQueue();
    await handleAfterInject(makeState() as any, makeLog(), queue as any, null);
    expect(queue.enqueue).not.toHaveBeenCalled();
  });

  it("returns early when payload.response is empty", async () => {
    const queue = makeQueue();
    await handleAfterInject(makeState() as any, makeLog(), queue as any, { response: "  " });
    expect(queue.enqueue).not.toHaveBeenCalled();
  });

  it("returns early when payload.response is not a string", async () => {
    const queue = makeQueue();
    await handleAfterInject(makeState() as any, makeLog(), queue as any, { response: 123 });
    expect(queue.enqueue).not.toHaveBeenCalled();
  });

  it("enqueues response entry when response is long enough (no multi-scale)", async () => {
    const state = makeState();
    const queue = makeQueue();
    const log = makeLog();
    const payload = {
      session: "my-session",
      message: "short",  // < 10 chars trimmed, should be skipped
      response: "This is a sufficiently long response for indexing",
    };

    await handleAfterInject(state as any, log, queue as any, payload);

    expect(queue.enqueue).toHaveBeenCalledTimes(1);
    const [entries, source] = queue.enqueue.mock.calls[0];
    expect(source).toMatch(/^realtime:/);
    expect(entries).toHaveLength(1);
    expect(entries[0].entry.source).toBe("realtime-assistant");
    expect(entries[0].entry.path).toBe("session:my-session");
    expect(entries[0].persist).toBe(true);
  });

  it("enqueues both user message and response when both are long enough", async () => {
    const state = makeState();
    const queue = makeQueue();
    const payload = {
      session: "sess1",
      message: "This user message is definitely long enough",
      response: "This assistant response is also long enough",
    };

    await handleAfterInject(state as any, makeLog(), queue as any, payload);

    // enqueue called twice: once for user, once for assistant
    expect(queue.enqueue).toHaveBeenCalledTimes(2);
    const userCall = queue.enqueue.mock.calls[0];
    const assistCall = queue.enqueue.mock.calls[1];
    expect(userCall[0][0].entry.source).toBe("realtime-user");
    expect(assistCall[0][0].entry.source).toBe("realtime-assistant");
  });

  it("uses 'unknown' as session name when payload.session is missing", async () => {
    const state = makeState();
    const queue = makeQueue();
    const payload = {
      response: "A long enough response for indexing test",
    };

    await handleAfterInject(state as any, makeLog(), queue as any, payload);

    const [entries] = queue.enqueue.mock.calls[0];
    expect(entries[0].entry.path).toBe("session:unknown");
  });

  it("uses multiScaleChunk when multi-scale is enabled", async () => {
    const state = makeState();
    state.config.chunking.multiScale = {
      enabled: true,
      scales: [{ chunkSize: 100, overlap: 20 }],
    };
    const queue = makeQueue();
    vi.mocked(multiScaleChunk).mockReturnValue([
      { entry: { id: "chunk-1", path: "p", startLine: 0, endLine: 0, source: "ms", snippet: "s", content: "c" }, text: "chunk text" },
    ] as any);

    const payload = {
      session: "ms-sess",
      response: "A response long enough for multi-scale chunking test",
    };

    await handleAfterInject(state as any, makeLog(), queue as any, payload);

    expect(multiScaleChunk).toHaveBeenCalled();
    const [entries] = queue.enqueue.mock.calls[0];
    expect(entries[0].persist).toBe(true);
  });

  it("enqueues capture candidates when autoCapture is enabled", async () => {
    const state = makeState();
    state.config.autoCapture.enabled = true;
    const queue = makeQueue();
    vi.mocked(extractFromConversation).mockReturnValue([
      { text: "Remember: project uses TypeScript", category: "preference" as any, confidence: 0.9, trigger: "remember" },
    ] as any);

    const payload = {
      session: "cap-sess",
      message: "Remember that this project uses TypeScript",
      response: "Got it, I will remember that this project uses TypeScript",
    };

    await handleAfterInject(state as any, makeLog(), queue as any, payload);

    // Should have calls for realtime indexing + capture
    const captureCall = queue.enqueue.mock.calls.find(
      ([, src]: [any, string]) => src.startsWith("auto-capture"),
    );
    expect(captureCall).toBeDefined();
    expect(captureCall![0][0].entry.source).toBe("auto-capture");
    expect(captureCall![0][0].entry.id).toMatch(/^cap-/);
  });

  it("does not run capture when autoCapture is disabled", async () => {
    const state = makeState();
    state.config.autoCapture.enabled = false;
    const queue = makeQueue();

    const payload = {
      session: "no-cap",
      response: "A long enough response to pass the guard",
    };

    await handleAfterInject(state as any, makeLog(), queue as any, payload);

    expect(extractFromConversation).not.toHaveBeenCalled();
  });

  it("catches errors in afterInject and logs without throwing", async () => {
    const state = makeState();
    const log = makeLog();
    const queue = makeQueue();
    // Force contentHash to throw
    vi.mocked(contentHash).mockImplementation(() => { throw new Error("hash boom"); });

    const payload = {
      session: "err-sess",
      message: "A long enough message to trigger indexing path",
      response: "A long enough response to trigger indexing path",
    };

    await handleAfterInject(state as any, log, queue as any, payload);

    expect(log.error).toHaveBeenCalledWith(
      expect.stringContaining("hash boom"),
    );

    // Restore normal mock
    vi.mocked(contentHash).mockImplementation((s: string) => `hash-${s.slice(0, 10)}`);
  });
});
