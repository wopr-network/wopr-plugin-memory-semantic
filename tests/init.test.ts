import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Mock all heavy dependencies so init.ts never touches real FS/network
vi.mock("../src/a2a-tools.js", () => ({ registerMemoryTools: vi.fn() }));
vi.mock("../src/core-memory/manager.js", () => ({
  MemoryIndexManager: { create: vi.fn() },
}));
vi.mock("../src/core-memory/session-hook.js", () => ({
  createSessionDestroyHandler: vi.fn().mockResolvedValue(vi.fn()),
}));
vi.mock("../src/core-memory/watcher.js", () => ({
  startWatcher: vi.fn().mockResolvedValue(undefined),
}));
vi.mock("../src/embeddings.js", () => ({
  createEmbeddingProvider: vi.fn(),
}));
vi.mock("../src/search.js", () => ({
  createSemanticSearchManager: vi.fn(),
}));
vi.mock("../src/persistence.js", () => ({
  loadChunkMetadata: vi.fn().mockResolvedValue(new Map()),
  getHnswPath: vi.fn().mockReturnValue("/tmp/test.hnsw"),
  persistNewEntryToDb: vi.fn(),
}));
vi.mock("../src/memory-schema.js", () => ({
  createMemoryPluginSchema: vi.fn().mockReturnValue({}),
}));

import { initialize } from "../src/init.js";
import type { InitLogger, PluginState } from "../src/init.js";
import { MemoryIndexManager } from "../src/core-memory/manager.js";
import { createEmbeddingProvider } from "../src/embeddings.js";
import { createSemanticSearchManager } from "../src/search.js";
import { DEFAULT_CONFIG } from "../src/types.js";
import type { EmbeddingProvider } from "../src/types.js";
import type { SemanticSearchManager } from "../src/search.js";
import { EmbeddingQueue } from "../src/embedding-queue.js";

function makeApi() {
  return {
    log: { info: vi.fn(), warn: vi.fn(), error: vi.fn() },
    storage: {
      register: vi.fn().mockResolvedValue(undefined),
      raw: vi.fn().mockResolvedValue(undefined),
    },
    events: { on: vi.fn().mockReturnValue(vi.fn()) },
  } as unknown as Parameters<typeof initialize>[0];
}

function makeState(): PluginState {
  return {
    config: DEFAULT_CONFIG,
    embeddingProvider: null,
    searchManager: null,
    memoryManager: null,
    api: null,
    initialized: false,
    eventCleanup: [],
    instanceId: undefined,
  };
}

function makeLog(): InitLogger {
  return { info: vi.fn(), warn: vi.fn(), error: vi.fn() };
}

function makeQueue(): EmbeddingQueue {
  return {
    attach: vi.fn(),
    bootstrap: vi.fn().mockResolvedValue(0),
  } as unknown as EmbeddingQueue;
}

describe("initialize failure paths", () => {
  let api: ReturnType<typeof makeApi>;
  let state: PluginState;
  let log: InitLogger;
  let queue: EmbeddingQueue;
  let savedInstanceId: string | undefined;

  beforeEach(() => {
    vi.clearAllMocks();
    api = makeApi();
    state = makeState();
    log = makeLog();
    queue = makeQueue();
    savedInstanceId = process.env.WOPR_INSTANCE_ID;

    // Happy-path defaults (tests override the one they want to break)
    vi.mocked(MemoryIndexManager.create).mockResolvedValue({
      sync: vi.fn().mockResolvedValue(undefined),
      search: vi.fn().mockResolvedValue([]),
    } as unknown as MemoryIndexManager);
    vi.mocked(createEmbeddingProvider).mockResolvedValue({
      id: "mock-provider",
      model: "test-model",
      embedQuery: vi.fn<[string], Promise<number[]>>().mockResolvedValue([0.1, 0.2]),
      embedBatch: vi.fn<[string[]], Promise<number[][]>>().mockResolvedValue([[0.1, 0.2]]),
    } satisfies EmbeddingProvider);
    vi.mocked(createSemanticSearchManager).mockResolvedValue({
      getEntryCount: vi.fn().mockReturnValue(0),
      search: vi.fn().mockResolvedValue([]),
      addEntry: vi.fn().mockResolvedValue(undefined),
      addEntriesBatch: vi.fn().mockResolvedValue(0),
      close: vi.fn().mockResolvedValue(undefined),
      hasEntry: vi.fn().mockReturnValue(false),
      getEntry: vi.fn().mockReturnValue(undefined),
    } satisfies SemanticSearchManager);
  });

  afterEach(() => {
    if (savedInstanceId === undefined) {
      delete process.env.WOPR_INSTANCE_ID;
    } else {
      process.env.WOPR_INSTANCE_ID = savedInstanceId;
    }
  });

  it("logs error when storage.register throws", async () => {
    vi.mocked(api.storage.register).mockRejectedValue(new Error("DB locked"));

    await initialize(api, state, queue, log);

    expect(api.log.error).toHaveBeenCalledWith(
      expect.stringContaining("DB locked"),
    );
    expect(state.initialized).toBe(false);
  });

  it("logs error when FTS5 virtual table creation fails", async () => {
    vi.mocked(api.storage.raw).mockRejectedValueOnce(new Error("FTS5 not available"));

    await initialize(api, state, queue, log);

    expect(api.log.error).toHaveBeenCalledWith(
      expect.stringContaining("FTS5 not available"),
    );
    expect(state.initialized).toBe(false);
  });

  it("logs error when MemoryIndexManager.create throws", async () => {
    vi.mocked(MemoryIndexManager.create).mockRejectedValue(
      new Error("corrupt index file"),
    );

    await initialize(api, state, queue, log);

    expect(api.log.error).toHaveBeenCalledWith(
      expect.stringContaining("corrupt index file"),
    );
    expect(state.initialized).toBe(false);
    expect(state.memoryManager).toBeNull();
  });

  it("logs error when createEmbeddingProvider throws", async () => {
    vi.mocked(createEmbeddingProvider).mockRejectedValue(
      new Error("No API key found for OpenAI"),
    );

    await initialize(api, state, queue, log);

    expect(api.log.error).toHaveBeenCalledWith(
      expect.stringContaining("No API key found"),
    );
    expect(state.initialized).toBe(false);
    expect(state.embeddingProvider).toBeNull();
  });

  it("logs error when createSemanticSearchManager throws", async () => {
    vi.mocked(createSemanticSearchManager).mockRejectedValue(
      new Error("dimension mismatch: expected 1536, got 384"),
    );

    await initialize(api, state, queue, log);

    expect(api.log.error).toHaveBeenCalledWith(
      expect.stringContaining("dimension mismatch"),
    );
    expect(state.initialized).toBe(false);
    expect(state.searchManager).toBeNull();
  });

  it("skips initialization when already initialized", async () => {
    state.initialized = true;

    await initialize(api, state, queue, log);

    // Should return immediately — no storage calls
    expect(api.storage.register).not.toHaveBeenCalled();
  });

  it("skips initialization when init is already in progress", async () => {
    // Call initialize without awaiting to set initInProgress=true
    const first = initialize(api, state, queue, log);
    // Second call should bail immediately
    const second = initialize(api, state, queue, log);

    await first;
    await second;

    // storage.register called exactly once (from first call)
    expect(api.storage.register).toHaveBeenCalledTimes(1);
  });

  it("resets initInProgress flag even after failure", async () => {
    vi.mocked(api.storage.register).mockRejectedValue(new Error("fail"));

    await initialize(api, state, queue, log);

    // initInProgress should be false again — a fresh state can init
    const freshState = makeState();
    vi.mocked(api.storage.register).mockResolvedValue(undefined);

    await initialize(api, freshState, queue, log);

    expect(freshState.initialized).toBe(true);
  });

  it("warns when no instanceId is configured", async () => {
    delete process.env.WOPR_INSTANCE_ID;

    await initialize(api, state, queue, log, {
      provider: "openai",
      model: "test",
    } as Partial<Parameters<typeof initialize>[4]>);

    expect(api.log.warn).toHaveBeenCalledWith(
      expect.stringContaining("multi-tenant isolation DISABLED"),
    );
  });
}); // end describe
