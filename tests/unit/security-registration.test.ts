import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../src/embeddings.js", () => ({
  createEmbeddingProvider: vi.fn().mockResolvedValue({
    id: "mock-provider",
    dimensions: 4,
    probe: vi.fn().mockResolvedValue(4),
    embed: vi.fn().mockResolvedValue([[0.1, 0.2, 0.3, 0.4]]),
  }),
}));

vi.mock("../../src/search.js", () => ({
  createSemanticSearchManager: vi.fn().mockResolvedValue({
    addEntriesBatch: vi.fn().mockResolvedValue(undefined),
    search: vi.fn().mockResolvedValue([]),
    hasEntry: vi.fn().mockReturnValue(false),
    getEntry: vi.fn().mockReturnValue(undefined),
    getEntryCount: vi.fn().mockReturnValue(0),
    close: vi.fn().mockResolvedValue(undefined),
  }),
}));

import plugin from "../../src/index.js";

function createMockContext() {
  const handlers = new Map<string, Function[]>();
  return {
    events: {
      on(event: string, handler: Function) {
        if (!handlers.has(event)) handlers.set(event, []);
        handlers.get(event)!.push(handler);
        return () => {
          const list = handlers.get(event);
          if (list) {
            const idx = list.indexOf(handler);
            if (idx >= 0) list.splice(idx, 1);
          }
        };
      },
      emit: vi.fn(),
      off: vi.fn(),
    },
    log: { info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn() },
    getConfig: vi.fn(() => ({ provider: "openai" })),
    getExtension: vi.fn(() => null),
    registerConfigSchema: vi.fn(),
    unregisterConfigSchema: vi.fn(),
    registerContextProvider: vi.fn(),
    unregisterContextProvider: vi.fn(),
    registerExtension: vi.fn(),
    unregisterExtension: vi.fn(),
    registerPermission: vi.fn(),
    unregisterPermission: vi.fn(),
    registerToolPermission: vi.fn(),
    unregisterToolPermission: vi.fn(),
    storage: {
      register: vi.fn().mockResolvedValue(undefined),
      raw: vi.fn().mockResolvedValue([]),
      transaction: vi.fn().mockImplementation((fn: () => Promise<void>) => fn()),
      get: vi.fn().mockResolvedValue(undefined),
      set: vi.fn().mockResolvedValue(undefined),
    },
    _handlers: handlers,
  } as any;
}

describe("security registration", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should register permissions on init", async () => {
    const ctx = createMockContext();
    await plugin.init(ctx);

    expect(ctx.registerPermission).toHaveBeenCalledWith("memory.read");
    expect(ctx.registerPermission).toHaveBeenCalledWith("memory.write");

    await plugin.shutdown();
  });

  it("should register tool-permission mappings on init", async () => {
    const ctx = createMockContext();
    await plugin.init(ctx);

    expect(ctx.registerToolPermission).toHaveBeenCalledWith("memory_read", "memory.read");
    expect(ctx.registerToolPermission).toHaveBeenCalledWith("memory_write", "memory.write");
    expect(ctx.registerToolPermission).toHaveBeenCalledWith("memory_search", "memory.read");
    expect(ctx.registerToolPermission).toHaveBeenCalledWith("memory_get", "memory.read");
    expect(ctx.registerToolPermission).toHaveBeenCalledWith("self_reflect", "memory.write");
    expect(ctx.registerToolPermission).toHaveBeenCalledWith("identity_get", "memory.read");
    expect(ctx.registerToolPermission).toHaveBeenCalledWith("identity_update", "memory.write");

    await plugin.shutdown();
  });

  it("should unregister permissions on shutdown", async () => {
    const ctx = createMockContext();
    await plugin.init(ctx);
    await plugin.shutdown();

    expect(ctx.unregisterPermission).toHaveBeenCalledWith("memory.read");
    expect(ctx.unregisterPermission).toHaveBeenCalledWith("memory.write");
  });

  it("should unregister tool-permission mappings on shutdown", async () => {
    const ctx = createMockContext();
    await plugin.init(ctx);
    await plugin.shutdown();

    expect(ctx.unregisterToolPermission).toHaveBeenCalledWith("memory_read");
    expect(ctx.unregisterToolPermission).toHaveBeenCalledWith("memory_write");
    expect(ctx.unregisterToolPermission).toHaveBeenCalledWith("memory_search");
    expect(ctx.unregisterToolPermission).toHaveBeenCalledWith("memory_get");
    expect(ctx.unregisterToolPermission).toHaveBeenCalledWith("self_reflect");
    expect(ctx.unregisterToolPermission).toHaveBeenCalledWith("identity_get");
    expect(ctx.unregisterToolPermission).toHaveBeenCalledWith("identity_update");
  });
});
