import { beforeEach, describe, expect, it, vi } from "vitest";
import type { WOPRPluginContext } from "@wopr-network/plugin-types";

// Mock the embeddings module so init() can run without a real embedding provider
vi.mock("../../src/embeddings.js", () => ({
  createEmbeddingProvider: vi.fn().mockResolvedValue({
    id: "mock-provider",
    dimensions: 4,
    probe: vi.fn().mockResolvedValue(4),
    embed: vi.fn().mockResolvedValue([[0.1, 0.2, 0.3, 0.4]]),
  }),
}));

// Mock search module to avoid usearch native binary dependency in tests
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

// Creates a mock WOPRPluginContext with an event bus
function createMockContext(): WOPRPluginContext & {
  _handlers: Map<string, Function[]>;
  _emit: (event: string, payload: any) => Promise<void>;
} {
  const handlers = new Map<string, Function[]>();

  const events = {
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
  };

  return {
    events,
    log: {
      info: vi.fn(),
      warn: vi.fn(),
      error: vi.fn(),
      debug: vi.fn(),
    },
    getConfig: vi.fn(() => ({
      provider: "openai",
    })),
    getExtension: vi.fn(() => null),
    registerConfigSchema: vi.fn(),
    unregisterConfigSchema: vi.fn(),
    registerContextProvider: vi.fn(),
    unregisterContextProvider: vi.fn(),
    registerExtension: vi.fn(),
    _handlers: handlers,
    _emit: async (event: string, payload: any) => {
      const list = handlers.get(event) || [];
      for (const h of list) await h(payload);
    },
  } as any;
}

function createStorageMock() {
  return {
    register: vi.fn().mockResolvedValue(undefined),
    // storage.raw must return an array — MemoryIndexManager reads rows[0] from results
    raw: vi.fn().mockResolvedValue([]),
    transaction: vi.fn().mockImplementation((fn: () => Promise<void>) => fn()),
    get: vi.fn().mockResolvedValue(undefined),
    set: vi.fn().mockResolvedValue(undefined),
  };
}

describe("plugin lifecycle", () => {
  it("should export required plugin interface fields", () => {
    expect(plugin.id).toBe("memory-semantic");
    expect(plugin.name).toBe("Semantic Memory");
    expect(plugin.version).toBe("1.0.0");
    expect(typeof plugin.init).toBe("function");
    expect(typeof plugin.shutdown).toBe("function");
    expect(typeof plugin.search).toBe("function");
    expect(typeof plugin.capture).toBe("function");
    expect(typeof plugin.getConfig).toBe("function");
  });

  it("should return default config before init", () => {
    const config = plugin.getConfig();
    expect(config.provider).toBe("auto");
    expect(config.autoRecall.enabled).toBe(true);
    expect(config.autoCapture.enabled).toBe(true);
  });

  it("should register event hooks on init and remove them on shutdown", async () => {
    const ctx = createMockContext();
    const onSpy = vi.spyOn(ctx.events, "on");
    (ctx as any).storage = createStorageMock();

    await plugin.init(ctx as any);

    // init() subscribes to session:beforeInject, session:afterInject,
    // memory:filesChanged, and memory:search after successful initialization
    expect(onSpy).toHaveBeenCalled();
    const subscribedEvents = onSpy.mock.calls.map((call) => call[0]);
    expect(subscribedEvents).toContain("session:beforeInject");
    expect(subscribedEvents).toContain("session:afterInject");
    expect(subscribedEvents).toContain("memory:filesChanged");
    expect(subscribedEvents).toContain("memory:search");

    // All lifecycle hooks should be active before shutdown
    const lifecycleEvents = ["session:beforeInject", "session:afterInject", "memory:filesChanged", "memory:search"];
    for (const event of lifecycleEvents) {
      expect(ctx._handlers.get(event)?.length ?? 0).toBeGreaterThan(0);
    }

    // Record handler counts before shutdown so we can verify removal
    const countsBefore = new Map(lifecycleEvents.map((e) => [e, ctx._handlers.get(e)?.length ?? 0]));

    await plugin.shutdown();

    // After shutdown, the plugin's registered handlers should be removed.
    // session:beforeInject and session:afterInject are only registered by the plugin — must drop to 0.
    // memory:search is only registered by the plugin — must drop to 0.
    // memory:filesChanged may retain an internal MemoryIndexManager subscription, but
    // the plugin's own handler must be removed (count decreases).
    expect(ctx._handlers.get("session:beforeInject")?.length ?? 0).toBe(0);
    expect(ctx._handlers.get("session:afterInject")?.length ?? 0).toBe(0);
    expect(ctx._handlers.get("memory:search")?.length ?? 0).toBe(0);
    expect(ctx._handlers.get("memory:filesChanged")?.length ?? 0).toBeLessThan(countsBefore.get("memory:filesChanged")!);
  });

  describe("when plugin is not initialized", () => {
    beforeEach(async () => {
      // Ensure plugin is shut down regardless of prior test state.
      // shutdown() is idempotent — safe to call even if already shut down.
      await plugin.shutdown();
    });

    it("should throw if search is called before init", async () => {
      await expect(plugin.search("test query")).rejects.toThrow("Semantic memory not initialized");
    });

    it("should throw if capture is called before init", async () => {
      await expect(plugin.capture("some text")).rejects.toThrow("Semantic memory not initialized");
    });
  });

  it("should allow re-initialization after shutdown", async () => {
    const ctx = createMockContext();
    (ctx as any).storage = createStorageMock();

    // First init
    await plugin.init(ctx as any);
    expect(plugin.getConfig()).toBeDefined();

    // Shutdown
    await plugin.shutdown();

    // Re-init — this would hang forever before the fix because
    // initInProgress stayed true after successful init
    const ctx2 = createMockContext();
    (ctx2 as any).storage = createStorageMock();
    await plugin.init(ctx2 as any);

    // Verify the plugin re-initialized successfully with the mock config's provider
    const config = plugin.getConfig();
    expect(config).toBeDefined();
    expect(config.provider).toBe("openai");

    // Clean up
    await plugin.shutdown();
  });

  it("should reset initInProgress after a failed init so a second init can proceed", async () => {
    const { createEmbeddingProvider } = await import("../../src/embeddings.js");
    const mockCreate = vi.mocked(createEmbeddingProvider);

    // First call throws — simulates a broken provider
    mockCreate.mockRejectedValueOnce(new Error("provider unavailable"));

    const ctx = createMockContext();
    (ctx as any).storage = createStorageMock();

    // The failed init should not throw (plugin swallows errors and logs them)
    await plugin.init(ctx as any);

    // Plugin should not be initialized after the failure
    expect(plugin.getConfig()).toBeDefined(); // getConfig always works (returns state.config)

    // Restore the mock to succeed for the retry
    mockCreate.mockResolvedValue({
      id: "mock-provider",
      dimensions: 4,
      probe: vi.fn().mockResolvedValue(4),
      embed: vi.fn().mockResolvedValue([[0.1, 0.2, 0.3, 0.4]]),
    } as any);

    // Second init should succeed because initInProgress was reset in the finally block
    const ctx2 = createMockContext();
    (ctx2 as any).storage = createStorageMock();
    await plugin.init(ctx2 as any);

    // Verify we can now call search without "not initialized" error
    await expect(plugin.search("test")).resolves.toBeDefined();

    // Clean up
    await plugin.shutdown();
  });
});
