import { describe, expect, it, vi } from "vitest";
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
      apiKey: "test-key-fake",
    })),
    getExtension: vi.fn(() => null),
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

  it("should throw if search is called before init", async () => {
    // The plugin is shut down from the previous test; search should throw
    await expect(plugin.search("test query")).rejects.toThrow("not initialized");
  });

  it("should throw if capture is called before init", async () => {
    await expect(plugin.capture("some text")).rejects.toThrow("not initialized");
  });
});
