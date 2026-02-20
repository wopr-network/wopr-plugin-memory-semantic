import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import plugin from "../../src/index.js";
import type { WOPRPluginContext } from "@wopr-network/plugin-types";

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

  // Additional tests would require mocking the embedding provider
  // which requires either: (a) a test OpenAI key, (b) a mock server,
  // or (c) refactoring createEmbeddingProvider to accept a factory
});
