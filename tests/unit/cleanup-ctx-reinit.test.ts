/**
 * Regression test for WOP-1779: cleanup closures must unregister from the original
 * plugin context, not from the new one after re-initialization.
 *
 * Before the fix, all cleanup closures closed over the mutable module-level `ctx`
 * variable. When init() was called a second time without an intervening shutdown(),
 * `ctx` was overwritten with the new api, causing cleanup closures from the first
 * init to call unregister* on the new context instead of the original one.
 */

import { describe, expect, it, vi } from "vitest";
import type { WOPRPluginContext } from "@wopr-network/plugin-types";

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

describe("cleanupCtx re-init (WOP-1779)", () => {
  it("cleanup closures from first init unregister from the first context, not the second", async () => {
    const ctx1 = createMockContext();
    const ctx2 = createMockContext();

    // First init — registers against ctx1
    await plugin.init(ctx1);
    expect(ctx1.registerPermission).toHaveBeenCalledWith("memory.read");
    expect(ctx1.registerPermission).toHaveBeenCalledWith("memory.write");

    // Second init without shutdown — should run cleanups for ctx1 registrations
    // BEFORE overwriting ctx, then register fresh against ctx2
    await plugin.init(ctx2);

    // The cleanup from the first init must have called unregister on ctx1
    expect(ctx1.unregisterPermission).toHaveBeenCalledWith("memory.read");
    expect(ctx1.unregisterPermission).toHaveBeenCalledWith("memory.write");
    expect(ctx1.unregisterContextProvider).toHaveBeenCalledWith("memory-semantic");
    expect(ctx1.unregisterConfigSchema).toHaveBeenCalledWith("wopr-plugin-memory-semantic");

    // ctx2 must NOT have had cleanup called on it (it's the active context now)
    expect(ctx2.unregisterPermission).not.toHaveBeenCalled();
    expect(ctx2.unregisterContextProvider).not.toHaveBeenCalled();

    // ctx2 must have the new registrations
    expect(ctx2.registerPermission).toHaveBeenCalledWith("memory.read");
    expect(ctx2.registerPermission).toHaveBeenCalledWith("memory.write");

    await plugin.shutdown();
  });
});
