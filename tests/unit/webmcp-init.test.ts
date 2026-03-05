import { beforeEach, describe, expect, it, vi } from "vitest";

// Mock the webmcp module
vi.mock("../../src/webmcp.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../src/webmcp.js")>();
  return {
    ...actual,
    registerMemoryTools: vi.fn(),
    unregisterMemoryTools: vi.fn(),
  };
});

// Mock init.ts to avoid real initialization
vi.mock("../../src/init.js", () => ({
  initialize: vi.fn(async (_ctx: any, state: any) => {
    state.initialized = true;
    state.config = { instanceId: "test-instance" };
    state.instanceId = "test-instance";
  }),
}));

vi.mock("../../src/core-memory/watcher.js", () => ({
  stopWatcher: vi.fn(),
}));

import plugin from "../../src/index.js";
import {
  registerMemoryTools as mockRegisterWebMCP,
  unregisterMemoryTools as mockUnregisterWebMCP,
} from "../../src/webmcp.js";

function createMockContext(opts: { hasWebmcpRegistry?: boolean } = {}) {
  const registry = {
    register: vi.fn(),
    unregister: vi.fn(),
  };
  return {
    log: { info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn() },
    events: { on: vi.fn(() => vi.fn()) },
    registerConfigSchema: vi.fn(),
    unregisterConfigSchema: vi.fn(),
    registerContextProvider: vi.fn(),
    unregisterContextProvider: vi.fn(),
    registerExtension: vi.fn(),
    unregisterExtension: vi.fn(),
    getConfig: vi.fn(),
    ...(opts.hasWebmcpRegistry ? { webmcpRegistry: registry } : {}),
  };
}

describe("WebMCP tool registration in init()", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should call registerMemoryTools when webmcpRegistry is present", async () => {
    const ctx = createMockContext({ hasWebmcpRegistry: true });
    await plugin.init(ctx as any);
    expect(mockRegisterWebMCP).toHaveBeenCalledWith(
      ctx.webmcpRegistry,
      "/api",
      expect.any(String),
      expect.any(Function),
    );
  });

  it("should pass a searchFn that delegates to state.searchManager.search", async () => {
    const ctx = createMockContext({ hasWebmcpRegistry: true });
    await plugin.init(ctx as any);

    // Extract the searchFn passed as the 4th argument
    const searchFn = (mockRegisterWebMCP as ReturnType<typeof vi.fn>).mock.calls[0][3] as (
      query: string,
      limit: number,
      instanceId?: string,
    ) => Promise<unknown[]>;

    expect(typeof searchFn).toBe("function");

    // searchFn should throw because searchManager is not initialized in the mock
    expect(() => searchFn("test", 10, "inst")).toThrow("Semantic memory not initialized");
  });

  it("should skip registerMemoryTools when webmcpRegistry is absent", async () => {
    const ctx = createMockContext({ hasWebmcpRegistry: false });
    await plugin.init(ctx as any);
    expect(mockRegisterWebMCP).not.toHaveBeenCalled();
  });
});

describe("WebMCP tool unregistration in shutdown()", () => {
  it("should call unregisterMemoryTools on shutdown when registry was present", async () => {
    const ctx = createMockContext({ hasWebmcpRegistry: true });
    await plugin.init(ctx as any);
    vi.clearAllMocks();
    await plugin.shutdown();
    expect(mockUnregisterWebMCP).toHaveBeenCalledWith(ctx.webmcpRegistry);
  });

  it("should not call unregisterMemoryTools when registry was absent", async () => {
    const ctx = createMockContext({ hasWebmcpRegistry: false });
    await plugin.init(ctx as any);
    vi.clearAllMocks();
    await plugin.shutdown();
    expect(mockUnregisterWebMCP).not.toHaveBeenCalled();
  });

  it("should not throw when shutdown() is called without prior init()", async () => {
    await expect(plugin.shutdown()).resolves.not.toThrow();
  });
});
