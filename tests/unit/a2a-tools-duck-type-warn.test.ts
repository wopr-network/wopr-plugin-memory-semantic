import { describe, it, expect, vi } from "vitest";
import { registerMemoryTools, unregisterMemoryTools } from "../../src/a2a-tools.js";

function createCtxWithoutRegisterTool() {
  return {
    log: { info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn() },
    // NOTE: no registerTool or unregisterTool
  };
}

function createMockManager() {
  return { search: vi.fn().mockResolvedValue([]) } as any;
}

describe("registerMemoryTools duck-type warning", () => {
  it("should warn when ctx lacks registerTool", () => {
    const ctx = createCtxWithoutRegisterTool() as any;
    registerMemoryTools(ctx, createMockManager());
    expect(ctx.log.warn).toHaveBeenCalledWith(
      expect.stringContaining("A2A memory tools will not be registered"),
    );
  });

  it("should warn when ctx.registerTool is a non-function value", () => {
    const ctx = { ...createCtxWithoutRegisterTool(), registerTool: undefined } as any;
    registerMemoryTools(ctx, createMockManager());
    expect(ctx.log.warn).toHaveBeenCalledWith(
      expect.stringContaining("A2A memory tools will not be registered"),
    );
  });
});

describe("unregisterMemoryTools duck-type warning", () => {
  it("should warn when ctx lacks unregisterTool", () => {
    const ctx = createCtxWithoutRegisterTool() as any;
    unregisterMemoryTools(ctx);
    expect(ctx.log.warn).toHaveBeenCalledWith(
      expect.stringContaining("unregisterTool"),
    );
  });

  it("should warn when ctx.unregisterTool is a non-function value", () => {
    const ctx = { ...createCtxWithoutRegisterTool(), unregisterTool: null } as any;
    unregisterMemoryTools(ctx);
    expect(ctx.log.warn).toHaveBeenCalledWith(
      expect.stringContaining("unregisterTool"),
    );
  });
});
