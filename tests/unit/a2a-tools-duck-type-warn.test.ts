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
      expect.stringContaining("registerTool"),
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
});
