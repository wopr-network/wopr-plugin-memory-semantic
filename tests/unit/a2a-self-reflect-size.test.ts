import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { mkdirSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { registerMemoryTools } from "../../src/a2a-tools.js";

function createMockCtx() {
  const tools: Record<string, any> = {};
  return {
    tools,
    log: { info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn() },
    registerTool: (tool: any) => {
      tools[tool.name] = tool;
    },
  };
}

function createMockManager() {
  return { search: vi.fn().mockResolvedValue([]) } as any;
}

describe("self_reflect content size limit", () => {
  let tmpBase: string;
  let ctx: ReturnType<typeof createMockCtx>;

  beforeEach(() => {
    tmpBase = join(tmpdir(), `wopr-test-reflect-${Date.now()}`);
    const memoryDir = join(tmpBase, "sessions", "default", "memory");
    mkdirSync(memoryDir, { recursive: true });
    process.env.WOPR_HOME = tmpBase;
    process.env.WOPR_GLOBAL_IDENTITY = join(tmpBase, "identity");
    mkdirSync(join(tmpBase, "identity", "memory"), { recursive: true });

    ctx = createMockCtx();
    registerMemoryTools(ctx as any, createMockManager());
  });

  afterEach(() => {
    rmSync(tmpBase, { recursive: true, force: true });
    delete process.env.WOPR_HOME;
    delete process.env.WOPR_GLOBAL_IDENTITY;
  });

  it("rejects reflection exceeding 64 KB", async () => {
    const oversized = "x".repeat(65_537);
    const result = await ctx.tools.self_reflect.handler(
      { reflection: oversized },
      { sessionName: "default" },
    );
    expect(result.isError).toBe(true);
    expect(result.content[0].text).toContain("exceeds maximum allowed size");
  });

  it("rejects tattoo exceeding 64 KB", async () => {
    const oversized = "x".repeat(65_537);
    const result = await ctx.tools.self_reflect.handler(
      { tattoo: oversized },
      { sessionName: "default" },
    );
    expect(result.isError).toBe(true);
    expect(result.content[0].text).toContain("exceeds maximum allowed size");
  });

  it("accepts reflection within 64 KB", async () => {
    const ok = "x".repeat(65_536);
    const result = await ctx.tools.self_reflect.handler(
      { reflection: ok },
      { sessionName: "default" },
    );
    expect(result.isError).toBeUndefined();
    expect(result.content[0].text).toContain("Reflection added");
  });

  it("accepts tattoo within 64 KB", async () => {
    const ok = "x".repeat(65_536);
    const result = await ctx.tools.self_reflect.handler(
      { tattoo: ok },
      { sessionName: "default" },
    );
    expect(result.isError).toBeUndefined();
    expect(result.content[0].text).toContain("Tattoo added");
  });
});
