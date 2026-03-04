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

  it("rejects section header exceeding 256 bytes", async () => {
    // "x".repeat(257) is 257 ASCII bytes — over the 256-byte section limit
    const oversizedSection = "x".repeat(257);
    const result = await ctx.tools.self_reflect.handler(
      { reflection: "ok", section: oversizedSection },
      { sessionName: "default" },
    );
    expect(result.isError).toBe(true);
    expect(result.content[0].text).toContain("section exceeds maximum allowed size");
  });

  it("rejects multi-byte reflection exceeding 64 KB by byte count", async () => {
    // Each '🤔' emoji is 4 bytes. 16_385 * 4 = 65_540 > 65_536
    const oversized = "🤔".repeat(16_385);
    expect(Buffer.byteLength(oversized, "utf-8")).toBeGreaterThan(65_536);
    const result = await ctx.tools.self_reflect.handler(
      { reflection: oversized },
      { sessionName: "default" },
    );
    expect(result.isError).toBe(true);
    expect(result.content[0].text).toContain("exceeds maximum allowed size");
  });

  it("accepts multi-byte reflection just within 64 KB by byte count", async () => {
    // Each '🤔' emoji is 4 bytes. 16_384 * 4 = 65_536 — exactly at limit
    const ok = "🤔".repeat(16_384);
    expect(Buffer.byteLength(ok, "utf-8")).toBe(65_536);
    const result = await ctx.tools.self_reflect.handler(
      { reflection: ok },
      { sessionName: "default" },
    );
    expect(result.isError).toBeUndefined();
    expect(result.content[0].text).toContain("Reflection added");
  });
});
