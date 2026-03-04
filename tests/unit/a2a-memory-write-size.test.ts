import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { registerMemoryTools, MEMORY_WRITE_MAX_BYTES } from "../../src/a2a-tools.js";

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

describe("memory_write content size limit", () => {
  let tmpBase: string;
  let ctx: ReturnType<typeof createMockCtx>;

  beforeEach(() => {
    tmpBase = join(tmpdir(), `wopr-test-write-size-${Date.now()}`);
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

  it("rejects content exceeding 1 MB", async () => {
    const oversized = "x".repeat(MEMORY_WRITE_MAX_BYTES + 1); // 1 byte over
    const result = await ctx.tools.memory_write.handler(
      { file: "test.md", content: oversized },
      { sessionName: "default" },
    );
    expect(result.isError).toBe(true);
    expect(result.content[0].text).toContain("exceeds maximum");
  });

  it("accepts content exactly at 1 MB", async () => {
    const ok = "x".repeat(MEMORY_WRITE_MAX_BYTES);
    const result = await ctx.tools.memory_write.handler(
      { file: "test.md", content: ok },
      { sessionName: "default" },
    );
    expect(result.isError).toBeUndefined();
    expect(result.content[0].text).toContain("Wrote");
  });

  it("rejects multi-byte content exceeding 1 MB by byte count", async () => {
    // Each emoji is 4 bytes. 262_145 * 4 = 1_048_580 > 1_048_576
    const oversized = "\u{1F914}".repeat(262_145);
    expect(Buffer.byteLength(oversized, "utf-8")).toBeGreaterThan(MEMORY_WRITE_MAX_BYTES);
    const result = await ctx.tools.memory_write.handler(
      { file: "test.md", content: oversized },
      { sessionName: "default" },
    );
    expect(result.isError).toBe(true);
    expect(result.content[0].text).toContain("exceeds maximum");
  });

  it("rejects append when combined size exceeds 1 MB", async () => {
    // Write a 900 KB file first
    const existingContent = "y".repeat(900_000);
    const memoryDir = join(tmpBase, "sessions", "default", "memory");
    writeFileSync(join(memoryDir, "big.md"), existingContent);

    // Append 200 KB — combined > 1 MB
    const appendContent = "z".repeat(200_000);
    const result = await ctx.tools.memory_write.handler(
      { file: "big.md", content: appendContent, append: true },
      { sessionName: "default" },
    );
    expect(result.isError).toBe(true);
    expect(result.content[0].text).toContain("exceeds maximum");
  });

  it("allows append when combined size is within 1 MB", async () => {
    const existingContent = "y".repeat(500_000);
    const memoryDir = join(tmpBase, "sessions", "default", "memory");
    writeFileSync(join(memoryDir, "ok.md"), existingContent);

    const appendContent = "z".repeat(500_000);
    const result = await ctx.tools.memory_write.handler(
      { file: "ok.md", content: appendContent, append: true },
      { sessionName: "default" },
    );
    expect(result.isError).toBeUndefined();
    expect(result.content[0].text).toContain("Appended to");
  });

  it("rejects overwrite content exceeding 1 MB even when file is small", async () => {
    const memoryDir = join(tmpBase, "sessions", "default", "memory");
    writeFileSync(join(memoryDir, "small.md"), "tiny");

    const oversized = "x".repeat(MEMORY_WRITE_MAX_BYTES + 1);
    const result = await ctx.tools.memory_write.handler(
      { file: "small.md", content: oversized },
      { sessionName: "default" },
    );
    expect(result.isError).toBe(true);
    expect(result.content[0].text).toContain("exceeds maximum");
  });

  it("uses config.maxWriteBytes when set, rejecting content exceeding it", async () => {
    const customMax = 512;
    const oversized = "x".repeat(513);
    const result = await ctx.tools.memory_write.handler(
      { file: "test.md", content: oversized },
      { sessionName: "default", config: { maxWriteBytes: customMax } },
    );
    expect(result.isError).toBe(true);
    expect(result.content[0].text).toContain("exceeds maximum");
    expect(result.content[0].text).toContain("512");
  });

  it("uses config.maxWriteBytes when set, accepting content within it", async () => {
    const customMax = 512;
    const ok = "x".repeat(512);
    const result = await ctx.tools.memory_write.handler(
      { file: "test.md", content: ok },
      { sessionName: "default", config: { maxWriteBytes: customMax } },
    );
    expect(result.isError).toBeUndefined();
    expect(result.content[0].text).toContain("Wrote");
  });

  it("config.maxWriteBytes overrides default for append path", async () => {
    const customMax = 1000;
    const memoryDir = join(tmpBase, "sessions", "default", "memory");
    writeFileSync(join(memoryDir, "capped.md"), "x".repeat(800));

    const appendContent = "z".repeat(300); // 800 + 2 + 300 = 1102 > 1000
    const result = await ctx.tools.memory_write.handler(
      { file: "capped.md", content: appendContent, append: true },
      { sessionName: "default", config: { maxWriteBytes: customMax } },
    );
    expect(result.isError).toBe(true);
    expect(result.content[0].text).toContain("exceeds maximum");
  });
});
