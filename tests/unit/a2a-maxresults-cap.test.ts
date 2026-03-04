import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { registerMemoryTools, MAX_SEARCH_RESULTS } from "../../src/a2a-tools.js";

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

describe("memory_search maxResults cap", () => {
  let tmpBase: string;
  let ctx: ReturnType<typeof createMockCtx>;
  let mgr: ReturnType<typeof createMockManager>;

  beforeEach(() => {
    tmpBase = mkdtempSync(join(tmpdir(), "wopr-test-maxresults-"));
    const memoryDir = join(tmpBase, "sessions", "default", "memory");
    mkdirSync(memoryDir, { recursive: true });
    mkdirSync(join(tmpBase, "identity", "memory"), { recursive: true });
    process.env.WOPR_HOME = tmpBase;
    process.env.WOPR_GLOBAL_IDENTITY = join(tmpBase, "identity");

    ctx = createMockCtx();
    mgr = createMockManager();
    registerMemoryTools(ctx as any, mgr, "test-instance");
  });

  afterEach(() => {
    rmSync(tmpBase, { recursive: true, force: true });
    delete process.env.WOPR_HOME;
    delete process.env.WOPR_GLOBAL_IDENTITY;
  });

  it("clamps maxResults to MAX_SEARCH_RESULTS when caller requests 999999", async () => {
    await ctx.tools.memory_search.handler(
      { query: "test", maxResults: 999999 },
      {},
    );
    expect(mgr.search).toHaveBeenCalledWith(
      "test",
      expect.objectContaining({ maxResults: MAX_SEARCH_RESULTS }),
    );
  });

  it("passes through maxResults when within bounds", async () => {
    await ctx.tools.memory_search.handler(
      { query: "test", maxResults: 5 },
      {},
    );
    expect(mgr.search).toHaveBeenCalledWith(
      "test",
      expect.objectContaining({ maxResults: 5 }),
    );
  });

  it("uses default of 10 when maxResults is omitted", async () => {
    await ctx.tools.memory_search.handler(
      { query: "test" },
      {},
    );
    expect(mgr.search).toHaveBeenCalledWith(
      "test",
      expect.objectContaining({ maxResults: 10 }),
    );
  });

  it(`clamps maxResults of exactly ${MAX_SEARCH_RESULTS + 1} to ${MAX_SEARCH_RESULTS}`, async () => {
    await ctx.tools.memory_search.handler(
      { query: "test", maxResults: MAX_SEARCH_RESULTS + 1 },
      {},
    );
    expect(mgr.search).toHaveBeenCalledWith(
      "test",
      expect.objectContaining({ maxResults: MAX_SEARCH_RESULTS }),
    );
  });

  it(`passes through maxResults of exactly ${MAX_SEARCH_RESULTS}`, async () => {
    await ctx.tools.memory_search.handler(
      { query: "test", maxResults: MAX_SEARCH_RESULTS },
      {},
    );
    expect(mgr.search).toHaveBeenCalledWith(
      "test",
      expect.objectContaining({ maxResults: MAX_SEARCH_RESULTS }),
    );
  });

  it("falls back to default 10 when maxResults is negative", async () => {
    await ctx.tools.memory_search.handler(
      { query: "test", maxResults: -1 },
      {},
    );
    expect(mgr.search).toHaveBeenCalledWith(
      "test",
      expect.objectContaining({ maxResults: 10 }),
    );
  });

  it("falls back to default 10 when maxResults is zero", async () => {
    await ctx.tools.memory_search.handler(
      { query: "test", maxResults: 0 },
      {},
    );
    expect(mgr.search).toHaveBeenCalledWith(
      "test",
      expect.objectContaining({ maxResults: 10 }),
    );
  });

  it("falls back to default 10 when maxResults is NaN", async () => {
    await ctx.tools.memory_search.handler(
      { query: "test", maxResults: Number.NaN },
      {},
    );
    expect(mgr.search).toHaveBeenCalledWith(
      "test",
      expect.objectContaining({ maxResults: 10 }),
    );
  });

  it("falls back to default 10 when maxResults is null (runtime coercion)", async () => {
    await ctx.tools.memory_search.handler(
      { query: "test", maxResults: null as any },
      {},
    );
    expect(mgr.search).toHaveBeenCalledWith(
      "test",
      expect.objectContaining({ maxResults: 10 }),
    );
  });

  it("truncates fractional maxResults to integer", async () => {
    await ctx.tools.memory_search.handler(
      { query: "test", maxResults: 7.9 },
      {},
    );
    expect(mgr.search).toHaveBeenCalledWith(
      "test",
      expect.objectContaining({ maxResults: 7 }),
    );
  });
});
