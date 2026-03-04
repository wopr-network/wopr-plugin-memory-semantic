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

describe("memory_search maxResults cap", () => {
  let tmpBase: string;
  let ctx: ReturnType<typeof createMockCtx>;
  let mgr: ReturnType<typeof createMockManager>;

  beforeEach(() => {
    tmpBase = join(tmpdir(), `wopr-test-maxresults-${Date.now()}`);
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

  it("clamps maxResults to 100 when caller requests 999999", async () => {
    await ctx.tools.memory_search.handler(
      { query: "test", maxResults: 999999 },
      {},
    );
    expect(mgr.search).toHaveBeenCalledWith(
      "test",
      expect.objectContaining({ maxResults: 100 }),
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

  it("clamps maxResults of exactly 101 to 100", async () => {
    await ctx.tools.memory_search.handler(
      { query: "test", maxResults: 101 },
      {},
    );
    expect(mgr.search).toHaveBeenCalledWith(
      "test",
      expect.objectContaining({ maxResults: 100 }),
    );
  });

  it("passes through maxResults of exactly 100", async () => {
    await ctx.tools.memory_search.handler(
      { query: "test", maxResults: 100 },
      {},
    );
    expect(mgr.search).toHaveBeenCalledWith(
      "test",
      expect.objectContaining({ maxResults: 100 }),
    );
  });
});
