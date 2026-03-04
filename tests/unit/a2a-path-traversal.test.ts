import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { existsSync, mkdirSync, readFileSync, writeFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { registerMemoryTools } from "../../src/a2a-tools.js";

// Minimal mock context with registerTool
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

describe("A2A tools path traversal protection", () => {
  let tmpBase: string;
  let sessionsDir: string;
  let sessionDir: string;
  let memoryDir: string;
  let ctx: ReturnType<typeof createMockCtx>;

  beforeEach(() => {
    tmpBase = join(tmpdir(), `wopr-test-${Date.now()}`);
    sessionsDir = join(tmpBase, "sessions");
    sessionDir = join(sessionsDir, "default");
    memoryDir = join(sessionDir, "memory");
    mkdirSync(memoryDir, { recursive: true });
    writeFileSync(join(memoryDir, "safe.md"), "safe content");

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

  describe("memory_read", () => {
    it("rejects path traversal in file parameter", async () => {
      const result = await ctx.tools.memory_read.handler(
        { file: "../../etc/passwd" },
        { sessionName: "default" },
      );
      expect(result.isError).toBe(true);
      expect(result.content[0].text).toContain("Path outside allowed directory");
    });

    it("allows normal filenames", async () => {
      const result = await ctx.tools.memory_read.handler(
        { file: "safe.md" },
        { sessionName: "default" },
      );
      expect(result.isError).toBeUndefined();
      expect(result.content[0].text).toBe("safe content");
    });
  });

  describe("memory_write", () => {
    it("rejects path traversal in file parameter", async () => {
      const result = await ctx.tools.memory_write.handler(
        { file: "../../../tmp/pwned.txt", content: "hacked" },
        { sessionName: "default" },
      );
      expect(result.isError).toBe(true);
      expect(result.content[0].text).toContain("Path outside allowed directory");
    });

    it("allows normal filenames", async () => {
      const result = await ctx.tools.memory_write.handler(
        { file: "notes.md", content: "hello" },
        { sessionName: "default" },
      );
      expect(result.isError).toBeUndefined();
      expect(readFileSync(join(memoryDir, "notes.md"), "utf-8")).toBe("hello");
    });
  });

  describe("memory_get", () => {
    it("rejects path traversal in path parameter", async () => {
      const result = await ctx.tools.memory_get.handler(
        { path: "../../etc/passwd" },
        { sessionName: "default" },
      );
      expect(result.isError).toBe(true);
      expect(result.content[0].text).toContain("Path outside allowed directory");
    });

    it("allows normal paths", async () => {
      const result = await ctx.tools.memory_get.handler(
        { path: "memory/safe.md" },
        { sessionName: "default" },
      );
      expect(result.isError).toBeUndefined();
    });
  });
});
