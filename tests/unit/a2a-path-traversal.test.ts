import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { existsSync, mkdirSync, readFileSync, writeFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { registerMemoryTools, validateSessionName, PathTraversalError } from "../../src/a2a-tools.js";

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

    it("rejects absolute-path traversal in file parameter", async () => {
      const absolutePath = process.platform === "win32"
        ? "C:\\Windows\\system32\\drivers\\etc\\hosts"
        : "/etc/passwd";
      const result = await ctx.tools.memory_read.handler(
        { file: absolutePath },
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
      const parsed = JSON.parse(result.content[0].text);
      expect(parsed.path).toBe("memory/safe.md");
      expect(parsed.totalLines).toBeGreaterThan(0);
      expect(parsed.text).toBe("safe content");
    });
  });

  describe("sessionName validation", () => {
    it("accepts valid session names", async () => {
      const tool = ctx.tools["memory_read"];
      for (const name of ["default", "my-session", "session_123", "A", "a".repeat(64)]) {
        // Create session dir for valid names so it doesn't fail on missing dir
        const dir = join(sessionsDir, name, "memory");
        mkdirSync(dir, { recursive: true });
        await expect(
          tool.handler({ file: "SELF.md" }, { sessionName: name })
        ).resolves.toBeDefined();
      }
    });

    it("rejects path traversal in sessionName", async () => {
      const tool = ctx.tools["memory_read"];
      for (const name of ["../etc", "foo/../bar", "..\\windows"]) {
        const result = await tool.handler({ file: "SELF.md" }, { sessionName: name });
        expect(result.isError).toBe(true);
        expect(result.content[0].text).toContain("Invalid session name");
      }
    });

    it("rejects null bytes in sessionName", async () => {
      const tool = ctx.tools["memory_read"];
      const result = await tool.handler({ file: "SELF.md" }, { sessionName: "session\x00name" });
      expect(result.isError).toBe(true);
      expect(result.content[0].text).toContain("Invalid session name");
    });

    it("rejects Windows reserved names", async () => {
      const tool = ctx.tools["memory_read"];
      for (const name of ["con", "CON", "aux", "AUX", "nul", "NUL", "prn", "PRN", "com1", "COM9", "lpt1", "LPT3"]) {
        const result = await tool.handler({ file: "SELF.md" }, { sessionName: name });
        expect(result.isError).toBe(true);
        expect(result.content[0].text).toContain("Invalid session name");
      }
    });

    it("rejects names longer than 64 characters", async () => {
      const tool = ctx.tools["memory_read"];
      const result = await tool.handler({ file: "SELF.md" }, { sessionName: "a".repeat(65) });
      expect(result.isError).toBe(true);
      expect(result.content[0].text).toContain("Invalid session name");
    });

    it("rejects empty session name via memory_read handler", async () => {
      // Note: handlers use `context.sessionName || "default"` so empty string
      // becomes "default" before reaching validateSessionName. Test via a name
      // that is explicitly invalid.
      const tool = ctx.tools["memory_read"];
      // Single char that's invalid — space — covers the empty/whitespace case
      const result = await tool.handler({ file: "SELF.md" }, { sessionName: " " });
      expect(result.isError).toBe(true);
      expect(result.content[0].text).toContain("Invalid session name");
    });

    it("validateSessionName throws PathTraversalError directly", () => {
      expect(() => validateSessionName(" ")).toThrow(PathTraversalError);
      expect(() => validateSessionName("../escape")).toThrow(PathTraversalError);
      expect(() => validateSessionName("con")).toThrow(PathTraversalError);
      expect(() => validateSessionName("a".repeat(65))).toThrow(PathTraversalError);
    });

    it("rejects names with special characters", async () => {
      const tool = ctx.tools["memory_read"];
      for (const name of ["foo bar", "foo/bar", "foo:bar", "foo*bar", "foo?bar", "foo<bar", "foo>bar", "foo|bar", "foo\"bar"]) {
        const result = await tool.handler({ file: "SELF.md" }, { sessionName: name });
        expect(result.isError).toBe(true);
        expect(result.content[0].text).toContain("Invalid session name");
      }
    });

    it("rejects invalid sessionName in self_reflect and returns isError", async () => {
      const tool = ctx.tools["self_reflect"];
      const result = await tool.handler({ reflection: "test" }, { sessionName: "../escape" });
      expect(result.isError).toBe(true);
      expect(result.content[0].text).toContain("Invalid session name");
    });

    it("rejects invalid sessionName in memory_write and returns isError", async () => {
      const tool = ctx.tools["memory_write"];
      const result = await tool.handler({ file: "notes.md", content: "x" }, { sessionName: "../escape" });
      expect(result.isError).toBe(true);
      expect(result.content[0].text).toContain("Invalid session name");
    });

    it("rejects invalid sessionName in memory_get and returns isError", async () => {
      const tool = ctx.tools["memory_get"];
      const result = await tool.handler({ path: "memory/safe.md" }, { sessionName: "../escape" });
      expect(result.isError).toBe(true);
      expect(result.content[0].text).toContain("Invalid session name");
    });
  });
});
