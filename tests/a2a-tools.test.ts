/**
 * Comprehensive tests for src/a2a-tools.ts
 *
 * Covers: validateSessionName, PathTraversalError, registerMemoryTools,
 * unregisterMemoryTools, and all 5 tool handlers (memory_read, memory_write,
 * memory_search, memory_get, self_reflect).
 *
 * WOP-1593
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import {
  MAX_SEARCH_RESULTS,
  MEMORY_WRITE_MAX_BYTES,
  PathTraversalError,
  registerMemoryTools,
  unregisterMemoryTools,
  validateSessionName,
} from "../src/a2a-tools.js";

// ---------------------------------------------------------------------------
// validateSessionName
// ---------------------------------------------------------------------------

describe("validateSessionName", () => {
  it("accepts valid alphanumeric names", () => {
    expect(() => validateSessionName("my-session_01")).not.toThrow();
    expect(() => validateSessionName("a")).not.toThrow();
    expect(() => validateSessionName("A".repeat(64))).not.toThrow();
  });

  it("rejects empty string", () => {
    expect(() => validateSessionName("")).toThrow(PathTraversalError);
  });

  it("rejects names longer than 64 characters", () => {
    expect(() => validateSessionName("a".repeat(65))).toThrow(PathTraversalError);
  });

  it("rejects path separators", () => {
    expect(() => validateSessionName("foo/bar")).toThrow(PathTraversalError);
    expect(() => validateSessionName("foo\\bar")).toThrow(PathTraversalError);
  });

  it("rejects dots and special characters", () => {
    expect(() => validateSessionName("..")).toThrow(PathTraversalError);
    expect(() => validateSessionName("foo.bar")).toThrow(PathTraversalError);
    expect(() => validateSessionName("foo bar")).toThrow(PathTraversalError);
  });

  it("rejects Windows reserved names", () => {
    expect(() => validateSessionName("con")).toThrow(PathTraversalError);
    expect(() => validateSessionName("PRN")).toThrow(PathTraversalError);
    expect(() => validateSessionName("aux")).toThrow(PathTraversalError);
    expect(() => validateSessionName("nul")).toThrow(PathTraversalError);
    expect(() => validateSessionName("com1")).toThrow(PathTraversalError);
    expect(() => validateSessionName("lpt9")).toThrow(PathTraversalError);
  });

  it("rejects null bytes", () => {
    expect(() => validateSessionName("foo\0bar")).toThrow(PathTraversalError);
  });
});

// ---------------------------------------------------------------------------
// PathTraversalError
// ---------------------------------------------------------------------------

describe("PathTraversalError", () => {
  it("has correct name and default message", () => {
    const err = new PathTraversalError();
    expect(err.name).toBe("PathTraversalError");
    expect(err.message).toBe("Path outside allowed directory");
  });

  it("accepts custom message", () => {
    const err = new PathTraversalError("custom");
    expect(err.message).toBe("custom");
  });

  it("is an instance of Error", () => {
    expect(new PathTraversalError()).toBeInstanceOf(Error);
  });
});

// ---------------------------------------------------------------------------
// Shared test setup
// ---------------------------------------------------------------------------

describe("a2a tool handlers", () => {
  let tmpDir: string;
  let sessionsDir: string;
  let globalIdentityDir: string;
  let globalMemoryDir: string;
  let registeredTools: Record<string, { inputSchema: unknown; handler: Function }>;
  let mockCtx: any;
  let mockManager: any;
  let origWoprHome: string | undefined;
  let origGlobalIdentity: string | undefined;

  beforeEach(() => {
    tmpDir = mkdtempSync(join(tmpdir(), "a2a-tools-test-"));
    sessionsDir = join(tmpDir, "sessions");
    globalIdentityDir = join(tmpDir, "identity");
    globalMemoryDir = join(globalIdentityDir, "memory");
    mkdirSync(sessionsDir, { recursive: true });
    mkdirSync(globalMemoryDir, { recursive: true });

    origWoprHome = process.env.WOPR_HOME;
    origGlobalIdentity = process.env.WOPR_GLOBAL_IDENTITY;
    process.env.WOPR_HOME = tmpDir;
    process.env.WOPR_GLOBAL_IDENTITY = globalIdentityDir;

    registeredTools = {};
    mockCtx = {
      log: { info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn() },
      registerTool: (tool: any) => {
        registeredTools[tool.name] = tool;
      },
      unregisterTool: vi.fn((name: string) => {
        delete registeredTools[name];
      }),
    };
    mockManager = { search: vi.fn().mockResolvedValue([]) };
  });

  afterEach(() => {
    rmSync(tmpDir, { recursive: true, force: true });
    if (origWoprHome === undefined) {
      delete process.env.WOPR_HOME;
    } else {
      process.env.WOPR_HOME = origWoprHome;
    }
    if (origGlobalIdentity === undefined) {
      delete process.env.WOPR_GLOBAL_IDENTITY;
    } else {
      process.env.WOPR_GLOBAL_IDENTITY = origGlobalIdentity;
    }
  });

  const getHandler = (name: string) => registeredTools[name].handler;
  const ctx = (session = "test-session") => ({ sessionName: session });

  const makeSessionDir = (session = "test-session") => {
    const dir = join(sessionsDir, session, "memory");
    mkdirSync(dir, { recursive: true });
    return dir;
  };

  // -------------------------------------------------------------------------
  // registerMemoryTools / unregisterMemoryTools
  // -------------------------------------------------------------------------

  describe("registerMemoryTools / unregisterMemoryTools", () => {
    it("registers all 5 tools when registerTool is available", () => {
      registerMemoryTools(mockCtx, mockManager);
      expect(Object.keys(registeredTools).sort()).toEqual([
        "memory_get",
        "memory_read",
        "memory_search",
        "memory_write",
        "self_reflect",
      ]);
    });

    it("warns and skips when registerTool is not available", () => {
      const noToolCtx = {
        log: { info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn() },
      };
      registerMemoryTools(noToolCtx as any, mockManager);
      expect(noToolCtx.log.warn).toHaveBeenCalledWith(
        expect.stringContaining("A2A memory tools will not be registered"),
      );
      expect(Object.keys(registeredTools)).toHaveLength(0);
    });

    it("warns when ctx.registerTool is a non-function value", () => {
      const badCtx = {
        log: { info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn() },
        registerTool: undefined,
      };
      registerMemoryTools(badCtx as any, mockManager);
      expect(badCtx.log.warn).toHaveBeenCalledWith(
        expect.stringContaining("A2A memory tools will not be registered"),
      );
    });

    it("unregisters all tools", () => {
      registerMemoryTools(mockCtx, mockManager);
      unregisterMemoryTools(mockCtx);
      expect(mockCtx.unregisterTool).toHaveBeenCalledTimes(5);
      expect(Object.keys(registeredTools)).toHaveLength(0);
    });

    it("warns when unregisterTool is not available", () => {
      const noToolCtx = {
        log: { info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn() },
      };
      unregisterMemoryTools(noToolCtx as any);
      expect(noToolCtx.log.warn).toHaveBeenCalledWith(
        expect.stringContaining("unregisterTool"),
      );
    });
  });

  // -------------------------------------------------------------------------
  // Path traversal regression (WOP-1540)
  // -------------------------------------------------------------------------

  describe("path traversal rejection (WOP-1540 regression)", () => {
    beforeEach(() => {
      registerMemoryTools(mockCtx, mockManager);
      makeSessionDir();
    });

    it("memory_read rejects ../../etc/passwd", async () => {
      const result = await getHandler("memory_read")(
        { file: "../../etc/passwd" },
        ctx(),
      );
      expect(result.isError).toBe(true);
      expect(result.content[0].text).toMatch(
        /Path outside allowed directory|Invalid session name/,
      );
    });

    it("memory_read rejects absolute path", async () => {
      const outside = join(tmpDir, "outside.md");
      writeFileSync(outside, "nope");
      const result = await getHandler("memory_read")({ file: outside }, ctx());
      expect(result.isError).toBe(true);
    });

    it("memory_write rejects ../../etc/passwd", async () => {
      const result = await getHandler("memory_write")(
        { file: "../../etc/passwd", content: "pwned" },
        ctx(),
      );
      expect(result.isError).toBe(true);
      expect(result.content[0].text).toMatch(
        /Path outside allowed directory|Invalid session name/,
      );
    });

    it("memory_write rejects path traversal with ../", async () => {
      const result = await getHandler("memory_write")(
        { file: "../../../etc/crontab", content: "pwned" },
        ctx(),
      );
      expect(result.isError).toBe(true);
    });

    it("memory_get rejects ../../etc/passwd", async () => {
      const result = await getHandler("memory_get")(
        { path: "../../etc/passwd" },
        ctx(),
      );
      expect(result.isError).toBe(true);
    });

    it("memory_get rejects absolute path", async () => {
      const outside = join(tmpDir, "outside.md");
      writeFileSync(outside, "nope");
      const result = await getHandler("memory_get")({ path: outside }, ctx());
      expect(result.isError).toBe(true);
    });

    it("rejects session name with path traversal", async () => {
      const result = await getHandler("memory_read")(
        { file: "MEMORY.md" },
        { sessionName: "../evil" },
      );
      expect(result.isError).toBe(true);
    });

    it("memory_read rejects symlink-based escape", async () => {
      const sessionMemory = makeSessionDir();
      const linkPath = join(sessionMemory, "evil-link.md");
      try {
        symlinkSync("/etc/passwd", linkPath);
      } catch {
        return; // symlink may not be supported in this env
      }
      const result = await getHandler("memory_read")({ file: "evil-link.md" }, ctx());
      expect(result.isError).toBe(true);
    });

    it("memory_write rejects symlink target", async () => {
      const sessionMemory = makeSessionDir();
      const linkPath = join(sessionMemory, "evil-link.md");
      try {
        symlinkSync("/tmp/evil-target.md", linkPath);
      } catch {
        return;
      }
      const result = await getHandler("memory_write")(
        { file: "evil-link.md", content: "pwned" },
        ctx(),
      );
      expect(result.isError).toBe(true);
    });
  });

  // -------------------------------------------------------------------------
  // memory_read happy path
  // -------------------------------------------------------------------------

  describe("memory_read", () => {
    beforeEach(() => {
      registerMemoryTools(mockCtx, mockManager);
      makeSessionDir();
    });

    it("lists available files when no file param given", async () => {
      const sessionMemory = join(sessionsDir, "test-session", "memory");
      writeFileSync(join(sessionMemory, "notes.md"), "# Notes");
      writeFileSync(join(globalMemoryDir, "global.md"), "# Global");

      const result = await getHandler("memory_read")({}, ctx());
      expect(result.content[0].text).toContain("notes.md");
      expect(result.content[0].text).toContain("global.md");
    });

    it("returns 'No memory files found' when no files exist", async () => {
      const result = await getHandler("memory_read")({}, ctx());
      expect(result.content[0].text).toContain("No memory files found");
    });

    it("reads a session memory file", async () => {
      const sessionMemory = join(sessionsDir, "test-session", "memory");
      writeFileSync(join(sessionMemory, "test.md"), "hello world");

      const result = await getHandler("memory_read")({ file: "test.md" }, ctx());
      expect(result.content[0].text).toBe("hello world");
    });

    it("reads a global memory file", async () => {
      writeFileSync(join(globalMemoryDir, "shared.md"), "global content");

      const result = await getHandler("memory_read")({ file: "shared.md" }, ctx());
      expect(result.content[0].text).toBe("global content");
    });

    it("reads a ROOT_FILE (MEMORY.md) from session", async () => {
      const sessionDir = join(sessionsDir, "test-session");
      writeFileSync(join(sessionDir, "MEMORY.md"), "# Session Memory");

      const result = await getHandler("memory_read")({ file: "MEMORY.md" }, ctx());
      expect(result.content[0].text).toBe("# Session Memory");
    });

    it("reads a ROOT_FILE (SOUL.md) from global identity", async () => {
      writeFileSync(join(globalIdentityDir, "SOUL.md"), "# Soul");

      const result = await getHandler("memory_read")({ file: "SOUL.md" }, ctx());
      expect(result.content[0].text).toBe("# Soul");
    });

    it("returns error for non-existent file", async () => {
      const result = await getHandler("memory_read")({ file: "missing.md" }, ctx());
      expect(result.isError).toBe(true);
      expect(result.content[0].text).toContain("File not found");
    });

    it("supports line range with from and lines params", async () => {
      const sessionMemory = join(sessionsDir, "test-session", "memory");
      writeFileSync(join(sessionMemory, "multi.md"), "line1\nline2\nline3\nline4\nline5");

      const result = await getHandler("memory_read")(
        { file: "multi.md", from: 2, lines: 2 },
        ctx(),
      );
      const parsed = JSON.parse(result.content[0].text);
      expect(parsed.from).toBe(2);
      expect(parsed.to).toBe(3);
      expect(parsed.text).toBe("line2\nline3");
    });

    it("reads daily logs with file='recent'", async () => {
      vi.useFakeTimers();
      vi.setSystemTime(new Date("2026-03-04T12:00:00Z"));
      try {
        const sessionMemory = join(sessionsDir, "test-session", "memory");
        const today = "2026-03-04";
        writeFileSync(join(sessionMemory, `${today}.md`), "today's log");

        const result = await getHandler("memory_read")({ file: "recent", days: 1 }, ctx());
        expect(result.content[0].text).toContain("today's log");
      } finally {
        vi.useRealTimers();
      }
    });

    it("returns 'No daily memory files yet' when none exist", async () => {
      const result = await getHandler("memory_read")({ file: "daily" }, ctx());
      expect(result.content[0].text).toContain("No daily memory files yet");
    });
  });

  // -------------------------------------------------------------------------
  // memory_write happy path
  // -------------------------------------------------------------------------

  describe("memory_write", () => {
    beforeEach(() => {
      registerMemoryTools(mockCtx, mockManager);
      makeSessionDir();
    });

    it("writes a new file", async () => {
      const result = await getHandler("memory_write")(
        { file: "notes.md", content: "hello" },
        ctx(),
      );
      expect(result.content[0].text).toContain("notes.md");
      const written = readFileSync(
        join(sessionsDir, "test-session", "memory", "notes.md"),
        "utf-8",
      );
      expect(written).toBe("hello");
    });

    it("overwrites existing file when append is false", async () => {
      const filePath = join(sessionsDir, "test-session", "memory", "notes.md");
      writeFileSync(filePath, "old");

      await getHandler("memory_write")(
        { file: "notes.md", content: "new", append: false },
        ctx(),
      );
      expect(readFileSync(filePath, "utf-8")).toBe("new");
    });

    it("appends when append is true", async () => {
      const filePath = join(sessionsDir, "test-session", "memory", "notes.md");
      writeFileSync(filePath, "old");

      await getHandler("memory_write")(
        { file: "notes.md", content: "new", append: true },
        ctx(),
      );
      const content = readFileSync(filePath, "utf-8");
      expect(content).toContain("old");
      expect(content).toContain("new");
    });

    it("auto-appends for date-named files (today)", async () => {
      vi.useFakeTimers();
      vi.setSystemTime(new Date("2026-03-04T12:00:00Z"));
      try {
        const today = "2026-03-04";
        const filePath = join(sessionsDir, "test-session", "memory", `${today}.md`);
        writeFileSync(filePath, "morning");

        await getHandler("memory_write")({ file: "today", content: "evening" }, ctx());
        const content = readFileSync(filePath, "utf-8");
        expect(content).toContain("morning");
        expect(content).toContain("evening");
      } finally {
        vi.useRealTimers();
      }
    });

    it("writes ROOT_FILES to session root", async () => {
      await getHandler("memory_write")(
        { file: "MEMORY.md", content: "# Memory" },
        ctx(),
      );
      const content = readFileSync(
        join(sessionsDir, "test-session", "MEMORY.md"),
        "utf-8",
      );
      expect(content).toBe("# Memory");
    });

    it("rejects content exceeding MEMORY_WRITE_MAX_BYTES", async () => {
      const bigContent = "x".repeat(MEMORY_WRITE_MAX_BYTES + 1);
      const result = await getHandler("memory_write")(
        { file: "big.md", content: bigContent },
        ctx(),
      );
      expect(result.isError).toBe(true);
      expect(result.content[0].text).toContain("exceeds maximum");
    });

    it("rejects append that would exceed max bytes", async () => {
      const filePath = join(sessionsDir, "test-session", "memory", "big.md");
      writeFileSync(filePath, "x".repeat(MEMORY_WRITE_MAX_BYTES - 10));

      const result = await getHandler("memory_write")(
        { file: "big.md", content: "y".repeat(100), append: true },
        ctx(),
      );
      expect(result.isError).toBe(true);
      expect(result.content[0].text).toContain("exceeds maximum");
    });

    it("creates memory directory if it does not exist", async () => {
      rmSync(join(sessionsDir, "test-session", "memory"), { recursive: true, force: true });

      await getHandler("memory_write")({ file: "new.md", content: "created" }, ctx());
      expect(existsSync(join(sessionsDir, "test-session", "memory", "new.md"))).toBe(true);
    });
  });

  // -------------------------------------------------------------------------
  // memory_search
  // -------------------------------------------------------------------------

  describe("memory_search", () => {
    beforeEach(() => {
      registerMemoryTools(mockCtx, mockManager, "test-instance");
    });

    it("returns formatted results from manager", async () => {
      mockManager.search.mockResolvedValueOnce([
        {
          source: "session",
          path: "notes.md",
          startLine: 1,
          endLine: 5,
          score: 0.85,
          snippet: "found it",
        },
      ]);

      const result = await getHandler("memory_search")({ query: "test" }, ctx());
      expect(result.content[0].text).toContain("Found 1");
      expect(result.content[0].text).toContain("found it");
      expect(result.content[0].text).toContain("0.85");
    });

    it("returns no matches message when empty", async () => {
      const result = await getHandler("memory_search")({ query: "nonexistent" }, ctx());
      expect(result.content[0].text).toContain("No matches found");
    });

    it("passes instanceId to manager.search", async () => {
      await getHandler("memory_search")({ query: "test" }, ctx());
      expect(mockManager.search).toHaveBeenCalledWith(
        "test",
        expect.objectContaining({ instanceId: "test-instance" }),
      );
    });

    it("clamps maxResults to MAX_SEARCH_RESULTS", async () => {
      await getHandler("memory_search")({ query: "test", maxResults: 999 }, ctx());
      expect(mockManager.search).toHaveBeenCalledWith(
        "test",
        expect.objectContaining({ maxResults: MAX_SEARCH_RESULTS }),
      );
    });

    it("handles non-finite maxResults (NaN) gracefully with default 10", async () => {
      await getHandler("memory_search")({ query: "test", maxResults: NaN }, ctx());
      expect(mockManager.search).toHaveBeenCalledWith(
        "test",
        expect.objectContaining({ maxResults: 10 }),
      );
    });

    it("returns error when search throws", async () => {
      mockManager.search.mockRejectedValueOnce(new Error("db gone"));
      const result = await getHandler("memory_search")({ query: "test" }, ctx());
      expect(result.isError).toBe(true);
      expect(result.content[0].text).toContain("db gone");
    });

    it("passes valid temporal filter to manager", async () => {
      await getHandler("memory_search")({ query: "test", temporal: "7d" }, ctx());
      expect(mockManager.search).toHaveBeenCalledWith(
        "test",
        expect.objectContaining({
          temporal: expect.objectContaining({ after: expect.any(Number) }),
        }),
      );
    });

    it("returns error for invalid temporal filter", async () => {
      const result = await getHandler("memory_search")(
        { query: "test", temporal: "garbage" },
        ctx(),
      );
      expect(result.content[0].text).toContain("Invalid temporal filter");
    });
  });

  // -------------------------------------------------------------------------
  // memory_get
  // -------------------------------------------------------------------------

  describe("memory_get", () => {
    beforeEach(() => {
      registerMemoryTools(mockCtx, mockManager);
      makeSessionDir();
    });

    it("reads full file and returns JSON with totalLines", async () => {
      writeFileSync(
        join(sessionsDir, "test-session", "memory", "data.md"),
        "line1\nline2\nline3",
      );

      const result = await getHandler("memory_get")({ path: "memory/data.md" }, ctx());
      const parsed = JSON.parse(result.content[0].text);
      expect(parsed.totalLines).toBe(3);
      expect(parsed.text).toContain("line1");
    });

    it("supports line range with from and lines", async () => {
      writeFileSync(
        join(sessionsDir, "test-session", "memory", "data.md"),
        "a\nb\nc\nd\ne",
      );

      const result = await getHandler("memory_get")(
        { path: "memory/data.md", from: 2, lines: 2 },
        ctx(),
      );
      const parsed = JSON.parse(result.content[0].text);
      expect(parsed.from).toBe(2);
      expect(parsed.to).toBe(3);
      expect(parsed.text).toBe("b\nc");
    });

    it("returns error for missing file", async () => {
      const result = await getHandler("memory_get")({ path: "memory/nope.md" }, ctx());
      expect(result.isError).toBe(true);
      expect(result.content[0].text).toContain("File not found");
    });

    it("falls back from session root to memory dir", async () => {
      writeFileSync(
        join(sessionsDir, "test-session", "memory", "fallback.md"),
        "found in memory",
      );

      const result = await getHandler("memory_get")({ path: "fallback.md" }, ctx());
      const parsed = JSON.parse(result.content[0].text);
      expect(parsed.text).toContain("found in memory");
    });
  });

  // -------------------------------------------------------------------------
  // self_reflect
  // -------------------------------------------------------------------------

  describe("self_reflect", () => {
    beforeEach(() => {
      registerMemoryTools(mockCtx, mockManager);
      makeSessionDir();
    });

    it("creates SELF.md and adds reflection", async () => {
      const result = await getHandler("self_reflect")(
        { reflection: "I learned something" },
        ctx(),
      );
      expect(result.content[0].text).toContain("Reflection added");

      const selfPath = join(sessionsDir, "test-session", "memory", "SELF.md");
      const content = readFileSync(selfPath, "utf-8");
      expect(content).toContain("I learned something");
    });

    it("adds a tattoo to SELF.md", async () => {
      const result = await getHandler("self_reflect")({ tattoo: "be kind" }, ctx());
      expect(result.content[0].text).toContain("be kind");

      const content = readFileSync(
        join(sessionsDir, "test-session", "memory", "SELF.md"),
        "utf-8",
      );
      expect(content).toContain("be kind");
    });

    it("appends tattoo to existing Tattoos section", async () => {
      const selfPath = join(sessionsDir, "test-session", "memory", "SELF.md");
      writeFileSync(selfPath, '# SELF.md\n\n## Tattoos\n\n- "first"\n\n## Reflections\n');

      await getHandler("self_reflect")({ tattoo: "second" }, ctx());

      const content = readFileSync(selfPath, "utf-8");
      expect(content).toContain('"first"');
      expect(content).toContain("second");
    });

    it("returns error when neither reflection nor tattoo given", async () => {
      const result = await getHandler("self_reflect")({}, ctx());
      expect(result.isError).toBe(true);
      expect(result.content[0].text).toContain("reflection");
    });

    it("rejects oversized reflection (>64 KB by bytes)", async () => {
      const big = "x".repeat(65_537);
      const result = await getHandler("self_reflect")({ reflection: big }, ctx());
      expect(result.isError).toBe(true);
      expect(result.content[0].text).toContain("exceeds maximum");
    });

    it("rejects oversized tattoo (>64 KB by bytes)", async () => {
      const big = "x".repeat(65_537);
      const result = await getHandler("self_reflect")({ tattoo: big }, ctx());
      expect(result.isError).toBe(true);
    });

    it("rejects oversized section header (>256 bytes)", async () => {
      const result = await getHandler("self_reflect")(
        { reflection: "ok", section: "x".repeat(257) },
        ctx(),
      );
      expect(result.isError).toBe(true);
      expect(result.content[0].text).toContain("section");
    });

    it("uses custom section header when provided", async () => {
      await getHandler("self_reflect")(
        { reflection: "deep thought", section: "Philosophy" },
        ctx(),
      );
      const content = readFileSync(
        join(sessionsDir, "test-session", "memory", "SELF.md"),
        "utf-8",
      );
      expect(content).toContain("Philosophy");
    });
  });
});
