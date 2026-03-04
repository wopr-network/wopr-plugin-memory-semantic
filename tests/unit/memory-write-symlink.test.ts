import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { existsSync, mkdirSync, readFileSync, writeFileSync, symlinkSync, rmSync } from "node:fs";
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

describe("memory_write symlink guard (TOCTOU)", () => {
  let tmpBase: string;
  let sessionsDir: string;
  let sessionDir: string;
  let memoryDir: string;
  let outsideDir: string;
  let outsideFile: string;
  let ctx: ReturnType<typeof createMockCtx>;

  beforeEach(() => {
    tmpBase = join(tmpdir(), `wopr-symlink-test-${Date.now()}`);
    sessionsDir = join(tmpBase, "sessions");
    sessionDir = join(sessionsDir, "default");
    memoryDir = join(sessionDir, "memory");
    outsideDir = join(tmpBase, "outside");
    outsideFile = join(outsideDir, "secret.txt");

    mkdirSync(memoryDir, { recursive: true });
    mkdirSync(outsideDir, { recursive: true });
    writeFileSync(outsideFile, "ORIGINAL_SECRET");

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

  it("should reject writing through a symlink pointing outside the sandbox", async () => {
    // Simulate TOCTOU: place symlink at filePath after assertWithinBase would have checked
    const symlinkPath = join(memoryDir, "evil.md");
    symlinkSync(outsideFile, symlinkPath);

    const memoryWrite = ctx.tools["memory_write"];
    expect(memoryWrite).toBeDefined();

    const result = await memoryWrite.handler(
      { file: "evil.md", content: "OVERWRITTEN" },
      { sessionName: "default" },
    );

    expect(result.isError).toBe(true);
    expect(result.content[0].text.toLowerCase()).toContain("path");

    // Original file outside sandbox must be untouched
    expect(readFileSync(outsideFile, "utf-8")).toBe("ORIGINAL_SECRET");
  });

  it("should reject appending through a symlink pointing outside the sandbox", async () => {
    // Use a date-named file which triggers append mode
    const symlinkPath = join(memoryDir, "2026-01-01.md");
    symlinkSync(outsideFile, symlinkPath);

    const memoryWrite = ctx.tools["memory_write"];
    const result = await memoryWrite.handler(
      { file: "2026-01-01.md", content: "APPENDED" },
      { sessionName: "default" },
    );

    expect(result.isError).toBe(true);
    expect(result.content[0].text.toLowerCase()).toContain("path");

    expect(readFileSync(outsideFile, "utf-8")).toBe("ORIGINAL_SECRET");
  });

  it("should still allow writing a normal (non-symlink) file", async () => {
    const memoryWrite = ctx.tools["memory_write"];
    const result = await memoryWrite.handler(
      { file: "notes.md", content: "hello" },
      { sessionName: "default" },
    );

    expect(result.isError).toBeFalsy();
    const writtenPath = join(memoryDir, "notes.md");
    expect(existsSync(writtenPath)).toBe(true);
    expect(readFileSync(writtenPath, "utf-8")).toBe("hello");
  });

  it("should reject a dangling symlink (target does not exist outside sandbox)", async () => {
    // dangling: the symlink target doesn't exist at all
    const danglingTarget = join(outsideDir, "nonexistent.txt");
    const symlinkPath = join(memoryDir, "dangling.md");
    symlinkSync(danglingTarget, symlinkPath);

    const memoryWrite = ctx.tools["memory_write"];
    const result = await memoryWrite.handler(
      { file: "dangling.md", content: "INJECTED" },
      { sessionName: "default" },
    );

    expect(result.isError).toBe(true);
    expect(result.content[0].text.toLowerCase()).toContain("path");
    // The dangling target must not have been created
    expect(existsSync(danglingTarget)).toBe(false);
  });

  it("should reject symlink in ROOT_FILES branch (e.g. MEMORY.md at session root)", async () => {
    // Place symlink at session root for a ROOT_FILES entry
    const symlinkPath = join(sessionDir, "MEMORY.md");
    symlinkSync(outsideFile, symlinkPath);

    const memoryWrite = ctx.tools["memory_write"];
    const result = await memoryWrite.handler(
      { file: "MEMORY.md", content: "OVERWRITTEN" },
      { sessionName: "default" },
    );

    expect(result.isError).toBe(true);
    expect(result.content[0].text.toLowerCase()).toContain("path");
    expect(readFileSync(outsideFile, "utf-8")).toBe("ORIGINAL_SECRET");
  });
});
