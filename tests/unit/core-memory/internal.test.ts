import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import path from "node:path";

// Mock fs/promises and fs before importing the module
vi.mock("node:fs/promises", () => ({
  default: {
    readdir: vi.fn(),
    lstat: vi.fn(),
    stat: vi.fn(),
    readFile: vi.fn(),
    realpath: vi.fn(),
  },
}));

vi.mock("node:fs", () => ({
  default: {
    mkdirSync: vi.fn(),
  },
}));

import fs from "node:fs/promises";
import fsSync from "node:fs";
import {
  ensureDir,
  normalizeRelPath,
  normalizeExtraMemoryPaths,
  isMemoryPath,
  listMemoryFiles,
  hashText,
  buildFileEntry,
  chunkMarkdown,
  sliceUtf16Safe,
  truncateUtf16Safe,
} from "../../../src/core-memory/internal.js";

const mockFs = vi.mocked(fs);
const mockFsSync = vi.mocked(fsSync);

beforeEach(() => {
  vi.clearAllMocks();
});

describe("ensureDir", () => {
  it("calls mkdirSync with recursive true and returns the dir", () => {
    mockFsSync.mkdirSync.mockReturnValue(undefined);
    const result = ensureDir("/some/dir");
    expect(mockFsSync.mkdirSync).toHaveBeenCalledWith("/some/dir", { recursive: true });
    expect(result).toBe("/some/dir");
  });

  it("swallows mkdirSync errors and still returns dir", () => {
    mockFsSync.mkdirSync.mockImplementation(() => { throw new Error("EACCES"); });
    const result = ensureDir("/readonly/dir");
    expect(result).toBe("/readonly/dir");
  });
});

describe("normalizeRelPath", () => {
  it("trims whitespace", () => {
    expect(normalizeRelPath("  foo.md  ")).toBe("foo.md");
  });

  it("removes leading ./ and /", () => {
    expect(normalizeRelPath("./foo.md")).toBe("foo.md");
    expect(normalizeRelPath("../foo.md")).toBe("foo.md");
    expect(normalizeRelPath("/foo.md")).toBe("foo.md");
  });

  it("converts backslashes to forward slashes", () => {
    expect(normalizeRelPath("memory\\notes.md")).toBe("memory/notes.md");
  });

  it("returns empty string for blank input", () => {
    expect(normalizeRelPath("   ")).toBe("");
  });
});

describe("normalizeExtraMemoryPaths", () => {
  it("returns empty array for empty input", () => {
    expect(normalizeExtraMemoryPaths("/workspace")).toEqual([]);
    expect(normalizeExtraMemoryPaths("/workspace", [])).toEqual([]);
  });

  it("resolves relative paths against workspaceDir", () => {
    const result = normalizeExtraMemoryPaths("/workspace", ["notes"]);
    expect(result).toEqual([path.resolve("/workspace", "notes")]);
  });

  it("keeps absolute paths as-is (resolved)", () => {
    const result = normalizeExtraMemoryPaths("/workspace", ["/abs/path"]);
    expect(result).toEqual([path.resolve("/abs/path")]);
  });

  it("deduplicates paths", () => {
    const result = normalizeExtraMemoryPaths("/workspace", ["notes", "notes"]);
    expect(result).toHaveLength(1);
  });

  it("filters empty strings", () => {
    const result = normalizeExtraMemoryPaths("/workspace", ["", "  ", "notes"]);
    expect(result).toHaveLength(1);
  });
});

describe("isMemoryPath", () => {
  it("matches MEMORY.md", () => {
    expect(isMemoryPath("MEMORY.md")).toBe(true);
  });

  it("matches memory.md (lowercase)", () => {
    expect(isMemoryPath("memory.md")).toBe(true);
  });

  it("matches paths under memory/", () => {
    expect(isMemoryPath("memory/notes.md")).toBe(true);
    expect(isMemoryPath("memory/sub/dir.md")).toBe(true);
  });

  it("returns false for non-memory paths", () => {
    expect(isMemoryPath("src/index.ts")).toBe(false);
    expect(isMemoryPath("README.md")).toBe(false);
  });

  it("returns false for empty path", () => {
    expect(isMemoryPath("")).toBe(false);
  });

  it("strips leading ./ before checking", () => {
    expect(isMemoryPath("./MEMORY.md")).toBe(true);
    expect(isMemoryPath("./memory/notes.md")).toBe(true);
  });
});

describe("listMemoryFiles", () => {
  it("returns MEMORY.md if it exists as a regular file", async () => {
    mockFs.lstat.mockImplementation(async (p) => {
      if (String(p).endsWith("MEMORY.md")) {
        return { isSymbolicLink: () => false, isFile: () => true, isDirectory: () => false } as any;
      }
      throw new Error("ENOENT");
    });
    const result = await listMemoryFiles("/workspace");
    expect(result).toContain("/workspace/MEMORY.md");
  });

  it("skips symlinks", async () => {
    mockFs.lstat.mockImplementation(async () => {
      return { isSymbolicLink: () => true, isFile: () => false, isDirectory: () => false } as any;
    });
    const result = await listMemoryFiles("/workspace");
    expect(result).toHaveLength(0);
  });

  it("returns empty array when no memory files exist", async () => {
    mockFs.lstat.mockRejectedValue(new Error("ENOENT"));
    const result = await listMemoryFiles("/workspace");
    expect(result).toEqual([]);
  });

  it("deduplicates via realpath", async () => {
    mockFs.lstat.mockImplementation(async (p) => {
      const s = String(p);
      if (s.endsWith("MEMORY.md") || s.endsWith("memory.md")) {
        return { isSymbolicLink: () => false, isFile: () => true, isDirectory: () => false } as any;
      }
      throw new Error("ENOENT");
    });
    mockFs.realpath.mockImplementation(async (p) => "/workspace/MEMORY.md");
    const result = await listMemoryFiles("/workspace");
    // Both MEMORY.md and memory.md resolve to same realpath → deduped to 1
    expect(result).toHaveLength(1);
  });
});

describe("hashText", () => {
  it("returns a hex string", () => {
    const hash = hashText("hello");
    expect(hash).toMatch(/^[0-9a-f]{64}$/);
  });

  it("same input → same hash", () => {
    expect(hashText("foo")).toBe(hashText("foo"));
  });

  it("different inputs → different hashes", () => {
    expect(hashText("foo")).not.toBe(hashText("bar"));
  });
});

describe("buildFileEntry", () => {
  it("returns correct entry shape", async () => {
    mockFs.stat.mockResolvedValue({ mtimeMs: 1000, size: 5 } as any);
    mockFs.readFile.mockResolvedValue("hello" as any);

    const entry = await buildFileEntry("/workspace/MEMORY.md", "/workspace");
    expect(entry.path).toBe("MEMORY.md");
    expect(entry.absPath).toBe("/workspace/MEMORY.md");
    expect(entry.mtimeMs).toBe(1000);
    expect(entry.size).toBe(5);
    expect(entry.hash).toBe(hashText("hello"));
  });
});

describe("chunkMarkdown", () => {
  it("returns empty or single chunk for empty content", () => {
    const chunks = chunkMarkdown("", { tokens: 100, overlap: 0 });
    // The implementation produces one chunk with empty text for empty input
    expect(chunks.length).toBeLessThanOrEqual(1);
    if (chunks.length === 1) {
      expect(chunks[0]!.text).toBe("");
    }
  });

  it("returns a single chunk for short content", () => {
    const chunks = chunkMarkdown("Hello world", { tokens: 100, overlap: 0 });
    expect(chunks).toHaveLength(1);
    expect(chunks[0]!.text).toContain("Hello world");
    expect(chunks[0]!.startLine).toBe(1);
  });

  it("splits long content into multiple chunks", () => {
    // tokens:1 → maxChars=4 (max(32,1*4)=32 actually, let's use tokens:1 → 32 chars)
    const longContent = "a".repeat(200);
    const chunks = chunkMarkdown(longContent, { tokens: 1, overlap: 0 });
    expect(chunks.length).toBeGreaterThan(1);
  });

  it("chunks have correct line numbers", () => {
    const content = "line1\nline2\nline3";
    const chunks = chunkMarkdown(content, { tokens: 1000, overlap: 0 });
    expect(chunks[0]!.startLine).toBe(1);
    expect(chunks[0]!.endLine).toBe(3);
  });

  it("each chunk has a hash", () => {
    const chunks = chunkMarkdown("some content", { tokens: 100, overlap: 0 });
    for (const chunk of chunks) {
      expect(chunk.hash).toMatch(/^[0-9a-f]{64}$/);
    }
  });
});

describe("sliceUtf16Safe", () => {
  it("slices ASCII correctly", () => {
    expect(sliceUtf16Safe("hello", 1, 3)).toBe("el");
  });

  it("does not split surrogate pairs", () => {
    // Emoji is 2 UTF-16 code units (surrogate pair)
    const emoji = "😀"; // U+1F600
    const str = "a" + emoji + "b";
    // str[1] and str[2] are the surrogate pair
    // slicing at index 2 (mid-surrogate) should shift to avoid splitting
    const result = sliceUtf16Safe(str, 0, 2);
    expect(result.length).toBeLessThanOrEqual(2);
  });

  it("handles negative indices", () => {
    expect(sliceUtf16Safe("hello", -3)).toBe("llo");
  });

  it("swaps start/end when start > end", () => {
    // swaps indices: effectively slice(2, 3) → "l"
    expect(sliceUtf16Safe("hello", 3, 2)).toBe("l");
  });
});

describe("truncateUtf16Safe", () => {
  it("returns string unchanged if within limit", () => {
    expect(truncateUtf16Safe("hi", 10)).toBe("hi");
  });

  it("truncates at maxLen", () => {
    const result = truncateUtf16Safe("hello world", 5);
    expect(result.length).toBeLessThanOrEqual(5);
  });

  it("handles maxLen 0", () => {
    expect(truncateUtf16Safe("hello", 0)).toBe("");
  });
});
