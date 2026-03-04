import { describe, it, expect, vi, beforeEach } from "vitest";

vi.mock("node:fs/promises", () => ({
  default: {
    readdir: vi.fn(),
    stat: vi.fn(),
    readFile: vi.fn(),
  },
}));

import fs from "node:fs/promises";
import {
  listSessionFiles,
  sessionPathForFile,
  extractSessionText,
  buildSessionEntry,
  getRecentSessionContent,
} from "../../../src/core-memory/session-files.js";

const mockFs = vi.mocked(fs);

beforeEach(() => {
  vi.clearAllMocks();
});

describe("listSessionFiles", () => {
  it("returns .conversation.jsonl files in directory", async () => {
    mockFs.readdir.mockResolvedValue([
      { name: "abc.conversation.jsonl", isFile: () => true } as any,
      { name: "def.conversation.jsonl", isFile: () => true } as any,
      { name: "other.txt", isFile: () => true } as any,
    ] as any);
    const files = await listSessionFiles("/sessions");
    expect(files).toHaveLength(2);
    expect(files[0]).toContain("abc.conversation.jsonl");
    expect(files[1]).toContain("def.conversation.jsonl");
  });

  it("returns empty array on readdir failure", async () => {
    mockFs.readdir.mockRejectedValue(new Error("ENOENT"));
    const files = await listSessionFiles("/sessions");
    expect(files).toEqual([]);
  });

  it("filters out non-files", async () => {
    mockFs.readdir.mockResolvedValue([
      { name: "dir", isFile: () => false } as any,
      { name: "abc.conversation.jsonl", isFile: () => true } as any,
    ] as any);
    const files = await listSessionFiles("/sessions");
    expect(files).toHaveLength(1);
  });
});

describe("sessionPathForFile", () => {
  it("returns sessions/<basename>", () => {
    const result = sessionPathForFile("/sessions/abc.conversation.jsonl");
    expect(result).toBe("sessions/abc.conversation.jsonl");
  });

  it("uses forward slashes", () => {
    const result = sessionPathForFile("/some/path/to/file.jsonl");
    expect(result).toBe("sessions/file.jsonl");
  });
});

describe("extractSessionText", () => {
  it("normalizes string input", () => {
    const result = extractSessionText("  hello   world  ");
    expect(result).toBe("hello world");
  });

  it("returns null for empty string", () => {
    expect(extractSessionText("")).toBeNull();
  });

  it("returns null for non-string, non-array", () => {
    expect(extractSessionText(42)).toBeNull();
    expect(extractSessionText(null)).toBeNull();
  });

  it("extracts text from array of content blocks", () => {
    const result = extractSessionText([
      { type: "text", text: "Hello" },
      { type: "image", data: "..." },
      { type: "text", text: "World" },
    ]);
    expect(result).toBe("Hello World");
  });

  it("returns null for array with no text blocks", () => {
    const result = extractSessionText([{ type: "image", data: "..." }]);
    expect(result).toBeNull();
  });

  it("returns null for empty array", () => {
    expect(extractSessionText([])).toBeNull();
  });
});

describe("buildSessionEntry", () => {
  it("parses WOPR format messages", async () => {
    const jsonl = [
      JSON.stringify({ type: "message", from: "Alice", content: "Hello there" }),
      JSON.stringify({ type: "response", from: "WOPR", content: "Hi Alice" }),
    ].join("\n");

    mockFs.stat.mockResolvedValue({ mtimeMs: 1000, size: jsonl.length } as any);
    mockFs.readFile.mockResolvedValue(jsonl as any);

    const entry = await buildSessionEntry("/sessions/test.conversation.jsonl");
    expect(entry).not.toBeNull();
    expect(entry!.content).toContain("User: Hello there");
    expect(entry!.content).toContain("Assistant: Hi Alice");
    expect(entry!.path).toBe("sessions/test.conversation.jsonl");
  });

  it("parses OpenClaw/Claude format messages", async () => {
    const jsonl = [
      JSON.stringify({ type: "message", message: { role: "user", content: "Question?" } }),
      JSON.stringify({ type: "message", message: { role: "assistant", content: "Answer!" } }),
    ].join("\n");

    mockFs.stat.mockResolvedValue({ mtimeMs: 2000, size: jsonl.length } as any);
    mockFs.readFile.mockResolvedValue(jsonl as any);

    const entry = await buildSessionEntry("/sessions/test.conversation.jsonl");
    expect(entry).not.toBeNull();
    expect(entry!.content).toContain("User: Question?");
    expect(entry!.content).toContain("Assistant: Answer!");
  });

  it("skips invalid JSON lines", async () => {
    const jsonl = "not json\n" + JSON.stringify({ type: "message", from: "Alice", content: "Hi" });
    mockFs.stat.mockResolvedValue({ mtimeMs: 1000, size: jsonl.length } as any);
    mockFs.readFile.mockResolvedValue(jsonl as any);

    const entry = await buildSessionEntry("/sessions/test.conversation.jsonl");
    expect(entry).not.toBeNull();
    expect(entry!.content).toContain("User: Hi");
  });

  it("returns null on stat failure", async () => {
    mockFs.stat.mockRejectedValue(new Error("ENOENT"));
    const entry = await buildSessionEntry("/sessions/missing.jsonl");
    expect(entry).toBeNull();
  });
});

describe("getRecentSessionContent", () => {
  it("returns last N messages", async () => {
    const messages = Array.from({ length: 20 }, (_, i) =>
      JSON.stringify({ type: "message", from: `User${i}`, content: `msg${i}` }),
    );
    mockFs.readFile.mockResolvedValue(messages.join("\n") as any);

    const content = await getRecentSessionContent("/sessions/test.jsonl", 5);
    expect(content).not.toBeNull();
    const lines = content!.split("\n");
    expect(lines).toHaveLength(5);
  });

  it("skips slash commands", async () => {
    const jsonl = [
      JSON.stringify({ type: "message", from: "Alice", content: "/clear" }),
      JSON.stringify({ type: "message", from: "Alice", content: "real message" }),
    ].join("\n");
    mockFs.readFile.mockResolvedValue(jsonl as any);

    const content = await getRecentSessionContent("/sessions/test.jsonl", 10);
    expect(content).not.toContain("/clear");
    expect(content).toContain("real message");
  });

  it("returns null on read failure", async () => {
    mockFs.readFile.mockRejectedValue(new Error("ENOENT"));
    const result = await getRecentSessionContent("/sessions/missing.jsonl");
    expect(result).toBeNull();
  });
});
