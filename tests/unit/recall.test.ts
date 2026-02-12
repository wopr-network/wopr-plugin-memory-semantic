/**
 * Auto-recall tests (WOP-98)
 *
 * Tests extractQueryFromMessage, formatMemoriesAsContext,
 * performAutoRecall, and injectMemoriesIntoMessages.
 */
import { describe, expect, it, vi } from "vitest";
import {
  extractQueryFromMessage,
  formatMemoriesAsContext,
  injectMemoriesIntoMessages,
  performAutoRecall,
  type RecallResult,
} from "../../src/recall.js";
import type { MemorySearchResult, SemanticMemoryConfig } from "../../src/types.js";
import { DEFAULT_CONFIG } from "../../src/types.js";

function makeConfig(overrides: Partial<SemanticMemoryConfig["autoRecall"]> = {}): SemanticMemoryConfig {
  return {
    ...DEFAULT_CONFIG,
    autoRecall: { ...DEFAULT_CONFIG.autoRecall, ...overrides },
  };
}

function makeMemory(overrides: Partial<MemorySearchResult> = {}): MemorySearchResult {
  return {
    path: "src/auth.ts",
    startLine: 10,
    endLine: 20,
    score: 0.85,
    snippet: "JWT tokens expire after 1 hour",
    content: "JWT tokens expire after 1 hour by default",
    source: "codebase",
    ...overrides,
  };
}

// =============================================================================
// extractQueryFromMessage
// =============================================================================

describe("extractQueryFromMessage", () => {
  it("should strip conversational prefixes", () => {
    // The regex strips one prefix per pass: "Hey" is removed, "can you" remains
    expect(extractQueryFromMessage("Hey can you help with authentication?")).toBe(
      "can you help with authentication?",
    );
    expect(extractQueryFromMessage("Please show me the database schema")).toBe(
      "show me the database schema",
    );
  });

  it("should return short messages as-is after prefix removal", () => {
    expect(extractQueryFromMessage("How does auth work?")).toBe("How does auth work?");
  });

  it("should extract question from long messages", () => {
    const long =
      "I've been working on the frontend and noticed some issues with the API. " +
      "What is the correct way to handle token refresh? " +
      "I tried several approaches but none worked.";
    const result = extractQueryFromMessage(long);
    expect(result).toContain("What is the correct way to handle token refresh?");
  });

  it("should extract first sentence from long non-question messages", () => {
    const long =
      "The authentication system uses JWT tokens with refresh rotation. " +
      "It was implemented last sprint and has been working well. " +
      "The tokens expire after one hour and need manual refresh.";
    const result = extractQueryFromMessage(long);
    expect(result).toContain("authentication system uses JWT tokens");
  });

  it("should handle very long messages without question or sentence patterns", () => {
    // A 300-char all-alpha string has no sentence breaks, so firstSentence match
    // captures the whole thing (>= 20 chars). The 200-char truncation only
    // applies when firstSentence match is too short.
    const long = "a".repeat(300);
    const result = extractQueryFromMessage(long);
    // Should return something (doesn't crash on long input)
    expect(result.length).toBeGreaterThan(0);
  });
});

// =============================================================================
// formatMemoriesAsContext
// =============================================================================

describe("formatMemoriesAsContext", () => {
  const config = makeConfig({ maxMemories: 5 });

  it("should return empty string for no memories", () => {
    expect(formatMemoriesAsContext([], config)).toBe("");
  });

  it("should wrap memories in relevant-memories tags", () => {
    const memories = [makeMemory()];
    const result = formatMemoriesAsContext(memories, config);
    expect(result).toContain("<relevant-memories>");
    expect(result).toContain("</relevant-memories>");
  });

  it("should include score as percentage", () => {
    const memories = [makeMemory({ score: 0.85 })];
    const result = formatMemoriesAsContext(memories, config);
    expect(result).toContain("[85%]");
  });

  it("should include snippet text", () => {
    const memories = [makeMemory({ snippet: "JWT tokens expire after 1 hour" })];
    const result = formatMemoriesAsContext(memories, config);
    expect(result).toContain("JWT tokens expire after 1 hour");
  });

  it("should include path and line info", () => {
    const memories = [makeMemory({ path: "src/auth.ts", startLine: 10 })];
    const result = formatMemoriesAsContext(memories, config);
    expect(result).toContain("(from: src/auth.ts:10)");
  });

  it("should limit to maxMemories", () => {
    const memories = Array.from({ length: 10 }, (_, i) =>
      makeMemory({ snippet: `Memory ${i}`, score: 0.9 - i * 0.05 }),
    );
    const result = formatMemoriesAsContext(memories, makeConfig({ maxMemories: 3 }));
    expect(result).toContain("Memory 0");
    expect(result).toContain("Memory 2");
    expect(result).not.toContain("Memory 3");
  });

  it("should skip path line for memories without path", () => {
    const memories = [makeMemory({ path: "" })];
    const result = formatMemoriesAsContext(memories, config);
    expect(result).not.toContain("(from:");
  });
});

// =============================================================================
// performAutoRecall
// =============================================================================

describe("performAutoRecall", () => {
  it("should return null when autoRecall is disabled", async () => {
    const config = makeConfig({ enabled: false });
    const searchManager = { search: vi.fn() } as any;

    const result = await performAutoRecall("How does auth work?", searchManager, config);
    expect(result).toBeNull();
    expect(searchManager.search).not.toHaveBeenCalled();
  });

  it("should return null for very short queries", async () => {
    const config = makeConfig({ enabled: true });
    const searchManager = { search: vi.fn() } as any;

    const result = await performAutoRecall("hi", searchManager, config);
    expect(result).toBeNull();
  });

  it("should search and return results above minScore", async () => {
    const config = makeConfig({ enabled: true, minScore: 0.4, maxMemories: 5 });
    const memories = [makeMemory({ score: 0.85 }), makeMemory({ score: 0.2 })];
    const searchManager = { search: vi.fn(async () => memories) } as any;

    const result = await performAutoRecall("How does auth work?", searchManager, config);
    expect(result).not.toBeNull();
    expect(result!.memories).toHaveLength(1); // Only score >= 0.4
    expect(result!.memories[0].score).toBe(0.85);
  });

  it("should return null when no results meet minScore", async () => {
    const config = makeConfig({ enabled: true, minScore: 0.9 });
    const memories = [makeMemory({ score: 0.5 })];
    const searchManager = { search: vi.fn(async () => memories) } as any;

    const result = await performAutoRecall("auth tokens", searchManager, config);
    expect(result).toBeNull();
  });

  it("should include formatted context in result", async () => {
    const config = makeConfig({ enabled: true, minScore: 0.3, maxMemories: 5 });
    const memories = [makeMemory({ score: 0.85, snippet: "JWT tokens expire hourly" })];
    const searchManager = { search: vi.fn(async () => memories) } as any;

    const result = await performAutoRecall("token expiry", searchManager, config);
    expect(result!.context).toContain("<relevant-memories>");
    expect(result!.context).toContain("JWT tokens expire hourly");
  });
});

// =============================================================================
// injectMemoriesIntoMessages
// =============================================================================

describe("injectMemoriesIntoMessages", () => {
  it("should return original messages when context is empty", () => {
    const messages = [
      { role: "user", content: "Hello" },
    ];
    const recall: RecallResult = { query: "test", memories: [], context: "" };

    const result = injectMemoriesIntoMessages(messages, recall);
    expect(result).toEqual(messages);
  });

  it("should inject context before the last user message", () => {
    const messages = [
      { role: "system", content: "You are helpful." },
      { role: "user", content: "How does auth work?" },
    ];
    const recall: RecallResult = {
      query: "auth",
      memories: [makeMemory()],
      context: "<relevant-memories>\n[85%] JWT tokens expire\n</relevant-memories>",
    };

    const result = injectMemoriesIntoMessages(messages, recall);
    expect(result).toHaveLength(3);
    expect(result[1].content).toContain("[Context from memory");
    expect(result[1].content).toContain("relevant-memories");
    expect(result[2].content).toBe("How does auth work?");
  });

  it("should not modify original array", () => {
    const messages = [
      { role: "user", content: "Hello" },
    ];
    const recall: RecallResult = {
      query: "test",
      memories: [makeMemory()],
      context: "<relevant-memories>\ntest\n</relevant-memories>",
    };

    const result = injectMemoriesIntoMessages(messages, recall);
    expect(messages).toHaveLength(1);
    expect(result).toHaveLength(2);
  });

  it("should return original messages if no user message exists", () => {
    const messages = [
      { role: "system", content: "You are helpful." },
    ];
    const recall: RecallResult = {
      query: "test",
      memories: [makeMemory()],
      context: "<relevant-memories>\ntest\n</relevant-memories>",
    };

    const result = injectMemoriesIntoMessages(messages, recall);
    expect(result).toEqual(messages);
  });
});
