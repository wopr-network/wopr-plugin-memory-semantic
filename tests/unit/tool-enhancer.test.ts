/**
 * Tool enhancer tests (WOP-98)
 *
 * Tests enhanceSearch and createSearchToolHandler.
 */
import { describe, expect, it, vi } from "vitest";
import { createSearchToolHandler, enhanceSearch, type ToolEnhancerOptions } from "../../src/tool-enhancer.js";
import type { MemorySearchResult, SemanticMemoryConfig } from "../../src/types.js";
import { DEFAULT_CONFIG } from "../../src/types.js";

function makeOptions(searchResults: MemorySearchResult[] = []): ToolEnhancerOptions {
  return {
    config: DEFAULT_CONFIG,
    searchManager: {
      search: vi.fn(async () => searchResults),
      addEntry: vi.fn(),
      addEntriesBatch: vi.fn(),
      close: vi.fn(),
      getEntryCount: vi.fn(() => 0),
      hasEntry: vi.fn(() => false),
      getEntry: vi.fn(),
    } as any,
  };
}

function makeResult(overrides: Partial<MemorySearchResult> = {}): MemorySearchResult {
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
// enhanceSearch
// =============================================================================

describe("enhanceSearch", () => {
  it("should return semantic results when available above minScore", async () => {
    const options = makeOptions([makeResult({ score: 0.85 })]);

    const result = await enhanceSearch({ query: "auth tokens", minScore: 0.3 }, options);

    expect(result.enhanced).toBe(true);
    expect(result.source).toBe("semantic");
    expect(result.results).toHaveLength(1);
  });

  it("should filter results below minScore", async () => {
    const options = makeOptions([
      makeResult({ score: 0.85 }),
      makeResult({ score: 0.2 }),
    ]);

    const result = await enhanceSearch({ query: "auth", minScore: 0.35 }, options);

    expect(result.results).toHaveLength(1);
    expect(result.results[0].score).toBe(0.85);
  });

  it("should fall back when no semantic results above minScore", async () => {
    const options = makeOptions([makeResult({ score: 0.1 })]);
    const fallback = vi.fn(async () => [makeResult({ score: 0.5, snippet: "fallback result" })]);

    const result = await enhanceSearch({ query: "auth", minScore: 0.35 }, options, fallback);

    expect(result.enhanced).toBe(false);
    expect(result.source).toBe("fallback");
    expect(result.results[0].snippet).toBe("fallback result");
  });

  it("should return empty results when no semantic results and no fallback", async () => {
    const options = makeOptions([makeResult({ score: 0.1 })]);

    const result = await enhanceSearch({ query: "auth", minScore: 0.35 }, options);

    expect(result.results).toEqual([]);
    expect(result.enhanced).toBe(true);
    expect(result.source).toBe("semantic");
  });

  it("should use fallback when semantic search throws", async () => {
    const options = makeOptions();
    (options.searchManager.search as any).mockRejectedValueOnce(new Error("Network error"));
    const fallback = vi.fn(async () => [makeResult({ snippet: "fallback" })]);

    const result = await enhanceSearch({ query: "auth" }, options, fallback);

    expect(result.enhanced).toBe(false);
    expect(result.source).toBe("fallback");
  });

  it("should propagate error when semantic search throws and no fallback", async () => {
    const options = makeOptions();
    (options.searchManager.search as any).mockRejectedValueOnce(new Error("Network error"));

    await expect(enhanceSearch({ query: "auth" }, options)).rejects.toThrow("Network error");
  });

  it("should return empty when both semantic and fallback fail", async () => {
    const options = makeOptions();
    (options.searchManager.search as any).mockRejectedValueOnce(new Error("Semantic fail"));
    const fallback = vi.fn(async () => {
      throw new Error("Fallback fail");
    });

    const result = await enhanceSearch({ query: "auth" }, options, fallback);

    expect(result.results).toEqual([]);
    expect(result.enhanced).toBe(false);
    expect(result.source).toBe("fallback");
  });

  it("should use default maxResults and minScore", async () => {
    const options = makeOptions([makeResult({ score: 0.85 })]);

    const result = await enhanceSearch({ query: "auth" }, options);

    expect(options.searchManager.search).toHaveBeenCalledWith("auth", 10);
    expect(result.results).toHaveLength(1);
  });
});

// =============================================================================
// createSearchToolHandler
// =============================================================================

describe("createSearchToolHandler", () => {
  it("should return formatted results for successful search", async () => {
    const options = makeOptions([makeResult({ score: 0.85, snippet: "JWT tokens" })]);
    const handler = createSearchToolHandler(options);

    const result = await handler({ query: "auth tokens" });

    expect(result.content).toHaveLength(1);
    expect(result.content[0].type).toBe("text");
    expect(result.content[0].text).toContain("Found 1 results");
    expect(result.content[0].text).toContain("JWT tokens");
    expect(result.content[0].text).toContain("(semantic)");
  });

  it("should return no-matches message when no results", async () => {
    const options = makeOptions([]);
    const handler = createSearchToolHandler(options);

    const result = await handler({ query: "nonexistent" });

    expect(result.content[0].text).toContain('No matches found for "nonexistent"');
  });

  it("should return error message when search fails", async () => {
    const options = makeOptions();
    (options.searchManager.search as any).mockRejectedValueOnce(new Error("Search crashed"));
    const handler = createSearchToolHandler(options);

    const result = await handler({ query: "auth" });

    expect(result.content[0].text).toContain("Search failed: Search crashed");
  });

  it("should respect custom maxResults and minScore", async () => {
    const options = makeOptions([
      makeResult({ score: 0.9, snippet: "high" }),
      makeResult({ score: 0.5, snippet: "mid" }),
      makeResult({ score: 0.1, snippet: "low" }),
    ]);
    const handler = createSearchToolHandler(options);

    const result = await handler({ query: "auth", maxResults: 2, minScore: 0.4 });

    // Only scores >= 0.4 should be included
    expect(result.content[0].text).toContain("Found 2 results");
    expect(result.content[0].text).not.toContain("low");
  });
});
