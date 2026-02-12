/**
 * Search helpers tests (WOP-98)
 *
 * Tests buildFtsQuery, bm25RankToScore, and mergeHybridResults.
 */
import { describe, expect, it } from "vitest";
import {
  bm25RankToScore,
  buildFtsQuery,
  mergeHybridResults,
  type HybridKeywordResult,
  type HybridVectorResult,
} from "../../src/search.js";

// =============================================================================
// buildFtsQuery
// =============================================================================

describe("buildFtsQuery", () => {
  it("should tokenize and quote words joined with AND", () => {
    expect(buildFtsQuery("hello world")).toBe('"hello" AND "world"');
  });

  it("should strip non-alphanumeric characters", () => {
    expect(buildFtsQuery("hello, world!")).toBe('"hello" AND "world"');
  });

  it("should return null for empty input", () => {
    expect(buildFtsQuery("")).toBeNull();
  });

  it("should return null for whitespace-only input", () => {
    expect(buildFtsQuery("   ")).toBeNull();
  });

  it("should return null for input with only special characters", () => {
    expect(buildFtsQuery("!@#$%")).toBeNull();
  });

  it("should handle single word", () => {
    expect(buildFtsQuery("authentication")).toBe('"authentication"');
  });

  it("should handle underscored identifiers", () => {
    expect(buildFtsQuery("my_function_name")).toBe('"my_function_name"');
  });

  it("should split on double quotes (tokenizer treats them as non-alphanumeric)", () => {
    // The tokenizer regex [A-Za-z0-9_]+ splits on quotes, producing three tokens
    expect(buildFtsQuery('he"llo world')).toBe('"he" AND "llo" AND "world"');
  });
});

// =============================================================================
// bm25RankToScore
// =============================================================================

describe("bm25RankToScore", () => {
  it("should return 1 for rank 0 (best match)", () => {
    expect(bm25RankToScore(0)).toBe(1);
  });

  it("should return 0.5 for rank 1", () => {
    expect(bm25RankToScore(1)).toBe(0.5);
  });

  it("should return decreasing scores for increasing ranks", () => {
    const score1 = bm25RankToScore(1);
    const score5 = bm25RankToScore(5);
    const score10 = bm25RankToScore(10);
    expect(score1).toBeGreaterThan(score5);
    expect(score5).toBeGreaterThan(score10);
  });

  it("should handle negative rank by clamping to 0", () => {
    expect(bm25RankToScore(-5)).toBe(1);
  });

  it("should handle NaN by using fallback rank", () => {
    const score = bm25RankToScore(NaN);
    expect(score).toBeCloseTo(1 / 1000, 5);
  });

  it("should handle Infinity by using fallback rank", () => {
    const score = bm25RankToScore(Infinity);
    expect(score).toBeCloseTo(1 / 1000, 5);
  });

  it("should always return a value between 0 and 1", () => {
    for (const rank of [0, 1, 5, 10, 100, 1000, -1, NaN, Infinity]) {
      const score = bm25RankToScore(rank);
      expect(score).toBeGreaterThan(0);
      expect(score).toBeLessThanOrEqual(1);
    }
  });
});

// =============================================================================
// mergeHybridResults
// =============================================================================

function makeVectorResult(overrides: Partial<HybridVectorResult> = {}): HybridVectorResult {
  return {
    id: "v1",
    path: "src/auth.ts",
    startLine: 1,
    endLine: 10,
    source: "codebase",
    snippet: "vector snippet",
    content: "vector content",
    vectorScore: 0.9,
    ...overrides,
  };
}

function makeKeywordResult(overrides: Partial<HybridKeywordResult> = {}): HybridKeywordResult {
  return {
    id: "k1",
    path: "src/auth.ts",
    startLine: 1,
    endLine: 10,
    source: "codebase",
    snippet: "keyword snippet",
    content: "keyword content",
    textScore: 0.8,
    ...overrides,
  };
}

describe("mergeHybridResults", () => {
  it("should merge vector and keyword results by weighted score", () => {
    const vector = [makeVectorResult({ id: "a", vectorScore: 0.9 })];
    const keyword = [makeKeywordResult({ id: "b", textScore: 0.8 })];

    const merged = mergeHybridResults({
      vector,
      keyword,
      vectorWeight: 0.7,
      textWeight: 0.3,
    });

    expect(merged).toHaveLength(2);
    // "a" has vectorScore=0.9*0.7 + textScore=0*0.3 = 0.63
    // "b" has vectorScore=0*0.7 + textScore=0.8*0.3 = 0.24
    expect(merged[0].score).toBeCloseTo(0.63, 2);
    expect(merged[1].score).toBeCloseTo(0.24, 2);
  });

  it("should combine scores when same ID appears in both", () => {
    const vector = [makeVectorResult({ id: "shared", vectorScore: 0.8 })];
    const keyword = [makeKeywordResult({ id: "shared", textScore: 0.6 })];

    const merged = mergeHybridResults({
      vector,
      keyword,
      vectorWeight: 0.7,
      textWeight: 0.3,
    });

    expect(merged).toHaveLength(1);
    // 0.8*0.7 + 0.6*0.3 = 0.56 + 0.18 = 0.74
    expect(merged[0].score).toBeCloseTo(0.74, 2);
  });

  it("should sort by combined score descending", () => {
    const vector = [
      makeVectorResult({ id: "low", vectorScore: 0.3 }),
      makeVectorResult({ id: "high", vectorScore: 0.95 }),
    ];
    const keyword = [makeKeywordResult({ id: "mid", textScore: 0.9 })];

    const merged = mergeHybridResults({
      vector,
      keyword,
      vectorWeight: 0.7,
      textWeight: 0.3,
    });

    expect(merged[0].score).toBeGreaterThanOrEqual(merged[1].score);
    expect(merged[1].score).toBeGreaterThanOrEqual(merged[2].score);
  });

  it("should handle empty vector results", () => {
    const keyword = [makeKeywordResult({ id: "k1", textScore: 0.8 })];

    const merged = mergeHybridResults({
      vector: [],
      keyword,
      vectorWeight: 0.7,
      textWeight: 0.3,
    });

    expect(merged).toHaveLength(1);
    expect(merged[0].score).toBeCloseTo(0.24, 2);
  });

  it("should handle empty keyword results", () => {
    const vector = [makeVectorResult({ id: "v1", vectorScore: 0.9 })];

    const merged = mergeHybridResults({
      vector,
      keyword: [],
      vectorWeight: 0.7,
      textWeight: 0.3,
    });

    expect(merged).toHaveLength(1);
    expect(merged[0].score).toBeCloseTo(0.63, 2);
  });

  it("should handle both empty", () => {
    const merged = mergeHybridResults({
      vector: [],
      keyword: [],
      vectorWeight: 0.7,
      textWeight: 0.3,
    });

    expect(merged).toEqual([]);
  });

  it("should prefer keyword snippet and content when both exist for same ID", () => {
    const vector = [
      makeVectorResult({ id: "shared", snippet: "vector snip", content: "vector full" }),
    ];
    const keyword = [
      makeKeywordResult({ id: "shared", snippet: "keyword snip", content: "keyword full" }),
    ];

    const merged = mergeHybridResults({
      vector,
      keyword,
      vectorWeight: 0.7,
      textWeight: 0.3,
    });

    expect(merged[0].snippet).toBe("keyword snip");
    expect(merged[0].content).toBe("keyword full");
  });
});
