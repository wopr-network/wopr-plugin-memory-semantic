/**
 * Types and DEFAULT_CONFIG tests (WOP-98)
 *
 * Validates DEFAULT_CONFIG has sensible defaults for all fields.
 */
import { describe, expect, it } from "vitest";
import { DEFAULT_CONFIG } from "../../src/types.js";

describe("DEFAULT_CONFIG", () => {
  it("should have auto provider", () => {
    expect(DEFAULT_CONFIG.provider).toBe("auto");
  });

  it("should have a default model", () => {
    expect(DEFAULT_CONFIG.model).toBe("text-embedding-3-small");
  });

  it("should have search config with sensible defaults", () => {
    expect(DEFAULT_CONFIG.search.maxResults).toBeGreaterThan(0);
    expect(DEFAULT_CONFIG.search.minScore).toBeGreaterThan(0);
    expect(DEFAULT_CONFIG.search.minScore).toBeLessThan(1);
    expect(DEFAULT_CONFIG.search.candidateMultiplier).toBeGreaterThanOrEqual(1);
  });

  it("should have hybrid search enabled with weights summing roughly to 1", () => {
    expect(DEFAULT_CONFIG.hybrid.enabled).toBe(true);
    const total = DEFAULT_CONFIG.hybrid.vectorWeight + DEFAULT_CONFIG.hybrid.textWeight;
    expect(total).toBeCloseTo(1.0, 1);
  });

  it("should have vector weight higher than text weight", () => {
    expect(DEFAULT_CONFIG.hybrid.vectorWeight).toBeGreaterThan(DEFAULT_CONFIG.hybrid.textWeight);
  });

  it("should have autoRecall enabled with sensible defaults", () => {
    expect(DEFAULT_CONFIG.autoRecall.enabled).toBe(true);
    expect(DEFAULT_CONFIG.autoRecall.maxMemories).toBeGreaterThan(0);
    expect(DEFAULT_CONFIG.autoRecall.minScore).toBeGreaterThan(0);
    expect(DEFAULT_CONFIG.autoRecall.minScore).toBeLessThan(1);
  });

  it("should have autoCapture enabled with sensible defaults", () => {
    expect(DEFAULT_CONFIG.autoCapture.enabled).toBe(true);
    expect(DEFAULT_CONFIG.autoCapture.maxPerConversation).toBeGreaterThan(0);
    expect(DEFAULT_CONFIG.autoCapture.minLength).toBeGreaterThan(0);
    expect(DEFAULT_CONFIG.autoCapture.maxLength).toBeGreaterThan(DEFAULT_CONFIG.autoCapture.minLength);
  });

  it("should have vector store enabled", () => {
    expect(DEFAULT_CONFIG.store.vectorEnabled).toBe(true);
  });

  it("should have cache enabled with a max entry limit", () => {
    expect(DEFAULT_CONFIG.cache.enabled).toBe(true);
    expect(DEFAULT_CONFIG.cache.maxEntries).toBeGreaterThan(0);
  });

  it("should have chunking config with multi-scale", () => {
    expect(DEFAULT_CONFIG.chunking.tokens).toBeGreaterThan(0);
    expect(DEFAULT_CONFIG.chunking.overlap).toBeGreaterThan(0);
    expect(DEFAULT_CONFIG.chunking.overlap).toBeLessThan(DEFAULT_CONFIG.chunking.tokens);
    expect(DEFAULT_CONFIG.chunking.multiScale?.enabled).toBe(true);
    expect(DEFAULT_CONFIG.chunking.multiScale?.scales.length).toBeGreaterThanOrEqual(2);
  });

  it("should have multi-scale scales ordered by increasing token count", () => {
    const scales = DEFAULT_CONFIG.chunking.multiScale!.scales;
    for (let i = 1; i < scales.length; i++) {
      expect(scales[i].tokens).toBeGreaterThan(scales[i - 1].tokens);
    }
  });
});
