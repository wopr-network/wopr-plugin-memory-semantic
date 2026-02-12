/**
 * Embeddings tests (WOP-98)
 *
 * Tests sanitizeAndNormalizeEmbedding (exported pure function)
 * and provider factory error handling.
 */
import { describe, expect, it } from "vitest";

import { createEmbeddingProvider, createOpenAiEmbeddingProvider, createGeminiEmbeddingProvider, sanitizeAndNormalizeEmbedding } from "../../src/embeddings.js";
import { DEFAULT_CONFIG, type SemanticMemoryConfig } from "../../src/types.js";

function makeConfig(overrides: Partial<SemanticMemoryConfig> = {}): SemanticMemoryConfig {
  return { ...DEFAULT_CONFIG, ...overrides };
}

// =============================================================================
// sanitizeAndNormalizeEmbedding (logic verification)
// =============================================================================

describe("sanitizeAndNormalizeEmbedding", () => {
  it("should normalize to unit length", () => {
    const result = sanitizeAndNormalizeEmbedding([3, 4]);
    const magnitude = Math.sqrt(result.reduce((sum, v) => sum + v * v, 0));
    expect(magnitude).toBeCloseTo(1.0, 5);
  });

  it("should handle all-zero vectors", () => {
    const result = sanitizeAndNormalizeEmbedding([0, 0, 0]);
    expect(result).toEqual([0, 0, 0]);
  });

  it("should replace NaN with 0", () => {
    const result = sanitizeAndNormalizeEmbedding([1, NaN, 2]);
    expect(result[1]).not.toBeNaN();
  });

  it("should replace Infinity with 0", () => {
    const result = sanitizeAndNormalizeEmbedding([1, Infinity, 2]);
    expect(result[1]).not.toBe(Infinity);
  });

  it("should replace -Infinity with 0", () => {
    const result = sanitizeAndNormalizeEmbedding([1, -Infinity, 2]);
    expect(Number.isFinite(result[1])).toBe(true);
  });

  it("should preserve relative direction", () => {
    const result = sanitizeAndNormalizeEmbedding([3, 4, 0]);
    // 3/5 = 0.6, 4/5 = 0.8
    expect(result[0]).toBeCloseTo(0.6, 5);
    expect(result[1]).toBeCloseTo(0.8, 5);
    expect(result[2]).toBeCloseTo(0, 5);
  });

  it("should handle single-element vectors", () => {
    const result = sanitizeAndNormalizeEmbedding([5]);
    expect(result[0]).toBeCloseTo(1.0, 5);
  });

  it("should handle negative values", () => {
    const result = sanitizeAndNormalizeEmbedding([-3, 4]);
    const magnitude = Math.sqrt(result.reduce((sum, v) => sum + v * v, 0));
    expect(magnitude).toBeCloseTo(1.0, 5);
    expect(result[0]).toBeLessThan(0);
  });
});

// =============================================================================
// Provider factory error handling
// =============================================================================

describe("createOpenAiEmbeddingProvider", () => {
  it("should throw when no API key is available", async () => {
    const originalKey = process.env.OPENAI_API_KEY;
    delete process.env.OPENAI_API_KEY;

    try {
      await expect(
        createOpenAiEmbeddingProvider(makeConfig({ apiKey: undefined })),
      ).rejects.toThrow("No API key found for OpenAI");
    } finally {
      if (originalKey !== undefined) process.env.OPENAI_API_KEY = originalKey;
      else delete process.env.OPENAI_API_KEY;
    }
  });
});

describe("createGeminiEmbeddingProvider", () => {
  it("should throw when no API key is available", async () => {
    const origGoogle = process.env.GOOGLE_API_KEY;
    const origGemini = process.env.GEMINI_API_KEY;
    delete process.env.GOOGLE_API_KEY;
    delete process.env.GEMINI_API_KEY;

    try {
      await expect(
        createGeminiEmbeddingProvider(makeConfig({ apiKey: undefined })),
      ).rejects.toThrow("No API key found for Gemini");
    } finally {
      if (origGoogle !== undefined) process.env.GOOGLE_API_KEY = origGoogle;
      else delete process.env.GOOGLE_API_KEY;
      if (origGemini !== undefined) process.env.GEMINI_API_KEY = origGemini;
      else delete process.env.GEMINI_API_KEY;
    }
  });
});

describe("createEmbeddingProvider", () => {
  it("should route to OpenAI provider when provider is 'openai'", async () => {
    // Without an API key it should throw the OpenAI-specific error
    const origKey = process.env.OPENAI_API_KEY;
    delete process.env.OPENAI_API_KEY;

    try {
      await expect(
        createEmbeddingProvider(makeConfig({ provider: "openai", apiKey: undefined })),
      ).rejects.toThrow("No API key found for OpenAI");
    } finally {
      if (origKey !== undefined) process.env.OPENAI_API_KEY = origKey;
      else delete process.env.OPENAI_API_KEY;
    }
  });

  it("should route to Gemini provider when provider is 'gemini'", async () => {
    const origGoogle = process.env.GOOGLE_API_KEY;
    const origGemini = process.env.GEMINI_API_KEY;
    delete process.env.GOOGLE_API_KEY;
    delete process.env.GEMINI_API_KEY;

    try {
      await expect(
        createEmbeddingProvider(makeConfig({ provider: "gemini", apiKey: undefined })),
      ).rejects.toThrow("No API key found for Gemini");
    } finally {
      if (origGoogle !== undefined) process.env.GOOGLE_API_KEY = origGoogle;
      else delete process.env.GOOGLE_API_KEY;
      if (origGemini !== undefined) process.env.GEMINI_API_KEY = origGemini;
      else delete process.env.GEMINI_API_KEY;
    }
  });
});
