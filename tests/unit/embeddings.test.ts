/**
 * Embeddings tests (WOP-98)
 *
 * Tests sanitizeAndNormalizeEmbedding (exported pure function)
 * and provider factory error handling.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { createEmbeddingProvider, createOpenAiEmbeddingProvider, createGeminiEmbeddingProvider, createLocalEmbeddingProvider, sanitizeAndNormalizeEmbedding } from "../../src/embeddings.js";
import { DEFAULT_CONFIG, type SemanticMemoryConfig } from "../../src/types.js";

function snapshotEnv(keys: string[]): () => void {
  const saved: Record<string, string | undefined> = {};
  for (const key of keys) saved[key] = process.env[key];
  return () => {
    for (const [key, value] of Object.entries(saved)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  };
}

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
  let restoreEnv: () => void;

  beforeEach(() => {
    restoreEnv = snapshotEnv(["OPENAI_API_KEY"]);
  });

  afterEach(() => {
    restoreEnv();
  });

  it("should throw when no API key is available", async () => {
    delete process.env.OPENAI_API_KEY;

    await expect(
      createOpenAiEmbeddingProvider(makeConfig({ apiKey: undefined })),
    ).rejects.toThrow("No API key found for OpenAI");
  });

  it("should ignore config.apiKey and only use env var", async () => {
    delete process.env.OPENAI_API_KEY;
    await expect(
      createOpenAiEmbeddingProvider(makeConfig({ provider: "openai", apiKey: "sk-from-config" })),
    ).rejects.toThrow("OPENAI_API_KEY");
  });
});

describe("createGeminiEmbeddingProvider", () => {
  let restoreEnv: () => void;

  beforeEach(() => {
    restoreEnv = snapshotEnv(["GOOGLE_API_KEY", "GEMINI_API_KEY"]);
  });

  afterEach(() => {
    restoreEnv();
  });

  it("should throw when no API key is available", async () => {
    delete process.env.GOOGLE_API_KEY;
    delete process.env.GEMINI_API_KEY;

    await expect(
      createGeminiEmbeddingProvider(makeConfig({ apiKey: undefined })),
    ).rejects.toThrow("No API key found for Gemini");
  });

  it("should ignore config.apiKey and only use env var", async () => {
    delete process.env.GOOGLE_API_KEY;
    delete process.env.GEMINI_API_KEY;
    await expect(
      createGeminiEmbeddingProvider(makeConfig({ provider: "gemini", apiKey: "gemini-from-config" })),
    ).rejects.toThrow("GOOGLE_API_KEY");
  });
});

// =============================================================================
// Gemini error payload truncation (WOP-1554)
// =============================================================================

describe("Gemini error payload truncation (WOP-1554)", () => {
  let restoreEnv: () => void;

  beforeEach(() => {
    restoreEnv = snapshotEnv(["GOOGLE_API_KEY", "GEMINI_API_KEY"]);
  });

  afterEach(() => {
    restoreEnv();
  });

  it("should truncate long error payloads to 200 chars", async () => {
    process.env.GOOGLE_API_KEY = "test-key";
    const longPayload = "x".repeat(500);
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(longPayload, { status: 400 }),
    );
    try {
      const provider = await createGeminiEmbeddingProvider(
        makeConfig({ provider: "gemini" }),
      );
      await expect(provider.embedQuery("hello")).rejects.toThrow(
        /Gemini embeddings failed: 400 x{200}\.\.\.\[truncated\]/,
      );
    } finally {
      fetchSpy.mockRestore();
    }
  });

  it("should preserve short error payloads as-is", async () => {
    process.env.GOOGLE_API_KEY = "test-key";
    const shortPayload = "Bad request";
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(shortPayload, { status: 400 }),
    );
    try {
      const provider = await createGeminiEmbeddingProvider(
        makeConfig({ provider: "gemini" }),
      );
      await expect(provider.embedQuery("hello")).rejects.toThrow(
        `Gemini embeddings failed: 400 Bad request`,
      );
    } finally {
      fetchSpy.mockRestore();
    }
  });

  it("should not leak API keys in truncated error messages", async () => {
    process.env.GOOGLE_API_KEY = "test-key";
    const apiKey = "AIzaSyDEADBEEF1234567890SECRETKEY";
    // Place the key after the 200-char mark so truncation removes it
    const payloadWithKey = "a".repeat(201) + apiKey;
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(payloadWithKey, { status: 400 }),
    );
    try {
      const provider = await createGeminiEmbeddingProvider(
        makeConfig({ provider: "gemini" }),
      );
      await expect(provider.embedQuery("hello")).rejects.toSatisfy((err: Error) => {
        return !err.message.includes(apiKey);
      });
    } finally {
      fetchSpy.mockRestore();
    }
  });

  it("should truncate long batch error payloads", async () => {
    process.env.GOOGLE_API_KEY = "test-key";
    const longPayload = "y".repeat(500);
    let callCount = 0;
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
      callCount++;
      return new Response(longPayload, { status: 500 });
    });
    try {
      const provider = await createGeminiEmbeddingProvider(
        makeConfig({ provider: "gemini" }),
      );
      await expect(provider.embedBatch(["hello", "world"])).rejects.toThrow(
        /Gemini batch embeddings failed: 500 y{200}\.\.\.\[truncated\]/,
      );
    } finally {
      fetchSpy.mockRestore();
    }
  });
});

describe("createLocalEmbeddingProvider", () => {
  it("should throw when node-llama-cpp is not installed", async () => {
    vi.mock("node-llama-cpp", () => {
      throw new Error("Cannot find module 'node-llama-cpp'");
    });
    await expect(
      createLocalEmbeddingProvider(makeConfig({ provider: "local" })),
    ).rejects.toThrow(/node-llama-cpp/);
    vi.unmock("node-llama-cpp");
  });
});

describe("createEmbeddingProvider", () => {
  let restoreEnv: () => void;

  beforeEach(() => {
    restoreEnv = snapshotEnv(["OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"]);
  });

  afterEach(() => {
    restoreEnv();
  });

  it("should route to OpenAI provider when provider is 'openai'", async () => {
    // Without an API key it should throw the OpenAI-specific error
    delete process.env.OPENAI_API_KEY;

    await expect(
      createEmbeddingProvider(makeConfig({ provider: "openai", apiKey: undefined })),
    ).rejects.toThrow("No API key found for OpenAI");
  });

  it("should route to Gemini provider when provider is 'gemini'", async () => {
    delete process.env.GOOGLE_API_KEY;
    delete process.env.GEMINI_API_KEY;

    await expect(
      createEmbeddingProvider(makeConfig({ provider: "gemini", apiKey: undefined })),
    ).rejects.toThrow("No API key found for Gemini");
  });
});
