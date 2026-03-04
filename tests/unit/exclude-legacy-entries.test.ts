/**
 * Tests for excludeLegacyEntries config flag (WOP-1553)
 *
 * Verifies that when excludeLegacyEntries is true, tenant-scoped queries
 * exclude entries with no instanceId (legacy/global entries).
 */
import { describe, expect, it } from "vitest";
import { createSemanticSearchManager } from "../../src/search.js";
import { DEFAULT_CONFIG, type SemanticMemoryConfig } from "../../src/types.js";

function createMockEmbeddingProvider(dims = 64) {
  const embedText = async (text: string): Promise<number[]> => {
    const vec = new Array(dims).fill(0);
    for (let i = 0; i < text.length; i++) {
      vec[i % dims] += text.charCodeAt(i) / 1000;
    }
    const mag = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
    return vec.map((v) => v / (mag || 1));
  };
  return {
    id: "mock",
    model: "mock-model",
    embedQuery: embedText,
    embedBatch: async (texts: string[]) => Promise.all(texts.map(embedText)),
  };
}

describe("excludeLegacyEntries config flag", () => {
  const baseConfig: SemanticMemoryConfig = {
    ...DEFAULT_CONFIG,
    search: { ...DEFAULT_CONFIG.search, minScore: 0.0, excludeLegacyEntries: false },
    hybrid: { enabled: false, vectorWeight: 0.7, textWeight: 0.3 },
  };

  async function seedLegacyAndScoped(config: SemanticMemoryConfig) {
    const provider = createMockEmbeddingProvider();
    const sm = await createSemanticSearchManager(config, provider);

    // Legacy entry (no instanceId)
    await sm.addEntry(
      {
        id: "legacy-1",
        path: "global/MEMORY.md",
        startLine: 0,
        endLine: 0,
        source: "global",
        snippet: "legacy shared memory",
        content: "legacy shared memory from before tenant isolation",
      },
      "legacy shared memory from before tenant isolation",
    );

    // Scoped entry for instance-a
    await sm.addEntry(
      {
        id: "scoped-1",
        path: "session:default",
        startLine: 0,
        endLine: 0,
        source: "realtime-user",
        snippet: "instance-a specific memory",
        content: "instance-a specific memory about project plans",
        instanceId: "instance-a",
      },
      "instance-a specific memory about project plans",
    );

    return sm;
  }

  it("default (false): legacy entries visible to tenant-scoped queries", async () => {
    const sm = await seedLegacyAndScoped(baseConfig);
    const results = await sm.search("memory", 10, "instance-a");
    const hasLegacy = results.some((r) => r.snippet?.includes("legacy"));
    expect(hasLegacy).toBe(true);
    await sm.close();
  });

  it("excludeLegacyEntries=true: legacy entries hidden from tenant-scoped queries", async () => {
    const config: SemanticMemoryConfig = {
      ...baseConfig,
      search: { ...baseConfig.search, excludeLegacyEntries: true },
    };
    const sm = await seedLegacyAndScoped(config);
    const results = await sm.search("memory", 10, "instance-a");
    const hasLegacy = results.some((r) => r.snippet?.includes("legacy"));
    expect(hasLegacy).toBe(false);
    // Own entries still visible
    const hasOwn = results.some((r) => r.snippet?.includes("instance-a"));
    expect(hasOwn).toBe(true);
    await sm.close();
  });

  it("excludeLegacyEntries=true: no-instanceId queries still return everything", async () => {
    const config: SemanticMemoryConfig = {
      ...baseConfig,
      search: { ...baseConfig.search, excludeLegacyEntries: true },
    };
    const sm = await seedLegacyAndScoped(config);
    // No instanceId filter -> returns everything regardless of flag
    const results = await sm.search("memory", 10);
    const hasLegacy = results.some((r) => r.snippet?.includes("legacy"));
    expect(hasLegacy).toBe(true);
    await sm.close();
  });
});
