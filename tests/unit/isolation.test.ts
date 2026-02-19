/**
 * Cross-tenant isolation tests (WOP-624)
 *
 * Tests that HNSW vector search properly isolates memories by instanceId,
 * preventing cross-bot memory leakage in multi-tenant deployments.
 */
import { mkdirSync, rmSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createSemanticSearchManager } from "../../src/search.js";
import { DEFAULT_CONFIG, type SemanticMemoryConfig } from "../../src/types.js";

// Deterministic mock embedding provider: same text -> same vector, different text -> different vector
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

const baseConfig: SemanticMemoryConfig = {
  ...DEFAULT_CONFIG,
  search: { maxResults: 10, minScore: 0.0, candidateMultiplier: 4 },
  hybrid: { enabled: false, vectorWeight: 0.7, textWeight: 0.3 },
};

// =============================================================================
// Cross-tenant isolation
// =============================================================================

describe("cross-tenant isolation", () => {
  it("search with instanceId should NOT return entries from other instances", async () => {
    const provider = createMockEmbeddingProvider();
    const sm = await createSemanticSearchManager(baseConfig, provider);

    // Instance A stores a memory
    await sm.addEntry(
      {
        id: "entry-a",
        path: "session:default",
        startLine: 0,
        endLine: 0,
        source: "realtime-user",
        snippet: "Customer A secret business plan",
        content: "Customer A secret business plan for Q3 expansion",
        instanceId: "instance-a",
      },
      "Customer A secret business plan for Q3 expansion",
    );

    // Instance B stores a memory
    await sm.addEntry(
      {
        id: "entry-b",
        path: "session:default",
        startLine: 0,
        endLine: 0,
        source: "realtime-user",
        snippet: "Customer B confidential data",
        content: "Customer B confidential financial data and projections",
        instanceId: "instance-b",
      },
      "Customer B confidential financial data and projections",
    );

    // Instance A searches — should only see its own data
    const resultsA = await sm.search("business plan expansion", 10, "instance-a");
    const leakedB = resultsA.find((r) => r.snippet?.includes("Customer B"));
    expect(leakedB).toBeUndefined();

    // Instance B searches — should only see its own data
    const resultsB = await sm.search("financial data projections", 10, "instance-b");
    const leakedA = resultsB.find((r) => r.snippet?.includes("Customer A"));
    expect(leakedA).toBeUndefined();

    await sm.close();
  });

  it("search without instanceId should return all entries (backward compat)", async () => {
    const provider = createMockEmbeddingProvider();
    const sm = await createSemanticSearchManager(baseConfig, provider);

    await sm.addEntry(
      {
        id: "entry-global",
        path: "global/MEMORY.md",
        startLine: 0,
        endLine: 0,
        source: "global",
        snippet: "shared knowledge base",
        content: "shared knowledge base for all instances",
      },
      "shared knowledge base for all instances",
    );

    // No instanceId filter -> returns everything
    const results = await sm.search("knowledge base", 10);
    expect(results.length).toBeGreaterThan(0);

    await sm.close();
  });

  it("entries without instanceId (legacy) are visible to all instances", async () => {
    const provider = createMockEmbeddingProvider();
    const sm = await createSemanticSearchManager(baseConfig, provider);

    // Legacy entry without instanceId (pre-migration)
    await sm.addEntry(
      {
        id: "legacy-entry",
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
        id: "scoped-entry",
        path: "session:default",
        startLine: 0,
        endLine: 0,
        source: "realtime-user",
        snippet: "instance-specific memory",
        content: "instance-specific memory for instance-a only",
        instanceId: "instance-a",
      },
      "instance-specific memory for instance-a only",
    );

    // Instance A should see both legacy (no instanceId) and its own
    const resultsA = await sm.search("memory", 10, "instance-a");
    const hasLegacy = resultsA.some((r) => r.snippet?.includes("legacy"));
    const hasOwn = resultsA.some((r) => r.snippet?.includes("instance-specific"));
    expect(hasLegacy).toBe(true);
    expect(hasOwn).toBe(true);

    // Instance B should see legacy but NOT instance-a's entry
    const resultsB = await sm.search("memory", 10, "instance-b");
    const hasLegacyForB = resultsB.some((r) => r.snippet?.includes("legacy"));
    const hasInstanceAForB = resultsB.some((r) => r.snippet?.includes("instance-specific"));
    expect(hasLegacyForB).toBe(true);
    expect(hasInstanceAForB).toBe(false);

    await sm.close();
  });

  it("HNSW save/load preserves instanceId across restarts", async () => {
    const provider = createMockEmbeddingProvider();
    const tmpDir = `/tmp/wopr-test-isolation-${Date.now()}`;
    mkdirSync(tmpDir, { recursive: true });
    const hnswPath = `${tmpDir}/test.hnsw`;

    try {
      // Create index, add entry, save
      const sm1 = await createSemanticSearchManager(baseConfig, provider, undefined, hnswPath);
      await sm1.addEntry(
        {
          id: "persisted-entry",
          path: "session:default",
          startLine: 0,
          endLine: 0,
          source: "realtime-user",
          snippet: "persisted tenant data",
          content: "persisted tenant data for persistence test",
          instanceId: "instance-persist",
        },
        "persisted tenant data for persistence test",
      );
      await sm1.close(); // triggers save

      // Reload and verify instanceId survived serialization
      const sm2 = await createSemanticSearchManager(baseConfig, provider, undefined, hnswPath);
      const entry = sm2.getEntry("persisted-entry");
      expect(entry?.instanceId).toBe("instance-persist");

      // Verify search isolation works after reload
      const results = await sm2.search("persisted tenant", 10, "other-instance");
      const leaked = results.find((r) => r.snippet?.includes("persisted tenant"));
      expect(leaked).toBeUndefined();

      // The instance that owns it can still find it
      const ownResults = await sm2.search("persisted tenant", 10, "instance-persist");
      const found = ownResults.find((r) => r.snippet?.includes("persisted tenant"));
      expect(found).toBeDefined();

      await sm2.close();
    } finally {
      rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  it("instance with many entries does not starve another instance's results", async () => {
    const provider = createMockEmbeddingProvider();
    const sm = await createSemanticSearchManager(baseConfig, provider);

    // Instance A floods with 20 similar entries
    for (let i = 0; i < 20; i++) {
      await sm.addEntry(
        {
          id: `flood-${i}`,
          path: "session:default",
          startLine: i,
          endLine: i + 1,
          source: "realtime-user",
          snippet: `customer data entry ${i}`,
          content: `customer data entry ${i} for instance a quarterly report business plan`,
          instanceId: "instance-a",
        },
        `customer data entry ${i} for instance a quarterly report business plan`,
      );
    }

    // Instance B has one entry
    await sm.addEntry(
      {
        id: "instance-b-sole",
        path: "session:default",
        startLine: 0,
        endLine: 1,
        source: "realtime-user",
        snippet: "instance b sole entry",
        content: "instance b sole entry quarterly report business plan",
        instanceId: "instance-b",
      },
      "instance b sole entry quarterly report business plan",
    );

    // Instance B should still find its own entry despite being outnumbered
    const resultsB = await sm.search("quarterly report business plan", 5, "instance-b");
    const found = resultsB.find((r) => r.snippet?.includes("instance b sole entry"));
    expect(found).toBeDefined();

    // And no flooding from instance A
    const leaked = resultsB.find((r) => r.snippet?.includes("customer data entry"));
    expect(leaked).toBeUndefined();

    await sm.close();
  });
});
