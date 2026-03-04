import { describe, expect, it, vi } from "vitest";

describe("search.ts dims-read catch logging", () => {
  it("logs debug when saved HNSW map is corrupt JSON", async () => {
    const { mkdirSync, writeFileSync, unlinkSync } = await import("node:fs");
    const { join } = await import("node:path");
    const os = await import("node:os");
    const { fallbackLogger } = await import("../../src/fallback-logger.js");

    const tmpDir = join(os.tmpdir(), `wop-1562-test-${Date.now()}`);
    mkdirSync(tmpDir, { recursive: true });
    const hnswPath = join(tmpDir, "test.hnsw");
    const mapPath = `${hnswPath}.map.json`;

    // Write corrupt JSON so the dims-read catch fires
    writeFileSync(mapPath, "NOT VALID JSON");

    const debugSpy = vi.spyOn(fallbackLogger, "debug");

    try {
      const { createSemanticSearchManager } = await import("../../src/search.js");
      await createSemanticSearchManager(
        {
          search: { maxResults: 10, minScore: 0.3, candidateMultiplier: 3, excludeLegacyEntries: false },
          hybrid: { enabled: false, vectorWeight: 0.7, textWeight: 0.3 },
          cache: { enabled: false },
        } as any,
        { embedQuery: vi.fn().mockRejectedValue(new Error("no provider")), embedBatch: vi.fn(), id: "test" } as any,
        undefined,
        hnswPath,
      ).catch(() => {
        // Expected: provider fails and no saved dims
      });

      const debugCalls = debugSpy.mock.calls.map((c) => c[0]);
      const matchingCall = debugCalls.find((msg) => typeof msg === "string" && msg.includes("Failed to read saved HNSW map for dims"));
      expect(matchingCall).toBeDefined();
    } finally {
      debugSpy.mockRestore();
      try { unlinkSync(mapPath); } catch { /* non-fatal: cleanup */ }
      try { unlinkSync(hnswPath); } catch { /* non-fatal: cleanup */ }
      try { (await import("node:fs/promises")).rmdir(tmpDir); } catch { /* non-fatal: cleanup */ }
    }
  });
});
