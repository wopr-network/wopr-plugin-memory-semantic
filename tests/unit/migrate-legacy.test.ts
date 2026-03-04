import { describe, it, expect, vi } from "vitest";

describe("migrateFromLegacyIndexSqlite", () => {
  it("should not call DETACH twice when renameSync throws", async () => {
    // Mock fs module
    const mockRenameSync = vi.fn().mockImplementation(() => {
      throw new Error("EPERM: operation not permitted");
    });
    const mockExistsSync = vi.fn().mockReturnValue(true);

    vi.doMock("node:fs", () => ({
      existsSync: mockExistsSync,
      renameSync: mockRenameSync,
    }));
    vi.doMock("node:path", () => ({
      join: (...parts: string[]) => parts.join("/"),
    }));

    // Set WOPR_HOME so migration runs
    const origHome = process.env.WOPR_HOME;
    process.env.WOPR_HOME = "/tmp/test-wopr-home";
    try {
      const rawCalls: string[] = [];
      const mockStorage = {
        raw: vi.fn().mockImplementation((sql: string) => {
          rawCalls.push(sql.trim());
          return Promise.resolve();
        }),
      };

      // Import fresh to pick up mocks
      const { createMemoryPluginSchema } = await import("../../src/memory-schema.js");
      const schema = createMemoryPluginSchema(undefined);

      // v0 → v1 triggers migrateFromLegacyIndexSqlite
      // renameSync will throw, but DETACH should only be called once (not in catch)
      let caughtError: Error | undefined;
      try {
        await schema.migrate(0, 1, mockStorage as any);
      } catch (err: any) {
        caughtError = err;
      }

      // renameSync error should propagate
      expect(caughtError?.message).toBe("EPERM: operation not permitted");

      const detachCalls = rawCalls.filter((sql) => sql.startsWith("DETACH"));
      expect(detachCalls).toHaveLength(1); // Only one DETACH, not two
    } finally {
      if (origHome === undefined) {
        delete process.env.WOPR_HOME;
      } else {
        process.env.WOPR_HOME = origHome;
      }
      vi.restoreAllMocks();
    }
  });
});
