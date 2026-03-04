import { describe, it, expect, vi } from "vitest";
import type { StorageApi } from "@wopr-network/plugin-types";
import {
  createMemoryPluginSchema,
  memoryPluginSchema,
  MEMORY_NAMESPACE,
  MEMORY_SCHEMA_VERSION,
} from "../../src/memory-schema.js";

// Helper: create a mock StorageApi with a recording raw() method
function createMockStorage(): StorageApi & { rawCalls: Array<[string, unknown[]?]> } {
  const rawCalls: Array<[string, unknown[]?]> = [];
  const mockStorage = {
    rawCalls,
    raw: vi.fn(async (sql: string, params?: unknown[]) => {
      rawCalls.push([sql, params]);
      return [];
    }),
    get: vi.fn(),
    put: vi.fn(),
    delete: vi.fn(),
    list: vi.fn(),
    count: vi.fn(),
    query: vi.fn(),
  } as unknown as StorageApi & { rawCalls: Array<[string, unknown[]?]> };
  return mockStorage;
}

describe("memory-schema", () => {
  describe("constants", () => {
    it("exports correct namespace", () => {
      expect(MEMORY_NAMESPACE).toBe("memory");
    });

    it("exports correct schema version", () => {
      expect(MEMORY_SCHEMA_VERSION).toBe(2);
    });
  });

  describe("createMemoryPluginSchema", () => {
    it("returns schema with correct namespace and version", () => {
      const schema = createMemoryPluginSchema(undefined);
      expect(schema.namespace).toBe("memory");
      expect(schema.version).toBe(2);
    });

    it("returns schema with files, chunks, and meta tables", () => {
      const schema = createMemoryPluginSchema(undefined);
      expect(Object.keys(schema.tables)).toEqual(["files", "chunks", "meta"]);
    });

    it("has migrate function", () => {
      const schema = createMemoryPluginSchema(undefined);
      expect(typeof schema.migrate).toBe("function");
    });
  });

  describe("default memoryPluginSchema export", () => {
    it("has correct namespace and version", () => {
      expect(memoryPluginSchema.namespace).toBe("memory");
      expect(memoryPluginSchema.version).toBe(2);
    });
  });

  describe("files table schema validation", () => {
    const schema = createMemoryPluginSchema(undefined);
    const filesSchema = schema.tables.files.schema;

    it("accepts valid file data", () => {
      const valid = {
        id: "abc123",
        path: "/src/index.ts",
        source: "local",
        hash: "sha256abc",
        mtime: 1700000000,
        size: 1024,
      };
      expect(() => filesSchema.parse(valid)).not.toThrow();
    });

    it("rejects missing required fields", () => {
      expect(() => filesSchema.parse({})).toThrow();
      expect(() => filesSchema.parse({ id: "x" })).toThrow();
    });

    it("rejects non-integer mtime", () => {
      const invalid = {
        id: "abc",
        path: "/x",
        source: "s",
        hash: "h",
        mtime: 1.5,
        size: 10,
      };
      expect(() => filesSchema.parse(invalid)).toThrow();
    });

    it("rejects non-integer size", () => {
      const invalid = {
        id: "abc",
        path: "/x",
        source: "s",
        hash: "h",
        mtime: 10,
        size: 1.5,
      };
      expect(() => filesSchema.parse(invalid)).toThrow();
    });

    it("rejects wrong types", () => {
      expect(() =>
        filesSchema.parse({ id: 123, path: true, source: null, hash: 0, mtime: "x", size: "y" }),
      ).toThrow();
    });
  });

  describe("chunks table schema validation", () => {
    const schema = createMemoryPluginSchema(undefined);
    const chunksSchema = schema.tables.chunks.schema;

    const validChunk = {
      id: "chunk1",
      path: "/src/file.ts",
      source: "local",
      start_line: 1,
      end_line: 10,
      hash: "sha256",
      model: "text-embedding-3-small",
      text: "function hello() {}",
      updated_at: 1700000000,
    };

    it("accepts valid chunk without instance_id", () => {
      expect(() => chunksSchema.parse(validChunk)).not.toThrow();
    });

    it("accepts valid chunk with instance_id", () => {
      expect(() => chunksSchema.parse({ ...validChunk, instance_id: "inst-1" })).not.toThrow();
    });

    it("rejects missing required fields", () => {
      expect(() => chunksSchema.parse({})).toThrow();
    });

    it("rejects non-integer start_line", () => {
      expect(() => chunksSchema.parse({ ...validChunk, start_line: 1.5 })).toThrow();
    });

    it("rejects non-integer end_line", () => {
      expect(() => chunksSchema.parse({ ...validChunk, end_line: 1.5 })).toThrow();
    });

    it("rejects non-integer updated_at", () => {
      expect(() => chunksSchema.parse({ ...validChunk, updated_at: 1.5 })).toThrow();
    });
  });

  describe("meta table schema validation", () => {
    const schema = createMemoryPluginSchema(undefined);
    const metaSchema = schema.tables.meta.schema;

    it("accepts valid meta entry", () => {
      expect(() => metaSchema.parse({ key: "version", value: "2" })).not.toThrow();
    });

    it("rejects missing key", () => {
      expect(() => metaSchema.parse({ value: "2" })).toThrow();
    });

    it("rejects missing value", () => {
      expect(() => metaSchema.parse({ key: "version" })).toThrow();
    });

    it("rejects non-string key", () => {
      expect(() => metaSchema.parse({ key: 123, value: "x" })).toThrow();
    });

    it("rejects non-string value", () => {
      expect(() => metaSchema.parse({ key: "k", value: 123 })).toThrow();
    });
  });

  describe("table indexes", () => {
    const schema = createMemoryPluginSchema(undefined);

    it("files has correct primary key and indexes", () => {
      expect(schema.tables.files.primaryKey).toBe("id");
      expect(schema.tables.files.indexes).toEqual([
        { fields: ["path", "source"], unique: true },
        { fields: ["source"] },
      ]);
    });

    it("chunks has correct primary key and indexes", () => {
      expect(schema.tables.chunks.primaryKey).toBe("id");
      expect(schema.tables.chunks.indexes).toEqual([
        { fields: ["path"] },
        { fields: ["source"] },
        { fields: ["instance_id"] },
      ]);
    });

    it("meta has correct primary key and no indexes", () => {
      expect(schema.tables.meta.primaryKey).toBe("key");
      expect(schema.tables.meta.indexes).toBeUndefined();
    });
  });

  describe("v1 to v2 migration", () => {
    it("adds instance_id column and index when migrating from v1 to v2", async () => {
      const storage = createMockStorage();
      const schema = createMemoryPluginSchema("my-instance");

      await schema.migrate!(1, 2, storage);

      const sqls = storage.rawCalls.map(([sql]) => sql);
      expect(sqls).toContainEqual(expect.stringContaining("ALTER TABLE memory_chunks ADD COLUMN instance_id TEXT"));
      expect(sqls).toContainEqual(expect.stringContaining("CREATE INDEX IF NOT EXISTS idx_memory_chunks_instance_id"));
      expect(sqls).toContainEqual(expect.stringContaining("UPDATE memory_chunks SET instance_id = ?"));
      const updateCall = storage.rawCalls.find(([sql]) => sql.includes("UPDATE memory_chunks SET instance_id"));
      expect(updateCall?.[1]).toEqual(["my-instance"]);
    });

    it("skips instance_id UPDATE when resolvedInstanceId is undefined", async () => {
      const storage = createMockStorage();
      const schema = createMemoryPluginSchema(undefined);

      await schema.migrate!(1, 2, storage);

      const sqls = storage.rawCalls.map(([sql]) => sql);
      expect(sqls).toContainEqual(expect.stringContaining("ALTER TABLE"));
      expect(sqls).toContainEqual(expect.stringContaining("CREATE INDEX"));
      expect(sqls).not.toContainEqual(expect.stringContaining("UPDATE memory_chunks SET instance_id"));
    });

    it("runs v0-to-v1 and v1-to-v2 when migrating from v0 to v2", async () => {
      const originalWoprHome = process.env.WOPR_HOME;
      delete process.env.WOPR_HOME;

      const storage = createMockStorage();
      const schema = createMemoryPluginSchema("inst-1");

      await schema.migrate!(0, 2, storage);

      const sqls = storage.rawCalls.map(([sql]) => sql);
      expect(sqls).toContainEqual(expect.stringContaining("ALTER TABLE memory_chunks ADD COLUMN instance_id"));
      expect(sqls).toContainEqual(expect.stringContaining("UPDATE memory_chunks SET instance_id"));

      if (originalWoprHome !== undefined) process.env.WOPR_HOME = originalWoprHome;
    });

    it("does not run v1-to-v2 migration when fromVersion >= 2", async () => {
      const storage = createMockStorage();
      const schema = createMemoryPluginSchema("inst-1");

      await schema.migrate!(2, 3, storage);

      expect(storage.rawCalls).toHaveLength(0);
    });

    it("handles ALTER TABLE failure gracefully (column already exists)", async () => {
      const storage = createMockStorage();
      (storage.raw as ReturnType<typeof vi.fn>).mockImplementation(async (sql: string, params?: unknown[]) => {
        storage.rawCalls.push([sql, params]);
        if (sql.includes("ALTER TABLE")) throw new Error("duplicate column name: instance_id");
        return [];
      });
      const schema = createMemoryPluginSchema("inst-1");

      await expect(schema.migrate!(1, 2, storage)).resolves.not.toThrow();

      const sqls = storage.rawCalls.map(([sql]) => sql);
      expect(sqls).toContainEqual(expect.stringContaining("UPDATE memory_chunks SET instance_id"));
    });
  });

  describe("migration idempotency", () => {
    it("running v1-to-v2 migration twice does not throw", async () => {
      const storage = createMockStorage();
      let alterCount = 0;
      (storage.raw as ReturnType<typeof vi.fn>).mockImplementation(async (sql: string, params?: unknown[]) => {
        storage.rawCalls.push([sql, params]);
        if (sql.includes("ALTER TABLE")) {
          alterCount++;
          if (alterCount > 1) throw new Error("duplicate column name: instance_id");
        }
        return [];
      });

      const schema = createMemoryPluginSchema("inst-1");

      await schema.migrate!(1, 2, storage);
      storage.rawCalls.length = 0;

      await expect(schema.migrate!(1, 2, storage)).resolves.not.toThrow();
    });

    it("UPDATE is idempotent — sets instance_id on already-tagged rows", async () => {
      const storage = createMockStorage();
      const schema = createMemoryPluginSchema("inst-1");

      await schema.migrate!(1, 2, storage);
      await schema.migrate!(1, 2, storage);

      const updateCalls = storage.rawCalls.filter(([sql]) => sql.includes("UPDATE memory_chunks"));
      expect(updateCalls).toHaveLength(2);
      for (const [, params] of updateCalls) {
        expect(params).toEqual(["inst-1"]);
      }
    });
  });
});
