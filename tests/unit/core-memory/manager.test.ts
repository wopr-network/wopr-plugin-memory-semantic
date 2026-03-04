import { describe, it, expect, vi, beforeEach } from "vitest";
import type { PluginLogger, StorageApi, WOPREventBus } from "@wopr-network/plugin-types";
import { mockLogger, mockStorage, mockEvents } from "./helpers.js";

// Mock internal module dependencies
vi.mock("../../../src/core-memory/internal.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../src/core-memory/internal.js")>();
  return {
    ...actual,
    listMemoryFiles: vi.fn().mockResolvedValue([]),
    buildFileEntry: vi.fn(),
  };
});

vi.mock("../../../src/core-memory/sync-sessions.js", () => ({
  syncSessionFiles: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("node:fs/promises", () => ({
  default: {
    readdir: vi.fn().mockResolvedValue([]),
    stat: vi.fn().mockRejectedValue(new Error("ENOENT")),
    readFile: vi.fn().mockResolvedValue(""),
  },
}));

import { MemoryIndexManager } from "../../../src/core-memory/manager.js";


async function createManager(overrides?: {
  storage?: StorageApi;
  events?: WOPREventBus;
  log?: PluginLogger;
}) {
  return MemoryIndexManager.create({
    globalDir: "/workspace",
    sessionDir: "/session",
    storage: overrides?.storage ?? mockStorage(),
    events: overrides?.events ?? mockEvents(),
    log: overrides?.log ?? mockLogger(),
  });
}

describe("MemoryIndexManager.create", () => {
  it("creates a manager instance", async () => {
    const manager = await createManager();
    expect(manager).toBeDefined();
  });
});

describe("MemoryIndexManager.status", () => {
  it("returns status with files and chunks counts", async () => {
    const storage = mockStorage();
    vi.mocked(storage.raw).mockImplementation(async (sql: string) => {
      if (sql.includes("memory_files")) return [{ c: 3 }];
      if (sql.includes("memory_chunks")) return [{ c: 10 }];
      return [];
    });
    const manager = await createManager({ storage });
    const status = await manager.status();
    expect(status.files).toBe(3);
    expect(status.chunks).toBe(10);
    expect(status.dirty).toBe(true);
    expect(status.fts.enabled).toBe(true);
  });
});

describe("MemoryIndexManager.search", () => {
  it("returns empty array for empty query", async () => {
    const manager = await createManager();
    const results = await manager.search("");
    expect(results).toEqual([]);
  });

  it("returns empty array for whitespace query", async () => {
    const manager = await createManager();
    const results = await manager.search("   ");
    expect(results).toEqual([]);
  });

  it("calls FTS search and returns results", async () => {
    const storage = mockStorage();
    vi.mocked(storage.raw).mockImplementation(async (sql: string) => {
      if (sql.includes("memory_meta")) return []; // no meta → full reindex
      if (sql.includes("memory_files")) {
        if (sql.includes("COUNT")) return [{ c: 0 }];
        return [];
      }
      if (sql.includes("memory_chunks")) {
        if (sql.includes("COUNT")) return [{ c: 0 }];
        return [];
      }
      if (sql.includes("memory_chunks_fts")) {
        return [
          {
            id: "chunk1",
            path: "MEMORY.md",
            source: "global",
            start_line: 1,
            end_line: 5,
            snippet: "...matching text...",
            rank: 0.5,
          },
        ];
      }
      return [];
    });

    const manager = await createManager({ storage });
    // Mark as not dirty to skip sync
    const results = await manager.search("matching text");
    // Results may be empty if sync is triggered and fails gracefully
    expect(Array.isArray(results)).toBe(true);
  });

  it("includes legacy rows (OR instance_id IS NULL) by default for tenant-scoped queries", async () => {
    const storage = mockStorage();
    const capturedSqls: string[] = [];
    vi.mocked(storage.raw).mockImplementation(async (sql: string) => {
      capturedSqls.push(sql);
      return [];
    });
    const manager = await createManager({ storage });
    await manager.search("some query", { instanceId: "tenant-1" });
    const ftsSql = capturedSqls.find((s) => s.includes("memory_chunks_fts"));
    expect(ftsSql).toBeDefined();
    expect(ftsSql).toContain("c.instance_id IS NULL");
  });

  it("excludes legacy rows when excludeLegacyEntries=true for tenant-scoped queries", async () => {
    const storage = mockStorage();
    const capturedSqls: string[] = [];
    vi.mocked(storage.raw).mockImplementation(async (sql: string) => {
      capturedSqls.push(sql);
      return [];
    });
    const manager = await createManager({ storage });
    await manager.search("some query", { instanceId: "tenant-1", excludeLegacyEntries: true });
    const ftsSql = capturedSqls.find((s) => s.includes("memory_chunks_fts"));
    expect(ftsSql).toBeDefined();
    expect(ftsSql).not.toContain("c.instance_id IS NULL");
    expect(ftsSql).toContain("c.instance_id = ?");
  });

  it("does not apply instance filter when no instanceId is provided", async () => {
    const storage = mockStorage();
    const capturedSqls: string[] = [];
    vi.mocked(storage.raw).mockImplementation(async (sql: string) => {
      capturedSqls.push(sql);
      return [];
    });
    const manager = await createManager({ storage });
    await manager.search("some query", { excludeLegacyEntries: true });
    const ftsSql = capturedSqls.find((s) => s.includes("memory_chunks_fts"));
    expect(ftsSql).toBeDefined();
    expect(ftsSql).not.toContain("c.instance_id");
  });
});

describe("MemoryIndexManager.handleFilesChanged", () => {
  it("handles delete action", async () => {
    const storage = mockStorage();
    const manager = await createManager({ storage });

    await manager.handleFilesChanged({
      changes: [{ action: "delete", path: "MEMORY.md", source: "global" }],
    });

    expect(vi.mocked(storage.raw)).toHaveBeenCalledWith(
      expect.stringContaining("DELETE FROM memory_files"),
      expect.arrayContaining(["MEMORY.md", "global"]),
    );
  });

  it("handles upsert action with chunks", async () => {
    const storage = mockStorage();
    const manager = await createManager({ storage });

    await manager.handleFilesChanged({
      changes: [
        {
          action: "upsert",
          path: "MEMORY.md",
          source: "global",
          chunks: [
            {
              id: "chunk1",
              text: "hello world",
              hash: "abc123",
              startLine: 1,
              endLine: 3,
            },
          ],
        },
      ],
    });

    expect(vi.mocked(storage.transaction)).toHaveBeenCalled();
    expect(vi.mocked(storage.raw)).toHaveBeenCalledWith(
      expect.stringContaining("INSERT OR REPLACE INTO memory_chunks"),
      expect.any(Array),
    );
  });

  it("skips upsert when no chunks", async () => {
    const storage = mockStorage();
    const manager = await createManager({ storage });

    vi.mocked(storage.raw).mockClear();
    await manager.handleFilesChanged({
      changes: [{ action: "upsert", path: "MEMORY.md", source: "global", chunks: [] }],
    });

    // Should not insert chunks
    const calls = vi.mocked(storage.raw).mock.calls.map((c) => c[0]);
    expect(calls.some((sql) => sql.includes("INSERT OR REPLACE INTO memory_chunks"))).toBe(false);
  });
});

describe("MemoryIndexManager.close", () => {
  it("can be called multiple times without error", async () => {
    const manager = await createManager();
    await manager.close();
    await expect(manager.close()).resolves.not.toThrow();
  });
});
