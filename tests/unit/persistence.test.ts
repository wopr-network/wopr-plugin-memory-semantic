import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { tmpdir } from "node:os";
import { mkdtempSync, mkdirSync, rmSync } from "node:fs";
import { join } from "node:path";
import {
  getDb,
  ensureEmbeddingColumn,
  openDbForRead,
  loadChunkMetadata,
  getHnswPath,
  persistNewEntryToDb,
} from "../../src/persistence.js";

// Guard: node:sqlite is only available on Node >= 22.5.0.
// On older versions the entire suite is skipped rather than aborting the run,
// mirroring the guarded dynamic import used in production code.
let DatabaseSync: any;
try {
  ({ DatabaseSync } = await import("node:sqlite"));
} catch {
  /* node:sqlite unavailable — SQLite-dependent tests will be skipped */
}
const hasSqlite = DatabaseSync !== undefined;

// ─── helpers ──────────────────────────────────────────────────────────────────

function makeLog() {
  return { info: vi.fn(), warn: vi.fn(), error: vi.fn() };
}

// Track created in-memory DB instances so we can close them after each test.
const openDbs: any[] = [];

function makeDb(opts?: { readOnly?: boolean }): any {
  const db = opts ? new DatabaseSync(":memory:", opts) : new DatabaseSync(":memory:");
  openDbs.push(db);
  return db;
}

afterEach(() => {
  for (const db of openDbs.splice(0)) {
    try {
      db.close();
    } catch {
      // ignore – db may already be closed
    }
  }
});

function createChunksTable(db: any) {
  db.exec(`
    CREATE TABLE IF NOT EXISTS chunks (
      id TEXT PRIMARY KEY,
      path TEXT NOT NULL,
      source TEXT NOT NULL,
      start_line INTEGER NOT NULL,
      end_line INTEGER NOT NULL,
      hash TEXT NOT NULL,
      model TEXT,
      text TEXT,
      updated_at INTEGER,
      embedding BLOB,
      instance_id TEXT
    )
  `);
}

/** Create a temp directory containing a valid SQLite DB at <dir>/memory/index.sqlite.
 *  Caller is responsible for cleanup via rmSync. */
function createTempSqliteDir(): { tempDir: string } {
  const tempDir = mkdtempSync(join(tmpdir(), "wopr-persistence-test-"));
  const memDir = join(tempDir, "memory");
  mkdirSync(memDir);
  const db = new DatabaseSync(join(memDir, "index.sqlite"));
  createChunksTable(db);
  db.close();
  return { tempDir };
}

// ─── getDb ───────────────────────────────────────────────────────────────────

describe.skipIf(!hasSqlite)("getDb", () => {
  it("returns db from extension", () => {
    const db = makeDb();
    const api = { getExtension: () => db };
    expect(getDb(api)).toBe(db);
  });

  it("returns null when extension absent", () => {
    const api = { getExtension: () => undefined };
    expect(getDb(api)).toBeNull();
  });
});

// ─── ensureEmbeddingColumn ────────────────────────────────────────────────────

describe.skipIf(!hasSqlite)("ensureEmbeddingColumn", () => {
  it("adds embedding column when missing", () => {
    const db = makeDb();
    db.exec(`CREATE TABLE chunks (id TEXT PRIMARY KEY, path TEXT)`);
    const log = makeLog();
    ensureEmbeddingColumn(db, log);
    const cols = db.prepare("PRAGMA table_info(chunks)").all() as Array<{ name: string }>;
    expect(cols.some((c) => c.name === "embedding")).toBe(true);
    expect(log.warn).not.toHaveBeenCalled();
  });

  it("is idempotent when column already exists", () => {
    const db = makeDb();
    db.exec(`CREATE TABLE chunks (id TEXT PRIMARY KEY, embedding BLOB)`);
    const log = makeLog();
    ensureEmbeddingColumn(db, log);
    ensureEmbeddingColumn(db, log);
    expect(log.warn).not.toHaveBeenCalled();
  });

  it("returns early when db is null", () => {
    const log = makeLog();
    expect(() => ensureEmbeddingColumn(null, log)).not.toThrow();
  });

  it("returns early when db has no prepare method", () => {
    const log = makeLog();
    expect(() => ensureEmbeddingColumn({} as any, log)).not.toThrow();
  });

  it("returns early when db has no exec method", () => {
    const log = makeLog();
    expect(() =>
      ensureEmbeddingColumn({ prepare: vi.fn() } as any, log),
    ).not.toThrow();
  });

  it("logs warning on SQL error", () => {
    const brokenDb = {
      prepare: () => ({ all: () => { throw new Error("boom"); } }),
      exec: vi.fn(),
    };
    const log = makeLog();
    ensureEmbeddingColumn(brokenDb as any, log);
    expect(log.warn).toHaveBeenCalledWith(expect.stringContaining("boom"));
  });
});

// ─── openDbForRead ────────────────────────────────────────────────────────────

describe.skipIf(!hasSqlite)("openDbForRead", () => {
  let origHome: string | undefined;

  beforeEach(() => {
    origHome = process.env.WOPR_HOME;
  });

  afterEach(() => {
    if (origHome === undefined) delete process.env.WOPR_HOME;
    else process.env.WOPR_HOME = origHome;
  });

  it("returns core db when extension available (owned=false)", async () => {
    const db = makeDb();
    const api = { getExtension: () => db };
    const log = makeLog();
    const result = await openDbForRead(api as any, log);
    expect(result).not.toBeNull();
    expect(result!.db).toBe(db);
    expect(result!.owned).toBe(false);
  });

  it("returns null when no extension and no WOPR_HOME", async () => {
    delete process.env.WOPR_HOME;
    const api = { getExtension: () => undefined };
    const log = makeLog();
    const result = await openDbForRead(api as any, log);
    expect(result).toBeNull();
  });

  it("returns null and warns when WOPR_HOME path does not exist", async () => {
    // Use a guaranteed-nonexistent sub-path of a fresh temp dir to avoid flakiness
    const tempBase = mkdtempSync(join(tmpdir(), "wopr-test-base-"));
    process.env.WOPR_HOME = join(tempBase, "definitely-missing");
    try {
      const api = { getExtension: () => undefined };
      const log = makeLog();
      const result = await openDbForRead(api as any, log);
      expect(result).toBeNull();
      expect(log.warn).toHaveBeenCalledWith(expect.stringContaining("Cannot open DB directly"));
    } finally {
      rmSync(tempBase, { recursive: true, force: true });
    }
  });

  it("opens a file-based DB directly and returns owned=true", async () => {
    const { tempDir } = createTempSqliteDir();
    process.env.WOPR_HOME = tempDir;
    try {
      const api = { getExtension: () => undefined };
      const log = makeLog();
      const result = await openDbForRead(api as any, log);
      expect(result).not.toBeNull();
      expect(result!.owned).toBe(true);
      expect(result!.db).toBeDefined();
      result!.db.close(); // close the owned handle we received
      expect(log.info).toHaveBeenCalledWith(expect.stringContaining("Opened direct read-only DB"));
    } finally {
      rmSync(tempDir, { recursive: true, force: true });
    }
  });
});

// ─── loadChunkMetadata ────────────────────────────────────────────────────────

describe.skipIf(!hasSqlite)("loadChunkMetadata", () => {
  let origHome: string | undefined;

  beforeEach(() => {
    origHome = process.env.WOPR_HOME;
  });

  afterEach(() => {
    if (origHome === undefined) delete process.env.WOPR_HOME;
    else process.env.WOPR_HOME = origHome;
  });

  it("returns empty map when no db available", async () => {
    delete process.env.WOPR_HOME;
    const api = { getExtension: () => undefined };
    const log = makeLog();
    const result = await loadChunkMetadata(api as any, log);
    expect(result.size).toBe(0);
  });

  it("loads rows and returns empty embedding placeholder", async () => {
    const db = makeDb();
    createChunksTable(db);
    db.prepare(
      `INSERT INTO chunks (id, path, source, start_line, end_line, hash, text, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
    ).run("id1", "/file.ts", "file", 1, 10, "abc", "hello world content", Date.now());

    const api = { getExtension: () => db };
    const log = makeLog();
    const result = await loadChunkMetadata(api as any, log);
    expect(result.size).toBe(1);
    const entry = result.get("id1")!;
    expect(entry.id).toBe("id1");
    expect(entry.path).toBe("/file.ts");
    expect(entry.startLine).toBe(1);
    expect(entry.endLine).toBe(10);
    expect(entry.content).toBe("hello world content");
    expect(entry.embedding).toEqual([]); // placeholder — HNSW is source of truth
    expect(log.info).toHaveBeenCalledWith(expect.stringContaining("1"));
  });

  it("truncates snippet to 500 chars", async () => {
    const db = makeDb();
    createChunksTable(db);
    const longText = "x".repeat(1000);
    db.prepare(
      `INSERT INTO chunks (id, path, source, start_line, end_line, hash, text, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
    ).run("id2", "/f.ts", "file", 1, 5, "h", longText, Date.now());

    const api = { getExtension: () => db };
    const log = makeLog();
    const result = await loadChunkMetadata(api as any, log);
    const entry = result.get("id2")!;
    expect(entry.snippet.length).toBe(500);
    expect(entry.content.length).toBe(1000);
  });

  it("handles missing text column gracefully", async () => {
    const db = makeDb();
    createChunksTable(db);
    db.prepare(
      `INSERT INTO chunks (id, path, source, start_line, end_line, hash, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?)`,
    ).run("id3", "/f.ts", "file", 1, 5, "h", Date.now());

    const api = { getExtension: () => db };
    const log = makeLog();
    const result = await loadChunkMetadata(api as any, log);
    const entry = result.get("id3")!;
    expect(entry.content).toBe("");
  });

  it("sets instanceId from row", async () => {
    const db = makeDb();
    createChunksTable(db);
    db.prepare(
      `INSERT INTO chunks (id, path, source, start_line, end_line, hash, text, instance_id, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    ).run("id4", "/f.ts", "file", 1, 5, "h", "txt", "inst-abc", Date.now());

    const api = { getExtension: () => db };
    const log = makeLog();
    const result = await loadChunkMetadata(api as any, log);
    expect(result.get("id4")!.instanceId).toBe("inst-abc");
  });

  it("logs error when query fails", async () => {
    const brokenDb = {
      prepare: () => ({ iterate: () => { throw new Error("query failed"); } }),
    };
    const api = { getExtension: () => brokenDb };
    const log = makeLog();
    const result = await loadChunkMetadata(api as any, log);
    expect(result.size).toBe(0);
    expect(log.error).toHaveBeenCalledWith(expect.stringContaining("Failed to load"));
  });

  it("reads from file-based DB and closes owned handle (owned=true cleanup)", async () => {
    const { tempDir } = createTempSqliteDir();
    process.env.WOPR_HOME = tempDir;
    try {
      // Seed the file DB with a row to verify data is returned via owned path
      const setupDb = new DatabaseSync(join(tempDir, "memory", "index.sqlite"));
      setupDb.prepare(
        `INSERT INTO chunks (id, path, source, start_line, end_line, hash, text, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
      ).run("file-id1", "/from-file.ts", "file", 1, 5, "h", "file content", Date.now());
      setupDb.close();

      const api = { getExtension: () => undefined }; // force owned=true path
      const log = makeLog();
      const result = await loadChunkMetadata(api as any, log);
      expect(result.size).toBe(1);
      expect(result.get("file-id1")!.path).toBe("/from-file.ts");
      // Verify the owned handle was closed by opening the file again for writing
      const verifyDb = new DatabaseSync(join(tempDir, "memory", "index.sqlite"));
      verifyDb.close();
    } finally {
      rmSync(tempDir, { recursive: true, force: true });
    }
  });
});

// ─── getHnswPath ──────────────────────────────────────────────────────────────

describe.skipIf(!hasSqlite)("getHnswPath", () => {
  let origHome: string | undefined;

  beforeEach(() => {
    origHome = process.env.WOPR_HOME;
  });

  afterEach(() => {
    if (origHome === undefined) delete process.env.WOPR_HOME;
    else process.env.WOPR_HOME = origHome;
  });

  it("uses db.location() when available", () => {
    const db = { location: () => "/data/index.sqlite" };
    const api = { getExtension: () => db };
    expect(getHnswPath(api as any)).toBe("/data/index.sqlite.hnsw");
  });

  it("uses db.name when location() unavailable", () => {
    const db = { name: "/data/index.sqlite" };
    const api = { getExtension: () => db };
    expect(getHnswPath(api as any)).toBe("/data/index.sqlite.hnsw");
  });

  it("skips :memory: db and falls back to WOPR_HOME", () => {
    process.env.WOPR_HOME = "/home/user";
    const db = { location: () => ":memory:" };
    const api = { getExtension: () => db };
    expect(getHnswPath(api as any)).toBe("/home/user/memory/index.sqlite.hnsw");
  });

  it("falls back to WOPR_HOME when no db", () => {
    process.env.WOPR_HOME = "/home/user";
    const api = { getExtension: () => undefined };
    expect(getHnswPath(api as any)).toBe("/home/user/memory/index.sqlite.hnsw");
  });

  it("returns undefined when no db and no WOPR_HOME", () => {
    delete process.env.WOPR_HOME;
    const api = { getExtension: () => undefined };
    expect(getHnswPath(api as any)).toBeUndefined();
  });
});

// ─── persistNewEntryToDb ──────────────────────────────────────────────────────

describe.skipIf(!hasSqlite)("persistNewEntryToDb", () => {
  it("inserts entry into db", () => {
    const db = makeDb();
    createChunksTable(db);
    const api = { getExtension: () => db };
    const entry = {
      id: "e1",
      path: "/f.ts",
      source: "file",
      startLine: 1,
      endLine: 5,
      snippet: "hello",
      content: "hello world",
      embedding: [0.1, 0.2, 0.3],
    };
    const searchManager = { getEntry: () => entry };
    const embeddingProvider = { id: "model-a" };
    const log = makeLog();

    persistNewEntryToDb(api as any, "e1", searchManager as any, embeddingProvider, "inst1", log);

    const row = db.prepare("SELECT * FROM chunks WHERE id = ?").get("e1") as any;
    expect(row).not.toBeNull();
    expect(row.path).toBe("/f.ts");
    expect(row.model).toBe("model-a");
    expect(row.instance_id).toBe("inst1");
    expect(row.embedding).toBeInstanceOf(Uint8Array);
    expect(log.warn).not.toHaveBeenCalled();
  });

  it("upserts on conflict", () => {
    const db = makeDb();
    createChunksTable(db);
    const api = { getExtension: () => db };
    const entry = {
      id: "e2", path: "/f.ts", source: "file", startLine: 1, endLine: 5,
      snippet: "s", content: "content v1", embedding: [0.1],
    };
    const searchManager = { getEntry: () => entry };
    const log = makeLog();

    persistNewEntryToDb(api as any, "e2", searchManager as any, { id: "m1" }, undefined, log);
    entry.content = "content v2";
    persistNewEntryToDb(api as any, "e2", searchManager as any, { id: "m1" }, undefined, log);

    const rows = db.prepare("SELECT * FROM chunks WHERE id = ?").all("e2");
    expect(rows.length).toBe(1);
    expect((rows[0] as any).text).toBe("content v2");
  });

  it("silently returns when entry not found", () => {
    const db = makeDb();
    createChunksTable(db);
    const api = { getExtension: () => db };
    const searchManager = { getEntry: () => undefined };
    const log = makeLog();
    expect(() =>
      persistNewEntryToDb(api as any, "x", searchManager as any, { id: "m" }, undefined, log),
    ).not.toThrow();
  });

  it("silently returns when embeddingProvider is null", () => {
    const db = makeDb();
    createChunksTable(db);
    const api = { getExtension: () => db };
    const entry = { id: "e3", path: "/f.ts", source: "file", startLine: 1, endLine: 5, snippet: "s", content: "c", embedding: [0.1] };
    const searchManager = { getEntry: () => entry };
    const log = makeLog();
    expect(() =>
      persistNewEntryToDb(api as any, "e3", searchManager as any, null, undefined, log),
    ).not.toThrow();
  });

  it("silently returns when embedding is empty", () => {
    const db = makeDb();
    createChunksTable(db);
    const api = { getExtension: () => db };
    const entry = { id: "e4", path: "/f.ts", source: "file", startLine: 1, endLine: 5, snippet: "s", content: "c", embedding: [] };
    const searchManager = { getEntry: () => entry };
    const log = makeLog();
    persistNewEntryToDb(api as any, "e4", searchManager as any, { id: "m" }, undefined, log);
    const row = db.prepare("SELECT * FROM chunks WHERE id = ?").get("e4");
    expect(row).toBeUndefined();
  });

  it("silently returns when no db", () => {
    const api = { getExtension: () => undefined };
    const entry = { id: "e5", path: "/f.ts", source: "file", startLine: 1, endLine: 5, snippet: "s", content: "c", embedding: [0.1] };
    const searchManager = { getEntry: () => entry };
    const log = makeLog();
    expect(() =>
      persistNewEntryToDb(api as any, "e5", searchManager as any, { id: "m" }, undefined, log),
    ).not.toThrow();
  });

  it("stores large payloads correctly", () => {
    const db = makeDb();
    createChunksTable(db);
    const api = { getExtension: () => db };
    const largeContent = "x".repeat(100_000);
    const largeEmbedding = Array.from({ length: 1536 }, (_, i) => i / 1536);
    const entry = { id: "e6", path: "/big.ts", source: "file", startLine: 1, endLine: 1000, snippet: "s", content: largeContent, embedding: largeEmbedding };
    const searchManager = { getEntry: () => entry };
    const log = makeLog();
    persistNewEntryToDb(api as any, "e6", searchManager as any, { id: "m" }, undefined, log);
    const row = db.prepare("SELECT * FROM chunks WHERE id = ?").get("e6") as any;
    expect(row.text.length).toBe(100_000);
    expect((row.embedding as Uint8Array).byteLength).toBe(1536 * 4);
    expect(log.warn).not.toHaveBeenCalled();
  });

  it("uses entry instanceId over passed instanceId", () => {
    const db = makeDb();
    createChunksTable(db);
    const api = { getExtension: () => db };
    const entry = { id: "e7", path: "/f.ts", source: "file", startLine: 1, endLine: 5, snippet: "s", content: "c", embedding: [0.1], instanceId: "entry-inst" };
    const searchManager = { getEntry: () => entry };
    const log = makeLog();
    persistNewEntryToDb(api as any, "e7", searchManager as any, { id: "m" }, "fallback-inst", log);
    const row = db.prepare("SELECT * FROM chunks WHERE id = ?").get("e7") as any;
    expect(row.instance_id).toBe("entry-inst");
  });

  it("warns on SQL error", () => {
    const brokenDb = {
      prepare: () => ({ all: () => [], run: () => { throw new Error("sql error"); } }),
      exec: vi.fn(),
    };
    const api = { getExtension: () => brokenDb };
    const entry = { id: "e8", path: "/f.ts", source: "file", startLine: 1, endLine: 5, snippet: "s", content: "c", embedding: [0.1] };
    const searchManager = { getEntry: () => entry };
    const log = makeLog();
    persistNewEntryToDb(api as any, "e8", searchManager as any, { id: "m" }, undefined, log);
    expect(log.warn).toHaveBeenCalledWith(expect.stringContaining("persistNewEntryToDb failed"));
  });
});
