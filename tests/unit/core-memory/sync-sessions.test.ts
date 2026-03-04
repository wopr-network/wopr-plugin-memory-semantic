import { describe, it, expect, vi, beforeEach } from "vitest";
import type { PluginLogger, StorageApi } from "@wopr-network/plugin-types";
import { mockLogger, mockStorage } from "./helpers.js";

vi.mock("../../../src/core-memory/session-files.js", () => ({
  listSessionNames: vi.fn(),
  buildSessionEntryFromSql: vi.fn(),
}));

import { listSessionNames, buildSessionEntryFromSql } from "../../../src/core-memory/session-files.js";
import { syncSessionFiles } from "../../../src/core-memory/sync-sessions.js";

function mockSessionApi() {
  return {
    getContext: vi.fn(),
    setContext: vi.fn(),
    readConversationLog: vi.fn().mockResolvedValue([]),
  };
}

async function runSync(overrides?: {
  storage?: StorageApi;
  log?: PluginLogger;
  sessionApi?: ReturnType<typeof mockSessionApi> | null;
  needsFullReindex?: boolean;
  sessionNames?: string[];
  dirtyFiles?: Set<string>;
}) {
  const storage = overrides?.storage ?? mockStorage();
  const log = overrides?.log ?? mockLogger();
  const sessionApi = overrides?.sessionApi !== undefined ? overrides.sessionApi : mockSessionApi();
  const indexSessionFile = vi.fn().mockResolvedValue(undefined);
  const runWithConcurrency = vi.fn().mockImplementation(async (tasks: Array<() => Promise<unknown>>) => {
    for (const task of tasks) {
      await task();
    }
    return { results: [], hadErrors: false };
  });

  await syncSessionFiles({
    storage,
    sessionsDir: "/sessions",
    needsFullReindex: overrides?.needsFullReindex ?? false,
    ftsTable: "memory_chunks_fts",
    ftsEnabled: true,
    ftsAvailable: true,
    model: "fts5",
    dirtyFiles: overrides?.dirtyFiles ?? new Set(),
    runWithConcurrency,
    indexSessionFile,
    concurrency: 2,
    log,
    sessionApi: sessionApi as any,
  });

  return { storage, log, indexSessionFile, runWithConcurrency };
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("syncSessionFiles", () => {
  it("warns and returns early when sessionApi is not provided", async () => {
    const log = mockLogger();
    await syncSessionFiles({
      storage: mockStorage(),
      sessionsDir: "/sessions",
      needsFullReindex: false,
      ftsTable: "memory_chunks_fts",
      ftsEnabled: true,
      ftsAvailable: true,
      model: "fts5",
      dirtyFiles: new Set(),
      runWithConcurrency: vi.fn(),
      indexSessionFile: vi.fn(),
      concurrency: 1,
      log,
      sessionApi: undefined,
    });
    expect(vi.mocked(log.warn)).toHaveBeenCalledWith(
      expect.stringContaining("ctx.session not available"),
    );
  });

  it("indexes all sessions when needsFullReindex=true", async () => {
    vi.mocked(listSessionNames).mockResolvedValue(["session-a", "session-b"]);
    vi.mocked(buildSessionEntryFromSql).mockResolvedValue({
      path: "sessions/session-a",
      absPath: "sql://sessions/session-a",
      mtimeMs: 1000,
      size: 100,
      hash: "hash1",
      content: "User: hi",
    } as any);

    const storage = mockStorage();
    vi.mocked(storage.raw).mockResolvedValue([]); // no existing records

    const { indexSessionFile } = await runSync({ storage, needsFullReindex: true });
    expect(indexSessionFile).toHaveBeenCalled();
  });

  it("skips sessions that haven't changed (hash match)", async () => {
    vi.mocked(listSessionNames).mockResolvedValue(["session-a"]);
    vi.mocked(buildSessionEntryFromSql).mockResolvedValue({
      path: "sessions/session-a",
      absPath: "sql://sessions/session-a",
      mtimeMs: 1000,
      size: 100,
      hash: "samehash",
      content: "content",
    } as any);

    const storage = mockStorage();
    vi.mocked(storage.raw).mockImplementation(async (sql: string) => {
      if (sql.includes("SELECT hash")) return [{ hash: "samehash" }];
      return [];
    });

    const { indexSessionFile } = await runSync({ storage, needsFullReindex: false });
    expect(indexSessionFile).not.toHaveBeenCalled();
  });

  it("indexes session when hash differs", async () => {
    vi.mocked(listSessionNames).mockResolvedValue(["session-a"]);
    vi.mocked(buildSessionEntryFromSql).mockResolvedValue({
      path: "sessions/session-a",
      absPath: "sql://sessions/session-a",
      mtimeMs: 1000,
      size: 100,
      hash: "newhash",
      content: "new content",
    } as any);

    const storage = mockStorage();
    vi.mocked(storage.raw).mockImplementation(async (sql: string) => {
      if (sql.includes("SELECT hash")) return [{ hash: "oldhash" }];
      return [];
    });

    const { indexSessionFile } = await runSync({ storage, needsFullReindex: false });
    expect(indexSessionFile).toHaveBeenCalled();
  });

  it("removes stale session entries from DB", async () => {
    vi.mocked(listSessionNames).mockResolvedValue(["active-session"]);
    vi.mocked(buildSessionEntryFromSql).mockResolvedValue(null);

    const storage = mockStorage();
    vi.mocked(storage.raw).mockImplementation(async (sql: string) => {
      if (sql.includes("SELECT path FROM memory_files WHERE source")) {
        return [{ path: "sessions/stale-session" }, { path: "sessions/active-session" }];
      }
      return [];
    });

    const { storage: s } = await runSync({ storage });

    const deleteCalls = vi.mocked(s.raw).mock.calls.filter((c) =>
      c[0].includes("DELETE FROM memory_files") && JSON.stringify(c[1]).includes("stale-session"),
    );
    expect(deleteCalls.length).toBeGreaterThan(0);
  });

  it("only indexes dirty sessions when dirtyFiles is non-empty and needsFullReindex=false", async () => {
    vi.mocked(listSessionNames).mockResolvedValue(["session-a", "session-b"]);
    vi.mocked(buildSessionEntryFromSql).mockImplementation(async (name) => ({
      path: `sessions/${name}`,
      absPath: `sql://sessions/${name}`,
      mtimeMs: 1000,
      size: 100,
      hash: `hash-${name}`,
      content: "content",
    } as any));

    const storage = mockStorage();
    vi.mocked(storage.raw).mockResolvedValue([]); // no existing records

    const dirtyFiles = new Set(["sessions/session-a"]);
    const { indexSessionFile } = await runSync({ storage, needsFullReindex: false, dirtyFiles });

    // Only session-a should be indexed
    const indexedPaths = (indexSessionFile as any).mock.calls.map((c: any) => c[0]?.path);
    expect(indexedPaths).toContain("sessions/session-a");
    expect(indexedPaths).not.toContain("sessions/session-b");
  });
});
