import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { discoverSessionMemoryDirs } from "../../src/core-memory/manager.js";

describe("discoverSessionMemoryDirs", () => {
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), "discover-sessions-"));
  });

  afterEach(async () => {
    await fs.rm(tmpDir, { recursive: true, force: true });
  });

  it("discovers dirs that contain a memory/ subdir", async () => {
    const s1 = path.join(tmpDir, "session-1");
    const s2 = path.join(tmpDir, "session-2");
    await fs.mkdir(path.join(s1, "memory"), { recursive: true });
    await fs.mkdir(s2, { recursive: true }); // no memory subdir

    const result = await discoverSessionMemoryDirs(tmpDir);
    expect(result).toEqual([s1]);
  });

  it("returns empty array for nonexistent path", async () => {
    const result = await discoverSessionMemoryDirs(path.join(tmpDir, "nope"));
    expect(result).toEqual([]);
  });

  it("ignores files (only returns directories)", async () => {
    await fs.writeFile(path.join(tmpDir, "not-a-dir"), "hello");
    const result = await discoverSessionMemoryDirs(tmpDir);
    expect(result).toEqual([]);
  });
});
