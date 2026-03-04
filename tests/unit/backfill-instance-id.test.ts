/**
 * Tests for backfillLegacyInstanceId admin tool (WOP-1553)
 */
import { describe, expect, it, vi } from "vitest";
import { backfillLegacyInstanceId } from "../../src/persistence.js";

function makeLogger() {
  return { info: vi.fn(), warn: vi.fn(), error: vi.fn() };
}

describe("backfillLegacyInstanceId", () => {
  it("returns 0 when no database handle is available", () => {
    const api = { getExtension: () => undefined };
    const log = makeLogger();
    expect(backfillLegacyInstanceId(api, "inst-1", log)).toBe(0);
    expect(log.warn).toHaveBeenCalledWith(expect.stringContaining("no database handle"));
  });

  it("updates rows with NULL instance_id and returns count", () => {
    const runResult = { changes: 3 };
    const prepareMock = vi.fn().mockReturnValue({ run: vi.fn().mockReturnValue(runResult) });
    const db = { prepare: prepareMock };
    const api = { getExtension: (name: string) => (name === "memory:db" ? db : undefined) };
    const log = makeLogger();

    const count = backfillLegacyInstanceId(api, "inst-1", log);
    expect(count).toBe(3);
    expect(prepareMock).toHaveBeenCalledWith(
      "UPDATE chunks SET instance_id = ? WHERE instance_id IS NULL",
    );
    expect(log.info).toHaveBeenCalledWith(expect.stringContaining("Backfilled 3"));
  });

  it("returns 0 on database error", () => {
    const prepareMock = vi.fn().mockImplementation(() => {
      throw new Error("db locked");
    });
    const db = { prepare: prepareMock };
    const api = { getExtension: (name: string) => (name === "memory:db" ? db : undefined) };
    const log = makeLogger();

    const count = backfillLegacyInstanceId(api, "inst-1", log);
    expect(count).toBe(0);
    expect(log.error).toHaveBeenCalledWith(expect.stringContaining("db locked"));
  });
});
