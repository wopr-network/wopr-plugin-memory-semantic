import { describe, expect, it, vi } from "vitest";

vi.mock("node:fs/promises", () => ({
  default: {
    readdir: vi.fn().mockResolvedValue([]),
    stat: vi.fn().mockRejectedValue(new Error("ENOENT")),
  },
}));

import { discoverSessionMemoryDirs } from "../../src/a2a-tools.js";

describe("discoverSessionMemoryDirs", () => {
  it("should be exported from a2a-tools", () => {
    expect(typeof discoverSessionMemoryDirs).toBe("function");
  });

  it("should return empty array when sessions dir is empty", async () => {
    const result = await discoverSessionMemoryDirs();
    expect(result).toEqual([]);
  });
});
