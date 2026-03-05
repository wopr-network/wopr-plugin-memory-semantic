import { describe, it, expect } from "vitest";
import { DEFAULT_CONFIG } from "../../src/types.js";

describe("DEFAULT_CONFIG security defaults", () => {
  it("excludeLegacyEntries defaults to true to prevent cross-tenant leakage", () => {
    expect(DEFAULT_CONFIG.search.excludeLegacyEntries).toBe(true);
  });
});
