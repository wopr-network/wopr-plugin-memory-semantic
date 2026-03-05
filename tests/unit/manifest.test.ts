import { describe, expect, it } from "vitest";
import { mapFlatConfigToNested } from "../../src/index.js";
import { pluginConfigSchema } from "../../src/manifest.js";

describe("pluginConfigSchema", () => {
  it("includes all required field names", () => {
    const names = pluginConfigSchema.fields.map((f) => f.name);
    expect(names).toContain("provider");
    expect(names).not.toContain("apiKey");
    expect(names).toContain("model");
    expect(names).toContain("maxWriteBytes");
    expect(names).toContain("autoRecallEnabled");
    expect(names).toContain("autoCaptureEnabled");
    expect(names).toContain("instanceId");
    expect(names).toContain("searchMaxResults");
    expect(names).toContain("searchHybridWeight");
  });

  it("autoRecallEnabled is boolean with default true", () => {
    const field = pluginConfigSchema.fields.find((f) => f.name === "autoRecallEnabled");
    expect(field).toBeDefined();
    expect(field!.type).toBe("boolean");
    expect(field!.default).toBe(true);
  });

  it("autoCaptureEnabled is boolean with default true", () => {
    const field = pluginConfigSchema.fields.find((f) => f.name === "autoCaptureEnabled");
    expect(field).toBeDefined();
    expect(field!.type).toBe("boolean");
    expect(field!.default).toBe(true);
  });

  it("instanceId is text with default 'default'", () => {
    const field = pluginConfigSchema.fields.find((f) => f.name === "instanceId");
    expect(field).toBeDefined();
    expect(field!.type).toBe("text");
    expect(field!.default).toBe("default");
  });

  it("searchMaxResults is number with default 10", () => {
    const field = pluginConfigSchema.fields.find((f) => f.name === "searchMaxResults");
    expect(field).toBeDefined();
    expect(field!.type).toBe("number");
    expect(field!.default).toBe(10);
  });

  it("searchHybridWeight is number with default 0.7", () => {
    const field = pluginConfigSchema.fields.find((f) => f.name === "searchHybridWeight");
    expect(field).toBeDefined();
    expect(field!.type).toBe("number");
    expect(field!.default).toBe(0.7);
  });

  it("searchHybridWeight has a 0-1 range pattern", () => {
    const field = pluginConfigSchema.fields.find((f) => f.name === "searchHybridWeight");
    expect(field).toBeDefined();
    expect(field!.pattern).toBeDefined();
    const re = new RegExp(field!.pattern!);
    expect(re.test("0")).toBe(true);
    expect(re.test("0.5")).toBe(true);
    expect(re.test("1")).toBe(true);
    expect(re.test("1.0")).toBe(true);
    expect(re.test("1.5")).toBe(false);
    expect(re.test("-0.1")).toBe(false);
    expect(re.test("2")).toBe(false);
  });

  it("searchMaxResults description mentions integer rounding", () => {
    const field = pluginConfigSchema.fields.find((f) => f.name === "searchMaxResults");
    expect(field).toBeDefined();
    expect(field!.description).toMatch(/integer|round/i);
  });
});

describe("mapFlatConfigToNested", () => {
  it("maps autoRecallEnabled to autoRecall.enabled", () => {
    const result = mapFlatConfigToNested({ autoRecallEnabled: false });
    expect(result.autoRecall?.enabled).toBe(false);
  });

  it("maps autoCaptureEnabled to autoCapture.enabled", () => {
    const result = mapFlatConfigToNested({ autoCaptureEnabled: false });
    expect(result.autoCapture?.enabled).toBe(false);
  });

  it("maps searchMaxResults to search.maxResults (rounded)", () => {
    const result = mapFlatConfigToNested({ searchMaxResults: 25.7 });
    expect(result.search?.maxResults).toBe(26);
  });

  it("maps searchHybridWeight to hybrid.vectorWeight and computes textWeight", () => {
    const result = mapFlatConfigToNested({ searchHybridWeight: 0.6 });
    expect(result.hybrid?.vectorWeight).toBeCloseTo(0.6);
    expect(result.hybrid?.textWeight).toBeCloseTo(0.4);
  });

  it("clamps searchHybridWeight to [0, 1]", () => {
    expect(mapFlatConfigToNested({ searchHybridWeight: 1.5 }).hybrid?.vectorWeight).toBe(1);
    expect(mapFlatConfigToNested({ searchHybridWeight: -0.2 }).hybrid?.vectorWeight).toBe(0);
  });

  it("passes through flat keys directly", () => {
    const result = mapFlatConfigToNested({ provider: "openai", model: "ada", maxWriteBytes: 2048, instanceId: "bot1" });
    expect(result.provider).toBe("openai");
    expect(result.model).toBe("ada");
    expect(result.maxWriteBytes).toBe(2048);
    expect(result.instanceId).toBe("bot1");
  });

  it("preserves already-nested keys when no flat override", () => {
    const result = mapFlatConfigToNested({ search: { maxResults: 50, minScore: 0.2, candidateMultiplier: 5 } });
    expect(result.search?.maxResults).toBe(50);
  });

  it("flat key takes precedence over nested key", () => {
    // When both are present, the flat key mapping wins because it runs after
    // the nested pass-through
    const result = mapFlatConfigToNested({
      searchMaxResults: 15,
      search: { maxResults: 50, minScore: 0.2, candidateMultiplier: 5 },
    });
    // searchMaxResults flat key creates config.search first; nested search is skipped
    expect(result.search?.maxResults).toBe(15);
  });

  it("returns empty object for empty input", () => {
    expect(mapFlatConfigToNested({})).toEqual({});
  });
});
