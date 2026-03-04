import { describe, expect, it } from "vitest";
import { pluginConfigSchema } from "../../src/manifest.js";

describe("pluginConfigSchema", () => {
  it("includes all required field names", () => {
    const names = pluginConfigSchema.fields.map((f) => f.name);
    expect(names).toContain("provider");
    expect(names).toContain("apiKey");
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

  it("instanceId is text with no default", () => {
    const field = pluginConfigSchema.fields.find((f) => f.name === "instanceId");
    expect(field).toBeDefined();
    expect(field!.type).toBe("text");
    expect(field!.default).toBeUndefined();
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
});
