import { describe, expect, it } from "vitest";
import { multiScaleChunk } from "../../src/chunking.js";

const META = { path: "test.ts", startLine: 1, endLine: 10, source: "test" };

describe("multiScaleChunk", () => {
  describe("empty and short input", () => {
    it("returns no entries for empty string", () => {
      const result = multiScaleChunk("", "id1", META, [{ tokens: 50, overlap: 10 }]);
      // Both canonical and scale entries require trim().length >= 10
      expect(result.some((e) => e.entry.id === "id1")).toBe(false);
      expect(result.some((e) => e.entry.id === "id1-s50")).toBe(false);
    });

    it("returns no entries for whitespace-only string", () => {
      const result = multiScaleChunk("         ", "id1", META, [{ tokens: 50, overlap: 10 }]);
      // Both canonical and scale entries require trim().length >= 10
      expect(result.some((e) => e.entry.id === "id1")).toBe(false);
      expect(result.some((e) => e.entry.id === "id1-s50")).toBe(false);
    });

    it("does not emit any entry for string shorter than 10 chars after trim", () => {
      const result = multiScaleChunk("short", "id1", META, [{ tokens: 50, overlap: 10 }]);
      // Both canonical and single-chunk scale entries require trim().length >= 10
      expect(result.some((e) => e.entry.id === "id1")).toBe(false);
      expect(result.some((e) => e.entry.id === "id1-s50")).toBe(false);
    });

    it("returns canonical + scale entry for input exactly 10 chars after trim", () => {
      const text = "0123456789"; // exactly 10 chars
      const result = multiScaleChunk(text, "id1", META, [{ tokens: 50, overlap: 10 }]);
      expect(result.some((e) => e.entry.id === "id1")).toBe(true);
      expect(result.some((e) => e.entry.id === "id1-s50")).toBe(true);
    });
  });

  describe("single-token input (fits in one chunk)", () => {
    it("emits canonical entry and one scale entry when text fits", () => {
      const text = "This is a small piece of text for testing purposes.";
      const result = multiScaleChunk(text, "base1", META, [{ tokens: 100, overlap: 10 }]);
      // canonical entry (baseId)
      const canonical = result.find((e) => e.entry.id === "base1");
      expect(canonical).toBeDefined();
      expect(canonical!.entry.content).toBe(text);
      expect(canonical!.text).toBe(text);
      // scale entry
      const scaleEntry = result.find((e) => e.entry.id === "base1-s100");
      expect(scaleEntry).toBeDefined();
      expect(scaleEntry!.entry.content).toBe(text);
      expect(scaleEntry!.text).toBe(text);
    });
  });

  describe("chunk boundaries at configured size", () => {
    it("splits text into correct number of chunks based on tokens * 4", () => {
      // tokens=10 => maxChars=40, overlap=0
      // 80 char text => 2 chunks of 40
      const text = "A".repeat(80);
      const result = multiScaleChunk(text, "id1", META, [{ tokens: 10, overlap: 0 }]);
      const scaleEntries = result.filter((e) => e.entry.id.startsWith("id1-s10-"));
      expect(scaleEntries).toHaveLength(2);
      expect(scaleEntries[0].entry.content).toBe("A".repeat(40));
      expect(scaleEntries[1].entry.content).toBe("A".repeat(40));
      expect(scaleEntries[0].entry.id).toBe("id1-s10-0");
      expect(scaleEntries[1].entry.id).toBe("id1-s10-1");
    });

    it("last chunk contains remainder when text is not evenly divisible", () => {
      // tokens=10 => maxChars=40, overlap=0
      // 50 char text => chunk 0 = 40 chars, chunk 1 = 10 chars
      const text = "B".repeat(50);
      const result = multiScaleChunk(text, "id1", META, [{ tokens: 10, overlap: 0 }]);
      const scaleEntries = result.filter((e) => e.entry.id.startsWith("id1-s10-"));
      expect(scaleEntries).toHaveLength(2);
      expect(scaleEntries[0].entry.content.length).toBe(40);
      expect(scaleEntries[1].entry.content.length).toBe(10);
    });

    it("canonical entry content is truncated to smallest scale maxChars", () => {
      const text = "C".repeat(200);
      const result = multiScaleChunk(text, "id1", META, [{ tokens: 10, overlap: 0 }]);
      const canonical = result.find((e) => e.entry.id === "id1");
      expect(canonical).toBeDefined();
      // smallest.tokens=10, maxChars=40
      expect(canonical!.entry.content.length).toBe(40);
      expect(canonical!.text.length).toBe(40);
    });
  });

  describe("overlap between consecutive chunks", () => {
    it("consecutive chunks overlap by the configured amount", () => {
      // tokens=10 => maxChars=40, overlap=5 => overlapChars=20
      // stride = 40 - 20 = 20 chars per step
      const text = "0123456789".repeat(10); // 100 chars
      const result = multiScaleChunk(text, "id1", META, [{ tokens: 10, overlap: 5 }]);
      const scaleEntries = result.filter((e) => e.entry.id.startsWith("id1-s10-"));
      // Check overlap: chunk[n+1] starts at end[n] - overlapChars
      for (let i = 0; i < scaleEntries.length - 1; i++) {
        const thisContent = scaleEntries[i].entry.content;
        const nextContent = scaleEntries[i + 1].entry.content;
        // Last 20 chars of this chunk should equal first 20 chars of next
        const overlapFromThis = thisContent.slice(-20);
        const overlapFromNext = nextContent.slice(0, 20);
        expect(overlapFromThis).toBe(overlapFromNext);
      }
    });

    it("overlap of zero produces no shared content", () => {
      const text = "X".repeat(80);
      const result = multiScaleChunk(text, "id1", META, [{ tokens: 10, overlap: 0 }]);
      const scaleEntries = result.filter((e) => e.entry.id.startsWith("id1-s10-"));
      // 80 chars / 40 maxChars = 2 chunks, no overlap
      expect(scaleEntries).toHaveLength(2);
      // Total content length = original text length (no duplication)
      const totalLen = scaleEntries.reduce((sum, e) => sum + e.entry.content.length, 0);
      expect(totalLen).toBe(80);
    });
  });

  describe("multi-scale output", () => {
    it("produces entries for each scale", () => {
      const text = "Hello world, this is a multi-scale chunking test input string.";
      const scales = [
        { tokens: 5, overlap: 1 }, // maxChars=20
        { tokens: 50, overlap: 10 }, // maxChars=200
      ];
      const result = multiScaleChunk(text, "id1", META, scales);
      // Should have entries for both scale 5 and scale 50
      expect(result.some((e) => e.entry.id.includes("-s5"))).toBe(true);
      expect(result.some((e) => e.entry.id.includes("-s50"))).toBe(true);
    });

    it("smallest scale is used for canonical entry", () => {
      const text = "D".repeat(100);
      const scales = [
        { tokens: 50, overlap: 0 }, // maxChars=200
        { tokens: 10, overlap: 0 }, // maxChars=40 (smallest)
      ];
      const result = multiScaleChunk(text, "id1", META, scales);
      const canonical = result.find((e) => e.entry.id === "id1");
      expect(canonical).toBeDefined();
      // Should use smallest scale (tokens=10, maxChars=40) for canonical
      expect(canonical!.entry.content.length).toBe(40);
    });

    it("large scale fits text in one chunk while small scale splits", () => {
      const text = "E".repeat(100);
      const scales = [
        { tokens: 10, overlap: 0 }, // maxChars=40 => splits
        { tokens: 50, overlap: 0 }, // maxChars=200 => single chunk
      ];
      const result = multiScaleChunk(text, "id1", META, scales);
      const smallScale = result.filter((e) => e.entry.id.startsWith("id1-s10-"));
      const largeScale = result.filter((e) => e.entry.id.startsWith("id1-s50"));
      expect(smallScale.length).toBeGreaterThan(1);
      // large scale: single entry (no suffix for single-chunk)
      expect(largeScale).toHaveLength(1);
      expect(largeScale[0].entry.content).toBe(text);
    });
  });

  describe("overlap clamping", () => {
    it("clamps overlap when overlapChars >= maxChars", () => {
      // tokens=10 => maxChars=40, overlap=10 => overlapChars=40 >= maxChars
      // Should clamp to maxChars - 4 = 36, stride = 4
      const text = "F".repeat(100);
      const result = multiScaleChunk(text, "id1", META, [{ tokens: 10, overlap: 10 }]);
      const scaleEntries = result.filter((e) => e.entry.id.startsWith("id1-s10-"));
      // Ensure it terminates (no infinite loop) and produces multiple chunks
      expect(scaleEntries.length).toBeGreaterThan(2);
    });

    it("clamps overlap when overlap exceeds tokens", () => {
      // tokens=5 => maxChars=20, overlap=100 => overlapChars=400 >= 20
      // Should clamp to 20 - 4 = 16, stride = 4
      const text = "G".repeat(60);
      const result = multiScaleChunk(text, "id1", META, [{ tokens: 5, overlap: 100 }]);
      const scaleEntries = result.filter((e) => e.entry.id.startsWith("id1-s5-"));
      expect(scaleEntries.length).toBeGreaterThan(1);
    });
  });

  describe("invalid scale guard", () => {
    it("skips scale with tokens=0", () => {
      const text = "Valid text for testing this edge case.";
      const result = multiScaleChunk(text, "id1", META, [
        { tokens: 0, overlap: 0 },
        { tokens: 50, overlap: 0 },
      ]);
      expect(result.some((e) => e.entry.id.includes("-s0"))).toBe(false);
      expect(result.some((e) => e.entry.id.includes("-s50"))).toBe(true);
    });

    it("canonical entry uses valid scale when invalid scale is listed first (tokens=0)", () => {
      const text = "Valid text for testing this edge case.";
      const result = multiScaleChunk(text, "id1", META, [
        { tokens: 0, overlap: 0 },
        { tokens: 50, overlap: 0 },
      ]);
      const canonical = result.find((e) => e.entry.id === "id1");
      expect(canonical).toBeDefined();
      // Canonical content must be non-empty (derived from valid scale tokens=50, maxChars=200)
      expect(canonical!.entry.content.length).toBeGreaterThan(0);
      expect(canonical!.entry.content).toBe(text);
    });

    it("skips scale with negative tokens", () => {
      const text = "Valid text for testing this edge case.";
      const result = multiScaleChunk(text, "id1", META, [
        { tokens: -5, overlap: 0 },
        { tokens: 50, overlap: 0 },
      ]);
      expect(result.some((e) => e.entry.id.includes("-s-5"))).toBe(false);
      expect(result.some((e) => e.entry.id.includes("-s50"))).toBe(true);
    });

    it("canonical entry uses valid scale when invalid scale is listed first (negative tokens)", () => {
      const text = "Valid text for testing this edge case.";
      const result = multiScaleChunk(text, "id1", META, [
        { tokens: -5, overlap: 0 },
        { tokens: 50, overlap: 0 },
      ]);
      const canonical = result.find((e) => e.entry.id === "id1");
      expect(canonical).toBeDefined();
      // Canonical content must be non-empty (derived from valid scale tokens=50, maxChars=200)
      expect(canonical!.entry.content.length).toBeGreaterThan(0);
    });

    it("skips scale with NaN tokens", () => {
      const text = "Valid text for testing this edge case.";
      const result = multiScaleChunk(text, "id1", META, [
        { tokens: NaN, overlap: 0 },
        { tokens: 50, overlap: 0 },
      ]);
      expect(result.some((e) => e.entry.id.includes("-sNaN"))).toBe(false);
    });

    it("skips scale with Infinity tokens", () => {
      const text = "Valid text for testing this edge case.";
      const result = multiScaleChunk(text, "id1", META, [
        { tokens: Infinity, overlap: 0 },
        { tokens: 50, overlap: 0 },
      ]);
      expect(result.some((e) => e.entry.id.includes("-sInfinity"))).toBe(false);
    });
  });

  describe("non-finite overlap handling", () => {
    it("treats NaN overlap as zero", () => {
      const text = "H".repeat(80);
      const resultNaN = multiScaleChunk(text, "id1", META, [{ tokens: 10, overlap: NaN }]);
      const resultZero = multiScaleChunk(text, "id2", META, [{ tokens: 10, overlap: 0 }]);
      const chunksNaN = resultNaN.filter((e) => e.entry.id.startsWith("id1-s10-"));
      const chunksZero = resultZero.filter((e) => e.entry.id.startsWith("id2-s10-"));
      expect(chunksNaN.length).toBe(chunksZero.length);
    });

    it("treats negative overlap as zero (clamped)", () => {
      // tokens=10 => maxChars=40; text=120 chars => 3 chunks at zero overlap
      // Without clamping, overlap=-5 => overlapChars=-20 => stride=60, only 2 chunks
      const text = "H".repeat(120);
      const resultNegative = multiScaleChunk(text, "id1", META, [{ tokens: 10, overlap: -5 }]);
      const resultZero = multiScaleChunk(text, "id2", META, [{ tokens: 10, overlap: 0 }]);
      const chunksNegative = resultNegative.filter((e) => e.entry.id.startsWith("id1-s10-"));
      const chunksZero = resultZero.filter((e) => e.entry.id.startsWith("id2-s10-"));
      expect(chunksNegative.length).toBe(chunksZero.length);
    });
  });

  describe("non-ASCII and emoji text", () => {
    it("handles emoji text correctly", () => {
      // Emojis are multi-byte but .length counts UTF-16 code units
      const emoji = "\u{1F600}"; // grinning face, .length = 2
      const text = emoji.repeat(30); // .length = 60
      const result = multiScaleChunk(text, "id1", META, [{ tokens: 5, overlap: 0 }]);
      // maxChars = 20, text.length = 60 => 3 chunks
      const scaleEntries = result.filter((e) => e.entry.id.startsWith("id1-s5-"));
      expect(scaleEntries).toHaveLength(3);
    });

    it("handles CJK characters", () => {
      const text = "\u4F60\u597D\u4E16\u754C".repeat(15); // 60 chars
      const result = multiScaleChunk(text, "id1", META, [{ tokens: 5, overlap: 0 }]);
      const scaleEntries = result.filter((e) => e.entry.id.startsWith("id1-s5-"));
      expect(scaleEntries).toHaveLength(3);
    });

    it("handles mixed ASCII and emoji", () => {
      const text = "Hello \u{1F600} World! ".repeat(5); // ~75 chars
      const result = multiScaleChunk(text, "id1", META, [{ tokens: 50, overlap: 0 }]);
      // Should produce at least canonical + scale entries without error
      expect(result.length).toBeGreaterThan(0);
    });
  });

  describe("snippet truncation", () => {
    it("snippet is truncated to 500 chars for canonical entry", () => {
      const text = "S".repeat(1000);
      const result = multiScaleChunk(text, "id1", META, [{ tokens: 500, overlap: 0 }]);
      const canonical = result.find((e) => e.entry.id === "id1");
      expect(canonical).toBeDefined();
      expect(canonical!.entry.snippet.length).toBe(500);
    });

    it("snippet is truncated to 500 chars for sub-chunks", () => {
      const text = "T".repeat(10000);
      const result = multiScaleChunk(text, "id1", META, [{ tokens: 1000, overlap: 0 }]);
      const subChunks = result.filter((e) => e.entry.id.startsWith("id1-s1000-"));
      for (const chunk of subChunks) {
        expect(chunk.entry.snippet.length).toBeLessThanOrEqual(500);
      }
    });
  });

  describe("metadata propagation", () => {
    it("propagates path, startLine, endLine, source to all entries", () => {
      const meta = { path: "foo/bar.ts", startLine: 42, endLine: 99, source: "bootstrap" };
      const text = "Metadata propagation test with enough length.";
      const result = multiScaleChunk(text, "id1", meta, [{ tokens: 50, overlap: 0 }]);
      for (const entry of result) {
        expect(entry.entry.path).toBe("foo/bar.ts");
        expect(entry.entry.startLine).toBe(42);
        expect(entry.entry.endLine).toBe(99);
        expect(entry.entry.source).toBe("bootstrap");
      }
    });

    it("propagates instanceId when provided", () => {
      const meta = {
        path: "a.ts",
        startLine: 1,
        endLine: 5,
        source: "test",
        instanceId: "tenant-42",
      };
      const text = "Instance ID propagation test text here.";
      const result = multiScaleChunk(text, "id1", meta, [{ tokens: 50, overlap: 0 }]);
      for (const entry of result) {
        expect(entry.entry.instanceId).toBe("tenant-42");
      }
    });

    it("instanceId is undefined when not provided", () => {
      const text = "No instance ID in this test case here.";
      const result = multiScaleChunk(text, "id1", META, [{ tokens: 50, overlap: 0 }]);
      for (const entry of result) {
        expect(entry.entry.instanceId).toBeUndefined();
      }
    });
  });

  describe("sub-chunk minimum length filter", () => {
    it("skips sub-chunks with trimmed length less than 10", () => {
      // tokens=10 => maxChars=40, overlap=0
      // Create text where last chunk is short whitespace
      const text = "A".repeat(40) + "   hi   "; // 48 chars, last chunk = "   hi   " (trim=2 < 10)
      const result = multiScaleChunk(text, "id1", META, [{ tokens: 10, overlap: 0 }]);
      const scaleEntries = result.filter((e) => e.entry.id.startsWith("id1-s10-"));
      // Only first chunk should be emitted
      expect(scaleEntries).toHaveLength(1);
      expect(scaleEntries[0].entry.id).toBe("id1-s10-0");
    });
  });
});
