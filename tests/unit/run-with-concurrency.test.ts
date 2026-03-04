import { describe, it, expect } from "vitest";
import { runWithConcurrency } from "../../src/core-memory/run-with-concurrency.js";

describe("runWithConcurrency", () => {
  it("should continue processing tasks when one rejects", async () => {
    const results = await runWithConcurrency(
      [
        () => Promise.resolve("a"),
        () => Promise.reject(new Error("fail")),
        () => Promise.resolve("c"),
      ],
      2,
      () => {},
    );
    expect(results).toContain("a");
    expect(results).toContain("c");
    expect(results).toHaveLength(2);
  });

  it("should call onError for each rejected task", async () => {
    const errors: unknown[] = [];
    await runWithConcurrency(
      [
        () => Promise.resolve("ok"),
        () => Promise.reject(new Error("boom")),
        () => Promise.reject(new Error("bang")),
      ],
      3,
      (err) => errors.push(err),
    );
    expect(errors).toHaveLength(2);
    expect((errors[0] as Error).message).toBe("boom");
    expect((errors[1] as Error).message).toBe("bang");
  });

  it("should handle all tasks rejecting", async () => {
    const errors: unknown[] = [];
    const results = await runWithConcurrency(
      [
        () => Promise.reject(new Error("e1")),
        () => Promise.reject(new Error("e2")),
      ],
      2,
      (err) => errors.push(err),
    );
    expect(results).toHaveLength(0);
    expect(errors).toHaveLength(2);
  });

  it("should respect concurrency limit even with failures", async () => {
    let concurrent = 0;
    let maxConcurrent = 0;
    const task = (val: string, fail = false) => async () => {
      concurrent++;
      maxConcurrent = Math.max(maxConcurrent, concurrent);
      await new Promise((r) => setTimeout(r, 10));
      concurrent--;
      if (fail) throw new Error("fail");
      return val;
    };

    await runWithConcurrency(
      [task("a"), task("b", true), task("c"), task("d"), task("e", true)],
      2,
      () => {},
    );
    expect(maxConcurrent).toBeLessThanOrEqual(2);
  });

  it("should not throw when onError is not provided and a task rejects", async () => {
    const results = await runWithConcurrency(
      [
        () => Promise.resolve("a"),
        () => Promise.reject(new Error("fail")),
        () => Promise.resolve("c"),
      ],
      2,
    );
    expect(results).toContain("a");
    expect(results).toContain("c");
    expect(results).toHaveLength(2);
  });
});
