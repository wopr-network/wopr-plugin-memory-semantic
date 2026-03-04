import { describe, it, expect } from "vitest";
import { runWithConcurrency } from "../../src/core-memory/run-with-concurrency.js";

describe("runWithConcurrency", () => {
  it("should continue processing tasks when one rejects", async () => {
    const { results } = await runWithConcurrency(
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
    const { results } = await runWithConcurrency(
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
    const { results } = await runWithConcurrency(
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

  it("should return hadErrors=true when a task rejects", async () => {
    const { hadErrors } = await runWithConcurrency(
      [
        () => Promise.resolve("a"),
        () => Promise.reject(new Error("fail")),
        () => Promise.resolve("c"),
      ],
      2,
      () => {},
    );
    expect(hadErrors).toBe(true);
  });

  it("should return hadErrors=false when all tasks succeed", async () => {
    const { hadErrors } = await runWithConcurrency(
      [() => Promise.resolve("a"), () => Promise.resolve("b")],
      2,
      () => {},
    );
    expect(hadErrors).toBe(false);
  });

  it("should return results in submission order, not completion order", async () => {
    const { results } = await runWithConcurrency(
      [
        () => new Promise<string>((r) => setTimeout(() => r("slow"), 50)),
        () => new Promise<string>((r) => setTimeout(() => r("fast"), 10)),
        () => new Promise<string>((r) => setTimeout(() => r("medium"), 30)),
      ],
      3,
    );
    expect(results).toEqual(["slow", "fast", "medium"]);
  });

  it("should preserve undefined results from successful tasks", async () => {
    const { results, hadErrors } = await runWithConcurrency<string | undefined>(
      [
        () => Promise.resolve("a"),
        () => Promise.resolve(undefined),
        () => Promise.resolve("c"),
      ],
      3,
    );
    expect(hadErrors).toBe(false);
    expect(results).toEqual(["a", undefined, "c"]);
    expect(results).toHaveLength(3);
  });

  it("should catch synchronous throws from tasks", async () => {
    const errors: unknown[] = [];
    const { results, hadErrors } = await runWithConcurrency(
      [
        () => Promise.resolve("a"),
        () => { throw new Error("sync throw"); },
        () => Promise.resolve("c"),
      ],
      2,
      (err) => errors.push(err),
    );
    expect(hadErrors).toBe(true);
    expect(errors).toHaveLength(1);
    expect((errors[0] as Error).message).toBe("sync throw");
    expect(results).toContain("a");
    expect(results).toContain("c");
  });
});
