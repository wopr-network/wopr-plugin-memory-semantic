const FAILED = Symbol("runWithConcurrency.failed");

export async function runWithConcurrency<T>(
  tasks: Array<() => Promise<T>>,
  concurrency: number,
  onError?: (err: unknown) => void,
): Promise<{ results: T[]; hadErrors: boolean }> {
  const slots: Array<T | typeof FAILED> = new Array(tasks.length).fill(FAILED);
  let hadErrors = false;
  const executing: Set<Promise<void>> = new Set();

  for (let i = 0; i < tasks.length; i++) {
    const task = tasks[i];
    const p = Promise.resolve()
      .then(() => task())
      .then((result) => {
        slots[i] = result;
      })
      .catch((err) => {
        hadErrors = true;
        if (onError) onError(err);
      })
      .finally(() => {
        executing.delete(p);
      });
    executing.add(p);

    if (executing.size >= concurrency) {
      await Promise.race(executing);
    }
  }

  await Promise.all(executing);
  const results = slots.filter((v): v is T => v !== FAILED);
  return { results, hadErrors };
}
