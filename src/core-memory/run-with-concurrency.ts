export async function runWithConcurrency<T>(
  tasks: Array<() => Promise<T>>,
  concurrency: number,
  onError?: (err: unknown) => void,
): Promise<{ results: T[]; hadErrors: boolean }> {
  const results: T[] = [];
  let hadErrors = false;
  const executing: Set<Promise<void>> = new Set();

  for (const task of tasks) {
    const p = Promise.resolve()
      .then(() => task())
      .then((result) => {
        results.push(result);
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
  return { results, hadErrors };
}
