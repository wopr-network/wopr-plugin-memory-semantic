export async function runWithConcurrency<T>(
  tasks: Array<() => Promise<T>>,
  concurrency: number,
  onError?: (err: unknown) => void,
): Promise<T[]> {
  const results: T[] = [];
  const executing: Set<Promise<void>> = new Set();

  for (const task of tasks) {
    const p = task()
      .then((result) => {
        results.push(result);
      })
      .catch((err) => {
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
  return results;
}
