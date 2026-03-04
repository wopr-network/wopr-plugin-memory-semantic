// Minimal ambient declaration for the optional runtime dependency `node-llama-cpp`.
// This lets TypeScript compile without error even when the package is not installed,
// while the actual exports are typed loosely (any) since the surface we use is small.
declare module "node-llama-cpp" {
  export const LlamaLogLevel: { error: number };
  export function getLlama(opts: { logLevel: number }): Promise<any>;
  export function resolveModelFile(path: string, cacheDir?: string): Promise<string>;
}
