// Ambient declaration for optional dependency node-llama-cpp.
// This package is not required at runtime; users who want local embeddings must install it separately.
declare module "node-llama-cpp" {
  export function getLlama(opts: { logLevel: number }): Promise<any>;
  export function resolveModelFile(path: string, cacheDir?: string): Promise<string>;
  export const LlamaLogLevel: { error: number };
}
