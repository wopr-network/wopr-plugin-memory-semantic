/**
 * Console-based fallback logger for modules that have no access to ctx.
 * Used by embeddings.ts and search.ts before the plugin context is available.
 */

export const fallbackLogger = {
  // biome-ignore lint/suspicious/noConsole: intentional fallback logging before ctx is available
  debug: (msg: string) => console.debug(`[semantic-memory] ${msg}`),
  info: (msg: string) => console.info(`[semantic-memory] ${msg}`),
  warn: (msg: string) => console.warn(`[semantic-memory] ${msg}`),
  error: (msg: string) => console.error(`[semantic-memory] ${msg}`),
};
