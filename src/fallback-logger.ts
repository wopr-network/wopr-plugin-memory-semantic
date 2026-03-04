/**
 * Console-based fallback logger for modules that have no access to ctx.
 * Used by embeddings.ts and search.ts before the plugin context is available.
 */

export const fallbackLogger = {
  debug: (msg: string, extra?: unknown) => {
    // biome-ignore lint/suspicious/noConsole: intentional fallback logging before ctx is available
    console.debug(`[semantic-memory] ${msg}`, ...(extra !== undefined ? [extra] : []));
  },
  info: (msg: string, extra?: unknown) => {
    console.info(`[semantic-memory] ${msg}`, ...(extra !== undefined ? [extra] : []));
  },
  warn: (msg: string, extra?: unknown) => {
    console.warn(`[semantic-memory] ${msg}`, ...(extra !== undefined ? [extra] : []));
  },
  error: (msg: string, extra?: unknown) => {
    console.error(`[semantic-memory] ${msg}`, ...(extra !== undefined ? [extra] : []));
  },
};
