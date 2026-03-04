import { multiScaleChunk } from "./chunking.js";
import type { EmbeddingQueue, PendingEntry } from "./embedding-queue.js";
import type { SemanticSearchManager } from "./search.js";
import type { SemanticMemoryConfig } from "./types.js";

interface EventLogger {
  info(msg: string): void;
  error(msg: string): void;
}

interface FilesChangedState {
  initialized: boolean;
  searchManager: SemanticSearchManager | null;
  config: SemanticMemoryConfig;
  instanceId: string | undefined;
}

interface SearchState {
  initialized: boolean;
  searchManager: SemanticSearchManager | null;
  instanceId: string | undefined;
}

interface ContextLogger {
  info(msg: string): void;
  error(msg: string): void;
  debug?: (msg: string) => void;
}

/** Handle memory:filesChanged — index new/updated file chunks via the embedding queue */
export async function handleFilesChanged(
  state: FilesChangedState,
  log: EventLogger,
  embeddingQueue: EmbeddingQueue,
  payload: any,
): Promise<void> {
  if (!state.initialized || !state.searchManager) return;
  if (embeddingQueue.bootstrapping) {
    log.info(`filesChanged: skipped (bootstrap in progress)`);
    return;
  }

  const changes = payload.changes || [];
  const entries: PendingEntry[] = [];
  const ms = state.config.chunking.multiScale;

  for (const change of changes) {
    if (change.action === "delete") continue;
    if (!change.chunks) continue;

    for (const chunk of change.chunks) {
      if (!chunk.text || chunk.text.trim().length < 10) continue;
      const id = chunk.id;

      if (ms?.enabled && ms.scales.length > 0) {
        const subChunks = multiScaleChunk(
          chunk.text,
          id,
          {
            path: change.absPath || change.path,
            startLine: chunk.startLine,
            endLine: chunk.endLine,
            source: change.source || "memory",
            instanceId: state.instanceId,
          },
          ms.scales,
        );
        for (const sc of subChunks) entries.push(sc);
      } else {
        entries.push({
          entry: {
            id,
            path: change.absPath || change.path,
            startLine: chunk.startLine,
            endLine: chunk.endLine,
            source: change.source || "memory",
            snippet: chunk.text.slice(0, 500),
            content: chunk.text,
            instanceId: state.instanceId,
          },
          text: chunk.text,
        });
      }
    }
  }

  if (entries.length > 0) {
    embeddingQueue.enqueue(entries, `filesChanged(${changes.length} files)`);
  }
}

/** Handle memory:search — provide semantic search results */
export async function handleMemorySearch(
  state: SearchState,
  log: ContextLogger,
  payload: {
    query: string;
    maxResults: number;
    minScore: number;
    sessionName: string;
    results: any[] | null;
  },
): Promise<void> {
  const queryPreview = payload.query.length > 60 ? `${payload.query.slice(0, 60)}…` : payload.query;
  log.debug?.(
    `[semantic-memory] memory:search handler called (query length=${payload.query.length}): "${queryPreview}"`,
  );

  if (!state.initialized || !state.searchManager) {
    log.info(`[semantic-memory] Not initialized, skipping (initialized=${state.initialized})`);
    return;
  }

  try {
    log.info(`[semantic-memory] Starting semantic search (instanceId=${state.instanceId ?? "none"})...`);
    const results = await state.searchManager.search(payload.query, payload.maxResults, state.instanceId);
    log.info(`[semantic-memory] Raw results: ${results.length}`);
    payload.results = results.filter((r) => r.score >= payload.minScore);
    log.info(`[semantic-memory] After filter: ${payload.results.length} results (minScore: ${payload.minScore})`);
  } catch (err) {
    log.error(`[semantic-memory] Search failed: ${err instanceof Error ? err.message : String(err)}`);
  }
}
