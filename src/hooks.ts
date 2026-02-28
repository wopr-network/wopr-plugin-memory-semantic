import { extractFromConversation } from "./capture.js";
import { multiScaleChunk } from "./chunking.js";
import type { EmbeddingQueue, PendingEntry } from "./embedding-queue.js";
import { contentHash } from "./manifest.js";
import { performAutoRecall } from "./recall.js";
import type { SemanticSearchManager } from "./search.js";
import type { SemanticMemoryConfig } from "./types.js";

interface HookLogger {
  info(msg: string): void;
  error(msg: string): void;
}

interface HookState {
  initialized: boolean;
  searchManager: SemanticSearchManager | null;
  config: SemanticMemoryConfig;
  instanceId: string | undefined;
}

/**
 * Before inject hook - auto-recall relevant memories
 */
export async function handleBeforeInject(state: HookState, log: HookLogger, payload: any): Promise<void> {
  if (!state.initialized || !state.searchManager || !state.config.autoRecall.enabled) {
    return;
  }

  // Payload is SessionInjectEvent: { session, message, from, channel? }
  // Not the expected messages array - skip for now until payload interface is resolved
  if (!payload || typeof payload.message !== "string" || !payload.message.trim()) {
    return;
  }

  const lastUserMessage = { role: "user", content: payload.message };

  try {
    const recall = await performAutoRecall(
      lastUserMessage.content,
      state.searchManager,
      state.config,
      state.instanceId,
    );

    if (recall && recall.memories.length > 0) {
      log.info(`[semantic-memory] Recalled ${recall.memories.length} memories (queryLen=${recall.query.length})`);
      // Prepend memory context to the mutable message payload
      // Core uses emitMutableIncoming → payload.message is mutable
      payload.message = `${recall.context}\n\n${payload.message}`;
    }
  } catch (err) {
    log.error(`[semantic-memory] Auto-recall failed: ${err instanceof Error ? err.message : String(err)}`);
  }
}

/**
 * After inject hook - real-time indexing of ALL session content
 * Payload is SessionResponseEvent: { session, message, response, from }
 */
export async function handleAfterInject(
  state: HookState,
  log: HookLogger,
  embeddingQueue: EmbeddingQueue,
  payload: any,
): Promise<void> {
  if (!state.initialized || !state.searchManager) {
    return;
  }

  // Validate payload structure
  if (!payload || typeof payload.response !== "string" || !payload.response.trim()) {
    return;
  }

  const sessionName = payload.session || "unknown";
  let indexedCount = 0;

  try {
    const ms = state.config.chunking.multiScale;

    // Helper: index a text with optional multi-scale — enqueues through the serialized queue
    const indexText = (text: string, baseId: string, source: string) => {
      const entries: PendingEntry[] = [];
      if (ms?.enabled && ms.scales.length > 0) {
        const subChunks = multiScaleChunk(
          text,
          baseId,
          { path: `session:${sessionName}`, startLine: 0, endLine: 0, source, instanceId: state.instanceId },
          ms.scales,
        );
        for (const sc of subChunks) {
          entries.push({ ...sc, persist: true });
        }
      } else {
        entries.push({
          entry: {
            id: baseId,
            path: `session:${sessionName}`,
            startLine: 0,
            endLine: 0,
            source,
            snippet: text.slice(0, 500),
            content: text,
            instanceId: state.instanceId,
          },
          text,
          persist: true,
        });
      }
      if (entries.length > 0) {
        embeddingQueue.enqueue(entries, `realtime:${source}`);
        indexedCount += entries.length;
      }
    };

    // REAL-TIME INDEXING: Index session content immediately with full text
    // Include session name in hash to prevent cross-session collisions
    if (payload.message && payload.message.trim().length > 10) {
      indexText(payload.message, `rt-${contentHash(`${sessionName}:user:${payload.message}`)}`, "realtime-user");
    }

    if (payload.response.trim().length > 10) {
      indexText(
        payload.response,
        `rt-${contentHash(`${sessionName}:assistant:${payload.response}`)}`,
        "realtime-assistant",
      );
    }

    if (indexedCount > 0) {
      log.info(`[semantic-memory] Real-time indexed ${indexedCount} entries from session ${sessionName}`);
    }

    // ALSO run capture analysis for important content (if enabled)
    if (state.config.autoCapture.enabled) {
      const messages = [
        { role: "user" as const, content: payload.message || "" },
        { role: "assistant" as const, content: payload.response },
      ];

      const candidates = extractFromConversation(messages, state.config);

      if (candidates.length > 0) {
        log.info(`[semantic-memory] Found ${candidates.length} capture candidates`);

        const captureEntries: PendingEntry[] = candidates.map((candidate) => ({
          entry: {
            id: `cap-${contentHash(candidate.text)}`,
            path: `session:${sessionName}`,
            startLine: 0,
            endLine: 0,
            source: "auto-capture",
            snippet: candidate.text.slice(0, 500),
            content: candidate.text,
            instanceId: state.instanceId,
          },
          text: candidate.text,
          persist: true,
        }));
        embeddingQueue.enqueue(captureEntries, `auto-capture(${candidates.length})`);

        log.info(`[semantic-memory] Queued ${candidates.length} capture memories`);
      }
    }
  } catch (err) {
    log.error(`[semantic-memory] Real-time indexing failed: ${err instanceof Error ? err.message : String(err)}`);
  }
}
