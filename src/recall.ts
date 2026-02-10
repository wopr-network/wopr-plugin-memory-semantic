/**
 * Auto-recall logic for injecting relevant memories before agent processing
 */

import type { SemanticSearchManager } from "./search.js";
import type { MemorySearchResult, SemanticMemoryConfig } from "./types.js";

// =============================================================================
// Query Extraction
// =============================================================================

/**
 * Extract a search query from a user message
 * Focuses on the key information while filtering noise
 */
export function extractQueryFromMessage(message: string): string {
  // Remove common conversational prefixes
  const query = message.replace(/^(hey|hi|hello|please|can you|could you|would you|i need|i want)\s+/gi, "").trim();

  // If message is very short, use as-is
  if (query.length < 50) {
    return query;
  }

  // For longer messages, try to extract the key question/topic
  // Look for question patterns
  const questionMatch = query.match(
    /(?:what|how|why|where|when|who|which|can|could|would|should|is|are|do|does)[^.?!]*[?]/i,
  );
  if (questionMatch) {
    return questionMatch[0];
  }

  // Otherwise take first sentence or first N characters
  const firstSentence = query.match(/^[^.!?]+[.!?]?/);
  if (firstSentence && firstSentence[0].length >= 20) {
    return firstSentence[0];
  }

  // Fallback: truncate
  return query.slice(0, 200);
}

// =============================================================================
// Memory Formatting
// =============================================================================

/**
 * Format search results as context to inject into the conversation
 */
export function formatMemoriesAsContext(memories: MemorySearchResult[], config: SemanticMemoryConfig): string {
  if (memories.length === 0) {
    return "";
  }

  const lines = ["<relevant-memories>"];

  for (const mem of memories.slice(0, config.autoRecall.maxMemories)) {
    const scorePercent = Math.round(mem.score * 100);
    lines.push(`[${scorePercent}%] ${mem.snippet}`);
    if (mem.path) {
      lines.push(`  (from: ${mem.path}:${mem.startLine})`);
    }
  }

  lines.push("</relevant-memories>");
  return lines.join("\n");
}

// =============================================================================
// Auto-Recall Handler
// =============================================================================

export interface RecallResult {
  query: string;
  memories: MemorySearchResult[];
  context: string;
}

/**
 * Perform auto-recall for an incoming message
 */
export async function performAutoRecall(
  message: string,
  searchManager: SemanticSearchManager,
  config: SemanticMemoryConfig,
): Promise<RecallResult | null> {
  if (!config.autoRecall.enabled) {
    return null;
  }

  // Extract search query from message
  const query = extractQueryFromMessage(message);
  if (!query || query.length < 3) {
    return null;
  }

  // Search for relevant memories
  const memories = await searchManager.search(query, config.autoRecall.maxMemories);

  // Filter by minimum score
  const relevant = memories.filter((m) => m.score >= config.autoRecall.minScore);

  if (relevant.length === 0) {
    return null;
  }

  // Format as context
  const context = formatMemoriesAsContext(relevant, config);

  return {
    query,
    memories: relevant,
    context,
  };
}

/**
 * Inject memories into a message array
 * Adds a system-style context message with relevant memories
 */
export function injectMemoriesIntoMessages(
  messages: Array<{ role: string; content: string }>,
  memories: RecallResult,
): Array<{ role: string; content: string }> {
  if (!memories.context) {
    return messages;
  }

  // Find the last user message and inject context before it
  const result = [...messages];
  let lastUserIndex = -1;
  for (let i = result.length - 1; i >= 0; i--) {
    if (result[i].role === "user") {
      lastUserIndex = i;
      break;
    }
  }

  if (lastUserIndex === -1) {
    return messages;
  }

  // Insert memory context as a system-like message before the user message
  // Most providers will handle this gracefully
  result.splice(lastUserIndex, 0, {
    role: "user", // Use user role with clear delimiters
    content: `[Context from memory - use if relevant]\n${memories.context}`,
  });

  return result;
}
