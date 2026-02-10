/**
 * Tool Enhancer - hooks into memory_search to add vector capabilities
 *
 * This module provides a way to enhance the core memory_search tool
 * with semantic search capabilities without replacing it entirely.
 */

import type { SemanticSearchManager } from "./search.js";
import type { MemorySearchResult, SemanticMemoryConfig } from "./types.js";

export interface ToolEnhancerOptions {
  config: SemanticMemoryConfig;
  searchManager: SemanticSearchManager;
}

export interface SearchToolInput {
  query: string;
  maxResults?: number;
  minScore?: number;
}

export interface SearchToolResult {
  results: MemorySearchResult[];
  enhanced: boolean;
  source: "semantic" | "fallback";
}

/**
 * Enhance a memory search query with semantic search
 *
 * This wraps the core search functionality and adds vector-based results.
 * If vector search fails, it falls back gracefully.
 */
export async function enhanceSearch(
  input: SearchToolInput,
  options: ToolEnhancerOptions,
  fallbackSearch?: (query: string, limit: number) => Promise<MemorySearchResult[]>,
): Promise<SearchToolResult> {
  const { query, maxResults = 10, minScore = 0.35 } = input;
  const { config, searchManager } = options;

  try {
    // Perform semantic search
    const results = await searchManager.search(query, maxResults);

    // Filter by min score
    const filtered = results.filter((r) => r.score >= minScore);

    if (filtered.length > 0) {
      return {
        results: filtered,
        enhanced: true,
        source: "semantic",
      };
    }

    // If no semantic results but we have a fallback, use it
    if (fallbackSearch) {
      const fallbackResults = await fallbackSearch(query, maxResults);
      return {
        results: fallbackResults,
        enhanced: false,
        source: "fallback",
      };
    }

    return {
      results: [],
      enhanced: true,
      source: "semantic",
    };
  } catch (err) {
    // Semantic search failed, use fallback if available
    if (fallbackSearch) {
      try {
        const fallbackResults = await fallbackSearch(query, maxResults);
        return {
          results: fallbackResults,
          enhanced: false,
          source: "fallback",
        };
      } catch {
        // Both failed
        return {
          results: [],
          enhanced: false,
          source: "fallback",
        };
      }
    }

    // No fallback available
    throw err;
  }
}

/**
 * Create a tool handler that can be registered with WOPR's A2A MCP
 *
 * This replaces the default memory_search implementation with one that
 * uses semantic search.
 */
export function createSearchToolHandler(options: ToolEnhancerOptions) {
  return async (args: SearchToolInput): Promise<{ content: Array<{ type: string; text: string }> }> => {
    const { query, maxResults = 10, minScore = 0.35 } = args;

    try {
      const result = await enhanceSearch({ query, maxResults, minScore }, options);

      if (result.results.length === 0) {
        return {
          content: [{ type: "text", text: `No matches found for "${query}"` }],
        };
      }

      const formatted = result.results
        .map(
          (r, i) =>
            `[${i + 1}] ${r.source}/${r.path}:${r.startLine}-${r.endLine} (score: ${r.score.toFixed(2)})\n${r.snippet}`,
        )
        .join("\n\n---\n\n");

      const enhancedNote = result.enhanced ? " (semantic)" : " (keyword)";
      return {
        content: [
          {
            type: "text",
            text: `Found ${result.results.length} results${enhancedNote}:\n\n${formatted}`,
          },
        ],
      };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      return {
        content: [{ type: "text", text: `Search failed: ${message}` }],
      };
    }
  };
}
