/**
 * Vector and hybrid search for semantic memory
 */

import type { EmbeddingProvider, MemorySearchResult, SemanticMemoryConfig } from "./types.js";

// =============================================================================
// Hybrid Search Helpers (ported from WOPR core)
// =============================================================================

export type HybridVectorResult = {
  id: string;
  path: string;
  startLine: number;
  endLine: number;
  source: string;
  snippet: string;
  vectorScore: number;
};

export type HybridKeywordResult = {
  id: string;
  path: string;
  startLine: number;
  endLine: number;
  source: string;
  snippet: string;
  textScore: number;
};

export function buildFtsQuery(raw: string): string | null {
  const tokens =
    raw
      .match(/[A-Za-z0-9_]+/g)
      ?.map((t) => t.trim())
      .filter(Boolean) ?? [];
  if (tokens.length === 0) {
    return null;
  }
  const quoted = tokens.map((t) => `"${t.replaceAll('"', "")}"`);
  return quoted.join(" AND ");
}

export function bm25RankToScore(rank: number): number {
  const normalized = Number.isFinite(rank) ? Math.max(0, rank) : 999;
  return 1 / (1 + normalized);
}

export function mergeHybridResults(params: {
  vector: HybridVectorResult[];
  keyword: HybridKeywordResult[];
  vectorWeight: number;
  textWeight: number;
}): MemorySearchResult[] {
  const byId = new Map<
    string,
    {
      id: string;
      path: string;
      startLine: number;
      endLine: number;
      source: string;
      snippet: string;
      vectorScore: number;
      textScore: number;
    }
  >();

  for (const r of params.vector) {
    byId.set(r.id, {
      id: r.id,
      path: r.path,
      startLine: r.startLine,
      endLine: r.endLine,
      source: r.source,
      snippet: r.snippet,
      vectorScore: r.vectorScore,
      textScore: 0,
    });
  }

  for (const r of params.keyword) {
    const existing = byId.get(r.id);
    if (existing) {
      existing.textScore = r.textScore;
      if (r.snippet && r.snippet.length > 0) {
        existing.snippet = r.snippet;
      }
    } else {
      byId.set(r.id, {
        id: r.id,
        path: r.path,
        startLine: r.startLine,
        endLine: r.endLine,
        source: r.source,
        snippet: r.snippet,
        vectorScore: 0,
        textScore: r.textScore,
      });
    }
  }

  const merged = Array.from(byId.values()).map((entry) => {
    const score = params.vectorWeight * entry.vectorScore + params.textWeight * entry.textScore;
    return {
      path: entry.path,
      startLine: entry.startLine,
      endLine: entry.endLine,
      score,
      snippet: entry.snippet,
      source: entry.source,
    };
  });

  return merged.sort((a, b) => b.score - a.score);
}

// =============================================================================
// Cosine Similarity
// =============================================================================

export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;

  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom > 0 ? dot / denom : 0;
}

// =============================================================================
// Semantic Search Manager
// =============================================================================

export interface VectorEntry {
  id: string;
  path: string;
  startLine: number;
  endLine: number;
  source: string;
  snippet: string;
  embedding: number[];
}

export interface SemanticSearchManager {
  search(query: string, maxResults?: number): Promise<MemorySearchResult[]>;
  addEntry(entry: Omit<VectorEntry, "embedding">, text: string): Promise<void>;
  close(): Promise<void>;
}

/**
 * Create a semantic search manager
 * Uses in-memory vector storage with optional persistence
 */
export async function createSemanticSearchManager(
  config: SemanticMemoryConfig,
  embeddingProvider: EmbeddingProvider,
  keywordSearchFn?: (query: string, limit: number) => Promise<HybridKeywordResult[]>
): Promise<SemanticSearchManager> {
  // In-memory vector store
  const vectors: VectorEntry[] = [];

  // Embedding cache
  const embeddingCache = new Map<string, number[]>();

  const getEmbedding = async (text: string): Promise<number[]> => {
    const cacheKey = text.slice(0, 200); // Truncate for cache key
    const cached = embeddingCache.get(cacheKey);
    if (cached) return cached;

    const embedding = await embeddingProvider.embedQuery(text);
    if (config.cache.enabled) {
      embeddingCache.set(cacheKey, embedding);
      // Evict old entries if cache is full
      if (config.cache.maxEntries && embeddingCache.size > config.cache.maxEntries) {
        const firstKey = embeddingCache.keys().next().value;
        if (firstKey) embeddingCache.delete(firstKey);
      }
    }
    return embedding;
  };

  const vectorSearch = async (
    queryEmbedding: number[],
    limit: number
  ): Promise<HybridVectorResult[]> => {
    const scored = vectors.map((entry) => ({
      ...entry,
      vectorScore: cosineSimilarity(queryEmbedding, entry.embedding),
    }));

    return scored
      .sort((a, b) => b.vectorScore - a.vectorScore)
      .slice(0, limit)
      .map(({ embedding: _, ...rest }) => rest);
  };

  return {
    async search(query: string, maxResults?: number): Promise<MemorySearchResult[]> {
      const limit = maxResults ?? config.search.maxResults;
      const candidateLimit = limit * config.search.candidateMultiplier;

      // Get query embedding
      const queryEmbedding = await getEmbedding(query);

      // Vector search
      const vectorResults = await vectorSearch(queryEmbedding, candidateLimit);

      // If hybrid disabled or no keyword search function, return vector only
      if (!config.hybrid.enabled || !keywordSearchFn) {
        return vectorResults
          .filter((r) => r.vectorScore >= config.search.minScore)
          .slice(0, limit)
          .map((r) => ({
            path: r.path,
            startLine: r.startLine,
            endLine: r.endLine,
            score: r.vectorScore,
            snippet: r.snippet,
            source: r.source,
          }));
      }

      // Keyword search
      const keywordResults = await keywordSearchFn(query, candidateLimit);

      // Merge results
      const merged = mergeHybridResults({
        vector: vectorResults,
        keyword: keywordResults,
        vectorWeight: config.hybrid.vectorWeight,
        textWeight: config.hybrid.textWeight,
      });

      return merged.filter((r) => r.score >= config.search.minScore).slice(0, limit);
    },

    async addEntry(entry: Omit<VectorEntry, "embedding">, text: string): Promise<void> {
      const embedding = await getEmbedding(text);
      vectors.push({ ...entry, embedding });
    },

    async close(): Promise<void> {
      embeddingCache.clear();
      vectors.length = 0;
    },
  };
}
