/**
 * Vector and hybrid search for semantic memory
 * With JSON file persistence for vector storage
 */

import type { EmbeddingProvider, MemorySearchResult, SemanticMemoryConfig } from "./types.js";
import { existsSync, readFileSync, writeFileSync, mkdirSync } from "fs";
import { dirname } from "path";

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
  content: string;      // Full indexed text for retrieval
  vectorScore: number;
};

export type HybridKeywordResult = {
  id: string;
  path: string;
  startLine: number;
  endLine: number;
  source: string;
  snippet: string;
  content: string;      // Full indexed text for retrieval
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
      content: string;
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
      content: r.content,
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
      if (r.content && r.content.length > 0) {
        existing.content = r.content;
      }
    } else {
      byId.set(r.id, {
        id: r.id,
        path: r.path,
        startLine: r.startLine,
        endLine: r.endLine,
        source: r.source,
        snippet: r.snippet,
        content: r.content,
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
      content: entry.content,
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
// Persistence Layer
// =============================================================================

// Store data inside the plugin directory
const PLUGIN_DATA_DIR = "/data/plugins/wopr-plugin-memory-semantic/data";
const VECTOR_STORE_PATH = `${PLUGIN_DATA_DIR}/vectors.json`;
const MTIME_STORE_PATH = `${PLUGIN_DATA_DIR}/mtimes.json`;

interface PersistedVectorStore {
  version: number;
  entries: VectorEntry[];
  lastSaved: number;
}

interface PersistedMtimeStore {
  version: number;
  mtimes: Record<string, number>; // path -> mtime
  lastSaved: number;
}

function ensureDir(filePath: string): void {
  const dir = dirname(filePath);
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }
}

function loadVectorStore(): VectorEntry[] {
  try {
    if (existsSync(VECTOR_STORE_PATH)) {
      const data = readFileSync(VECTOR_STORE_PATH, "utf-8");
      const store: PersistedVectorStore = JSON.parse(data);
      if (store.version === 1 && Array.isArray(store.entries)) {
        return store.entries;
      }
    }
  } catch (err) {
    console.error("[semantic-memory] Failed to load vector store:", err);
  }
  return [];
}

function saveVectorStore(entries: VectorEntry[]): void {
  try {
    ensureDir(VECTOR_STORE_PATH);
    const store: PersistedVectorStore = {
      version: 1,
      entries,
      lastSaved: Date.now(),
    };
    writeFileSync(VECTOR_STORE_PATH, JSON.stringify(store));
  } catch (err) {
    console.error("[semantic-memory] Failed to save vector store:", err);
  }
}

export function loadMtimeStore(): Map<string, number> {
  try {
    if (existsSync(MTIME_STORE_PATH)) {
      const data = readFileSync(MTIME_STORE_PATH, "utf-8");
      const store: PersistedMtimeStore = JSON.parse(data);
      if (store.version === 1 && store.mtimes) {
        return new Map(Object.entries(store.mtimes));
      }
    }
  } catch (err) {
    console.error("[semantic-memory] Failed to load mtime store:", err);
  }
  return new Map();
}

export function saveMtimeStore(mtimes: Map<string, number>): void {
  try {
    ensureDir(MTIME_STORE_PATH);
    const store: PersistedMtimeStore = {
      version: 1,
      mtimes: Object.fromEntries(mtimes),
      lastSaved: Date.now(),
    };
    writeFileSync(MTIME_STORE_PATH, JSON.stringify(store));
  } catch (err) {
    console.error("[semantic-memory] Failed to save mtime store:", err);
  }
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
  content: string;      // Full indexed text for retrieval
  embedding: number[];
}

export interface SemanticSearchManager {
  search(query: string, maxResults?: number): Promise<MemorySearchResult[]>;
  addEntry(entry: Omit<VectorEntry, "embedding">, text: string): Promise<void>;
  addEntriesBatch(entries: Array<{ entry: Omit<VectorEntry, "embedding">; text: string }>): Promise<number>;
  close(): Promise<void>;
  getEntryCount(): number;
  hasEntry(id: string): boolean;
}

/**
 * Create a semantic search manager
 * Uses JSON file persistence for vector storage
 */
export async function createSemanticSearchManager(
  config: SemanticMemoryConfig,
  embeddingProvider: EmbeddingProvider,
  keywordSearchFn?: (query: string, limit: number) => Promise<HybridKeywordResult[]>
): Promise<SemanticSearchManager> {
  // Load persisted vectors
  const vectors: VectorEntry[] = loadVectorStore();
  const existingIds = new Set(vectors.map(v => v.id));

  console.log(`[semantic-memory] Loaded ${vectors.length} vectors from persistent storage`);

  // Embedding cache
  const embeddingCache = new Map<string, number[]>();

  // Track if we need to save
  let dirty = false;
  let saveTimeout: NodeJS.Timeout | null = null;

  // Debounced save - wait 5 seconds after last change before saving
  const scheduleSave = () => {
    dirty = true;
    if (saveTimeout) {
      clearTimeout(saveTimeout);
    }
    saveTimeout = setTimeout(() => {
      if (dirty) {
        saveVectorStore(vectors);
        dirty = false;
      }
    }, 5000);
  };

  const getEmbedding = async (text: string): Promise<number[]> => {
    // Truncate text to ~4000 chars to stay safely under 8192 token limit
    const truncatedText = text.length > 4000 ? text.slice(0, 4000) : text;

    const cacheKey = truncatedText.slice(0, 200);
    const cached = embeddingCache.get(cacheKey);
    if (cached) return cached;

    const embedding = await embeddingProvider.embedQuery(truncatedText);
    if (config.cache.enabled) {
      embeddingCache.set(cacheKey, embedding);
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

      const queryEmbedding = await getEmbedding(query);
      const vectorResults = await vectorSearch(queryEmbedding, candidateLimit);

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
            content: r.content,
            source: r.source,
          }));
      }

      const keywordResults = await keywordSearchFn(query, candidateLimit);
      const merged = mergeHybridResults({
        vector: vectorResults,
        keyword: keywordResults,
        vectorWeight: config.hybrid.vectorWeight,
        textWeight: config.hybrid.textWeight,
      });

      return merged.filter((r) => r.score >= config.search.minScore).slice(0, limit);
    },

    async addEntry(entry: Omit<VectorEntry, "embedding">, text: string): Promise<void> {
      // Skip if already indexed
      if (existingIds.has(entry.id)) {
        return;
      }

      const embedding = await getEmbedding(text);
      vectors.push({ ...entry, embedding });
      existingIds.add(entry.id);
      scheduleSave();
    },

    async addEntriesBatch(entries: Array<{ entry: Omit<VectorEntry, "embedding">; text: string }>): Promise<number> {
      // Filter out already indexed entries
      const newEntries = entries.filter(e => !existingIds.has(e.entry.id));
      if (newEntries.length === 0) return 0;

      // Truncate texts to stay under token limit (~4000 chars = ~1000 tokens, safe under 8192)
      const truncatedTexts = newEntries.map(e =>
        e.text.length > 4000 ? e.text.slice(0, 4000) : e.text
      );

      // OpenAI limit: 300,000 tokens per request, ~4 chars per token
      const MAX_TOKENS_PER_REQUEST = 280000; // Leave margin below 300k
      const CHARS_PER_TOKEN = 4;
      let addedCount = 0;

      // Build batches based on actual token estimates
      let batchStart = 0;
      let batchNum = 0;
      while (batchStart < truncatedTexts.length) {
        let batchTokens = 0;
        let batchEnd = batchStart;

        // Add texts until we hit the token limit
        while (batchEnd < truncatedTexts.length) {
          const textTokens = Math.ceil(truncatedTexts[batchEnd].length / CHARS_PER_TOKEN);
          if (batchTokens + textTokens > MAX_TOKENS_PER_REQUEST && batchEnd > batchStart) {
            break; // This text would exceed limit, stop here
          }
          batchTokens += textTokens;
          batchEnd++;
        }

        const batchTexts = truncatedTexts.slice(batchStart, batchEnd);
        const batchEntries = newEntries.slice(batchStart, batchEnd);
        batchNum++;

        console.log(`[semantic-memory] Embedding batch ${batchNum} (${batchTexts.length} chunks, ~${batchTokens} tokens)...`);

        // Retry with exponential backoff on rate limits
        let embeddings: number[][] = [];
        let retries = 0;
        const maxRetries = 5;
        while (retries < maxRetries) {
          try {
            embeddings = await embeddingProvider.embedBatch(batchTexts);
            break;
          } catch (err: any) {
            const errMsg = err?.message || String(err);
            // Check for rate limit (429) errors
            if (errMsg.includes("429") || errMsg.includes("rate_limit") || errMsg.includes("Rate limit")) {
              // Extract wait time from error message if present
              const waitMatch = errMsg.match(/try again in (\d+\.?\d*)/i);
              const waitSecs = waitMatch ? Math.ceil(parseFloat(waitMatch[1])) + 1 : Math.pow(2, retries) * 5;
              console.log(`[semantic-memory] Rate limited, waiting ${waitSecs}s before retry ${retries + 1}/${maxRetries}...`);
              await new Promise(resolve => setTimeout(resolve, waitSecs * 1000));
              retries++;
            } else {
              throw err; // Non-rate-limit error, propagate
            }
          }
        }
        if (retries >= maxRetries) {
          throw new Error(`Failed after ${maxRetries} rate limit retries`);
        }

        batchStart = batchEnd;

        for (let j = 0; j < batchEntries.length; j++) {
          const { entry } = batchEntries[j];
          const embedding = embeddings[j] || [];

          vectors.push({ ...entry, embedding });
          existingIds.add(entry.id);
          addedCount++;
        }
      }

      if (addedCount > 0) {
        scheduleSave();
      }

      return addedCount;
    },

    async close(): Promise<void> {
      // Force save on close
      if (saveTimeout) {
        clearTimeout(saveTimeout);
      }
      if (dirty || vectors.length > 0) {
        saveVectorStore(vectors);
        console.log(`[semantic-memory] Final save: ${vectors.length} vectors`);
      }
      embeddingCache.clear();
    },

    getEntryCount(): number {
      return vectors.length;
    },

    hasEntry(id: string): boolean {
      return existingIds.has(id);
    },
  };
}
