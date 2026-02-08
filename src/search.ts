/**
 * Vector and hybrid search for semantic memory
 * Uses usearch HNSW index for O(log n) approximate nearest neighbor search
 */

import { existsSync, readFileSync, writeFileSync, mkdirSync, unlinkSync } from "node:fs";
import { createHash } from "node:crypto";
import { dirname } from "node:path";
import { Index, MetricKind, ScalarKind } from "usearch";
import winston from "winston";
import type { EmbeddingProvider, MemorySearchResult, SemanticMemoryConfig } from "./types.js";
import { existsSync, readFileSync, writeFileSync, mkdirSync } from "fs";
import { dirname } from "path";

const log = winston.createLogger({
  level: "debug",
  format: winston.format.combine(
    winston.format.timestamp({ format: "HH:mm:ss.SSS" }),
    winston.format.printf(({ timestamp, level, message }) =>
      `${timestamp} [semantic-search] ${level}: ${message}`
    ),
  ),
  transports: [new winston.transports.Console()],
});

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
  getEntry(id: string): VectorEntry | undefined;
}

/** Sidecar file: maps HNSW label (array index) → full entry metadata.
 *  Self-contained — no SQLite needed for reconstruction on load. */
interface HnswMapEntryMeta {
  id: string;
  path: string;
  source: string;
  snippet: string;
  content: string;
}

interface HnswMapFile {
  dims: number;
  entries: (HnswMapEntryMeta | null)[];
}

/**
 * Create a semantic search manager
 * Vectors are kept in-memory; HNSW binary is persisted to disk after first build.
 *
 * @param chunkMetadata - Map of chunk ID → VectorEntry (metadata only, embeddings may be empty).
 *   Used to reconstruct the metadata Map when loading HNSW from disk.
 * @param hnswPathOrFn - Static path or lazy resolver for HNSW persistence.
 */
export async function createSemanticSearchManager(
  config: SemanticMemoryConfig,
  embeddingProvider: EmbeddingProvider,
  keywordSearchFn?: (query: string, limit: number) => Promise<HybridKeywordResult[]>,
  chunkMetadata?: Map<string, VectorEntry>,
  hnswPathOrFn?: string | (() => string | undefined),
): Promise<SemanticSearchManager> {
  // HNSW index + metadata maps (replaces brute-force vectors[] array)
  const metadata = new Map<bigint, VectorEntry>();   // label → full entry
  const idToLabel = new Map<string, bigint>();        // string ID → numeric label
  const existingIds = new Set<string>();
  let nextLabel = 0n;

  /** Resolve the HNSW path (may be lazy) */
  const resolveHnswPath = (): string | undefined =>
    typeof hnswPathOrFn === 'function' ? hnswPathOrFn() : hnswPathOrFn;

  // ── Save helper ──────────────────────────────────────────────────────
  const saveIndex = (): void => {
    const hnswPath = resolveHnswPath();
    if (!hnswPath) return;
    const mapPath = `${hnswPath}.map.json`;
    try {
      mkdirSync(dirname(hnswPath), { recursive: true });
      index.save(hnswPath);

      const entries: (HnswMapEntryMeta | null)[] = [];
      for (let i = 0n; i < nextLabel; i++) {
        const entry = metadata.get(i);
        if (entry) {
          entries.push({
            id: entry.id,
            path: entry.path,
            source: entry.source,
            snippet: entry.snippet,
            content: entry.content,
          });
        } else {
          entries.push(null);
        }
      }
      const map: HnswMapFile = { dims: index.dimensions(), entries };
      writeFileSync(mapPath, JSON.stringify(map));
      log.info(`Saved HNSW index to disk: ${metadata.size} vectors, ${hnswPath}`);
    } catch (err) {
      log.warn(`Failed to save HNSW index: ${err instanceof Error ? err.message : err}`);
    }
  };

  // ── Determine dimensions ─────────────────────────────────────────────
  // Probe the current provider to get actual embedding dimensions
  let dims = 1536; // fallback: OpenAI text-embedding-3-small
  let savedDims: number | undefined;
  {
    const initPath = resolveHnswPath();
    const initMapPath = initPath ? `${initPath}.map.json` : undefined;
    if (initMapPath && existsSync(initMapPath)) {
      try {
        const saved = JSON.parse(readFileSync(initMapPath, "utf-8")) as HnswMapFile;
        if (saved.dims) savedDims = saved.dims;
      } catch { /* will fall through to rebuild */ }
    }

    // Probe provider for actual dims (short test string)
    try {
      const probe = await embeddingProvider.embedQuery("dimension probe");
      if (probe.length > 0) dims = probe.length;
    } catch {
      // Provider may not be ready yet; fall back to saved dims
      if (savedDims) dims = savedDims;
    }

    // Detect provider change (dimension mismatch with saved index)
    if (savedDims && savedDims !== dims && initPath && initMapPath) {
      log.warn(
        `Dimension mismatch: saved=${savedDims}, current=${dims}. ` +
        `Provider changed? Deleting old HNSW, will rebuild from events.`
      );
      try { unlinkSync(initPath); } catch {}
      try { unlinkSync(initMapPath); } catch {}
      savedDims = undefined; // Force fresh start
    }
  }

  const index = new Index({
    metric: MetricKind.Cos,
    dimensions: dims,
    connectivity: 16,
    quantization: ScalarKind.F32,
    expansion_add: 128,
    expansion_search: 64,
    multi: false,
  });

  // ── Try loading from disk ────────────────────────────────────────────
  let loadedFromDisk = false;

  {
    const initPath = resolveHnswPath();
    const initMapPath = initPath ? `${initPath}.map.json` : undefined;

    if (initPath && initMapPath && existsSync(initPath) && existsSync(initMapPath)) {
      try {
        const saved = JSON.parse(readFileSync(initMapPath, "utf-8"));

        // Must have entries array — old format (ids-only) gets nuked
        if (!Array.isArray(saved.entries)) {
          throw new Error("old map format (no entries), deleting and rebuilding");
        }

        index.load(initPath);

        const map = saved as HnswMapFile;
        if (index.size() !== map.entries.length) {
          throw new Error(`size mismatch: index=${index.size()}, map=${map.entries.length}`);
        }

        let matched = 0;
        let orphaned = 0;
        for (let i = 0; i < map.entries.length; i++) {
          const e = map.entries[i];
          if (!e) { orphaned++; continue; }
          const label = BigInt(i);
          metadata.set(label, {
            id: e.id,
            path: e.path,
            startLine: 0,
            endLine: 0,
            source: e.source,
            snippet: e.snippet,
            content: e.content,
            embedding: [],
          });
          idToLabel.set(e.id, label);
          existingIds.add(e.id);
          matched++;
        }
        nextLabel = BigInt(map.entries.length);
        loadedFromDisk = true;

        log.info(
          `Loaded HNSW from disk: ${matched} matched, ${orphaned} orphaned, ` +
          `${index.size()}-node graph`
        );
      } catch (err) {
        log.warn(
          `Failed to load HNSW from disk, rebuilding: ${err instanceof Error ? err.message : err}`
        );
        // Delete stale files so we start clean
        try { unlinkSync(initPath); } catch {}
        try { unlinkSync(initMapPath); } catch {}
        metadata.clear();
        idToLabel.clear();
        existingIds.clear();
        nextLabel = 0n;
        loadedFromDisk = false;
      }
    }
  }

  // ── No HNSW on disk: start empty ────────────────────────────────────
  // Vectors arrive via addEntry / addEntriesBatch (memory:filesChanged events).
  // On a fresh install the HNSW will be built from those events and persisted.
  if (!loadedFromDisk) {
    log.info("No persisted HNSW found — starting with empty index. Vectors arrive via events.");
  }

  log.info(
    `HNSW ready: dims=${index.dimensions()}, vectors=${metadata.size}, ` +
    `connectivity=${index.connectivity()}, persisted=${!!resolveHnswPath()}`
  );

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

    const cacheKey = createHash('sha256').update(truncatedText).digest('hex').slice(0, 24);
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
    if (metadata.size === 0) {
      log.debug("vectorSearch called on empty index, returning []");
      return [];
    }

    // Clamp limit to index size (usearch throws if k > size)
    const k = Math.min(limit, metadata.size);
    const t0 = performance.now();
    const results = index.search(new Float32Array(queryEmbedding), k, 0);
    const elapsed = (performance.now() - t0).toFixed(2);

    const scored: HybridVectorResult[] = [];
    for (let i = 0; i < results.keys.length; i++) {
      const entry = metadata.get(results.keys[i]);
      if (!entry) continue;
      scored.push({
        id: entry.id,
        path: entry.path,
        startLine: entry.startLine,
        endLine: entry.endLine,
        source: entry.source,
        snippet: entry.snippet,
        content: entry.content,
        vectorScore: 1 - results.distances[i],  // cosine distance → similarity
      });
    }

    log.debug(`HNSW search: k=${k}, returned=${scored.length}, took=${elapsed}ms`);
    return scored;
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
      const full: VectorEntry = { ...entry, embedding };
      const label = nextLabel++;
      index.add(label, new Float32Array(embedding));
      metadata.set(label, full);
      idToLabel.set(entry.id, label);
      existingIds.add(entry.id);
      log.debug(`Added entry id=${entry.id} as label=${label}, index size=${metadata.size}`);
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

        log.info(`Embedding batch ${batchNum}: ${batchTexts.length} chunks, ~${batchTokens} tokens`);

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
              log.warn(`Rate limited, waiting ${waitSecs}s before retry ${retries + 1}/${maxRetries}`);
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

        if (embeddings.length !== batchEntries.length) {
          log.warn(`Embedding batch size mismatch: expected ${batchEntries.length}, got ${embeddings.length}`);
        }

        batchStart = batchEnd;

        const t0 = performance.now();
        for (let j = 0; j < batchEntries.length; j++) {
          const { entry } = batchEntries[j];
          const embedding = embeddings[j];
          if (!embedding || embedding.length !== dims) {
            log.warn(`Skipping entry ${entry.id}: expected ${dims}-dim embedding, got ${embedding?.length ?? 0}`);
            continue;
          }
          const full: VectorEntry = { ...entry, embedding };
          const label = nextLabel++;
          index.add(label, new Float32Array(embedding));
          metadata.set(label, full);
          idToLabel.set(entry.id, label);
          existingIds.add(entry.id);
          addedCount++;
        }
        const indexMs = (performance.now() - t0).toFixed(1);
        log.info(`Batch ${batchNum} indexed ${batchEntries.length} vectors into HNSW in ${indexMs}ms`);
      }

      log.info(`addEntriesBatch complete: ${addedCount} added, index size=${metadata.size}`);
      if (addedCount > 0) saveIndex();
      return addedCount;
    },

    async close(): Promise<void> {
      saveIndex();
      embeddingCache.clear();
      log.info(`Closed: saved index, cleared cache, final size=${metadata.size}`);
    },

    getEntryCount(): number {
      return metadata.size;
    },

    hasEntry(id: string): boolean {
      return existingIds.has(id);
    },

    getEntry(id: string): VectorEntry | undefined {
      const label = idToLabel.get(id);
      return label !== undefined ? metadata.get(label) : undefined;
    },
  };
}
