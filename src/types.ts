/**
 * Types for semantic memory plugin
 */

export interface SemanticMemoryConfig {
  // Embedding provider
  provider: "openai" | "gemini" | "local" | "auto";
  model: string;

  // API configuration
  apiKey?: string;
  baseUrl?: string;

  // Local model config (for node-llama-cpp)
  local?: {
    modelPath?: string;
    modelCacheDir?: string;
  };

  // Search configuration
  search: {
    maxResults: number;
    minScore: number;
    candidateMultiplier: number;
  };

  // Hybrid search weights
  hybrid: {
    enabled: boolean;
    vectorWeight: number;
    textWeight: number;
  };

  // Auto-recall configuration
  autoRecall: {
    enabled: boolean;
    maxMemories: number;
    minScore: number;
  };

  // Auto-capture configuration
  autoCapture: {
    enabled: boolean;
    maxPerConversation: number;
    minLength: number;
    maxLength: number;
  };

  // Storage
  store: {
    path?: string; // SQLite database path
    vectorEnabled: boolean;
  };

  // Caching
  cache: {
    enabled: boolean;
    maxEntries?: number;
  };

  // Chunking
  chunking: {
    tokens: number;
    overlap: number;
  };
}

export const DEFAULT_CONFIG: SemanticMemoryConfig = {
  provider: "auto",
  model: "text-embedding-3-small",
  search: {
    maxResults: 10,
    minScore: 0.3,
    candidateMultiplier: 3,
  },
  hybrid: {
    enabled: true,
    vectorWeight: 0.7,
    textWeight: 0.3,
  },
  autoRecall: {
    enabled: true,
    maxMemories: 5,
    minScore: 0.4,
  },
  autoCapture: {
    enabled: true,
    maxPerConversation: 3,
    minLength: 10,
    maxLength: 500,
  },
  store: {
    vectorEnabled: true,
  },
  cache: {
    enabled: true,
    maxEntries: 10000,
  },
  chunking: {
    tokens: 512,
    overlap: 64,
  },
};

export interface EmbeddingProvider {
  id: string;
  model: string;
  embedQuery: (text: string) => Promise<number[]>;
  embedBatch: (texts: string[]) => Promise<number[][]>;
}

export interface MemoryEntry {
  id: string;
  text: string;
  category: MemoryCategory;
  importance: number;
  createdAt: number;
  source: string;
}

export type MemoryCategory = "preference" | "decision" | "entity" | "fact" | "other";

export interface MemorySearchResult {
  path: string;
  startLine: number;
  endLine: number;
  score: number;
  snippet: string;
  content: string;      // Full indexed text for retrieval
  source: string;
}

export interface CaptureCandidate {
  text: string;
  category: MemoryCategory;
  importance: number;
}
