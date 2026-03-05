/**
 * Types for semantic memory plugin
 */

export interface SemanticMemoryConfig {
  // Embedding provider
  provider: "openai" | "gemini" | "local" | "ollama" | "auto";
  model: string;

  // API configuration
  /** @deprecated Ignored at runtime. Use environment variables instead (OPENAI_API_KEY, GOOGLE_API_KEY, GEMINI_API_KEY). */
  apiKey?: string;
  baseUrl?: string;

  // Local model config (for node-llama-cpp)
  local?: {
    modelPath?: string;
    modelCacheDir?: string;
  };

  // Ollama config
  ollama?: {
    baseUrl?: string; // default: http://ollama:11434 (Docker) or http://localhost:11434
    model?: string; // default: qwen3-embedding:8b
  };

  // Search configuration
  search: {
    maxResults: number;
    minScore: number;
    candidateMultiplier: number;
    /** When true, tenant-scoped queries exclude entries with no instanceId (legacy/global). Default: true. */
    excludeLegacyEntries: boolean;
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
    /** @deprecated unused; storage is derived from api.storage/WOPR_HOME */
    path?: string;
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
    multiScale?: {
      enabled: boolean;
      scales: Array<{ tokens: number; overlap: number }>;
    };
  };

  // Sync configuration
  sync?: {
    watch?: boolean;
    watchDebounceMs?: number;
    indexSessions?: boolean;
  };

  /** Instance ID for multi-tenant isolation. Set by the daemon per bot instance. */
  instanceId?: string;

  /** Maximum byte size for a single memory_write call. Default: 1_048_576 (1 MB). */
  maxWriteBytes?: number;
}

export const DEFAULT_CONFIG: SemanticMemoryConfig = {
  provider: "auto",
  model: "text-embedding-3-small",
  search: {
    maxResults: 10,
    minScore: 0.3,
    candidateMultiplier: 3,
    excludeLegacyEntries: true,
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
    multiScale: {
      enabled: true,
      scales: [
        { tokens: 512, overlap: 64 },
        { tokens: 2048, overlap: 256 },
        { tokens: 4096, overlap: 512 },
      ],
    },
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
  content: string; // Full indexed text for retrieval
  source: string;
  /** Instance ID that owns this memory entry */
  instanceId?: string;
}

export interface CaptureCandidate {
  text: string;
  category: MemoryCategory;
  importance: number;
}

// Session API types (mirrors WOP-1538 SessionApi from core)
// Remove once @wopr-network/plugin-types is republished with SessionApi

export interface ConversationEntry {
  ts: number;
  from: string;
  senderId?: string;
  content: string;
  type: "context" | "message" | "response" | "middleware";
  channel?: { id: string; type: string; name?: string };
}

export interface SessionApi {
  getContext(sessionName: string, filename: string): Promise<string | null>;
  setContext(sessionName: string, filename: string, content: string, source: "global" | "session"): Promise<void>;
  readConversationLog(sessionName: string, limit?: number): Promise<ConversationEntry[]>;
}
