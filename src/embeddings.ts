/**
 * Embedding providers for semantic memory
 * Supports OpenAI, Gemini, and local (node-llama-cpp)
 */

import type { EmbeddingProvider, SemanticMemoryConfig } from "./types.js";

// =============================================================================
// OpenAI Embeddings
// =============================================================================

const DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1";

export async function createOpenAiEmbeddingProvider(
  config: SemanticMemoryConfig
): Promise<EmbeddingProvider> {
  const apiKey = config.apiKey || process.env.OPENAI_API_KEY?.trim();
  if (!apiKey) {
    throw new Error("No API key found for OpenAI. Set OPENAI_API_KEY environment variable.");
  }

  const baseUrl = (config.baseUrl || DEFAULT_OPENAI_BASE_URL).replace(/\/$/, "");
  const model = config.model || "text-embedding-3-small";
  const url = `${baseUrl}/embeddings`;

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Authorization: `Bearer ${apiKey}`,
  };

  const embed = async (input: string[]): Promise<number[][]> => {
    if (input.length === 0) return [];

    const res = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify({ model, input }),
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`OpenAI embeddings failed: ${res.status} ${text}`);
    }

    const payload = (await res.json()) as {
      data?: Array<{ embedding?: number[] }>;
    };

    return (payload.data ?? []).map((entry) => entry.embedding ?? []);
  };

  return {
    id: "openai",
    model,
    embedQuery: async (text) => {
      const [vec] = await embed([text]);
      return vec ?? [];
    },
    embedBatch: embed,
  };
}

// =============================================================================
// Gemini Embeddings
// =============================================================================

const DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta";
const DEFAULT_GEMINI_MODEL = "gemini-embedding-001";

export async function createGeminiEmbeddingProvider(
  config: SemanticMemoryConfig
): Promise<EmbeddingProvider> {
  const apiKey =
    config.apiKey ||
    process.env.GOOGLE_API_KEY?.trim() ||
    process.env.GEMINI_API_KEY?.trim();

  if (!apiKey) {
    throw new Error(
      "No API key found for Gemini. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
    );
  }

  const baseUrl = (config.baseUrl || DEFAULT_GEMINI_BASE_URL).replace(/\/$/, "");
  const model = config.model || DEFAULT_GEMINI_MODEL;
  const modelPath = model.startsWith("models/") ? model : `models/${model}`;

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    "x-goog-api-key": apiKey,
  };

  const embedQuery = async (text: string): Promise<number[]> => {
    if (!text.trim()) return [];

    const url = `${baseUrl}/${modelPath}:embedContent`;
    const res = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify({
        content: { parts: [{ text }] },
        taskType: "RETRIEVAL_QUERY",
      }),
    });

    if (!res.ok) {
      const payload = await res.text();
      throw new Error(`Gemini embeddings failed: ${res.status} ${payload}`);
    }

    const payload = (await res.json()) as { embedding?: { values?: number[] } };
    return payload.embedding?.values ?? [];
  };

  const embedBatch = async (texts: string[]): Promise<number[][]> => {
    if (texts.length === 0) return [];

    const url = `${baseUrl}/${modelPath}:batchEmbedContents`;
    const requests = texts.map((text) => ({
      model: modelPath,
      content: { parts: [{ text }] },
      taskType: "RETRIEVAL_DOCUMENT",
    }));

    const res = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify({ requests }),
    });

    if (!res.ok) {
      const payload = await res.text();
      throw new Error(`Gemini batch embeddings failed: ${res.status} ${payload}`);
    }

    const payload = (await res.json()) as {
      embeddings?: Array<{ values?: number[] }>;
    };

    const embeddings = Array.isArray(payload.embeddings) ? payload.embeddings : [];
    return texts.map((_, i) => embeddings[i]?.values ?? []);
  };

  return {
    id: "gemini",
    model,
    embedQuery,
    embedBatch,
  };
}

// =============================================================================
// Local Embeddings (node-llama-cpp)
// =============================================================================

const DEFAULT_LOCAL_MODEL = "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf";

async function importNodeLlamaCpp(): Promise<{
  getLlama: (opts: { logLevel: number }) => Promise<any>;
  resolveModelFile: (path: string, cacheDir?: string) => Promise<string>;
  LlamaLogLevel: { error: number };
}> {
  // Dynamic import for optional dependency
  return await (Function('return import("node-llama-cpp")')() as Promise<any>);
}

function sanitizeAndNormalizeEmbedding(vec: number[]): number[] {
  const sanitized = vec.map((value) => (Number.isFinite(value) ? value : 0));
  const magnitude = Math.sqrt(sanitized.reduce((sum, value) => sum + value * value, 0));
  if (magnitude < 1e-10) return sanitized;
  return sanitized.map((value) => value / magnitude);
}

export async function createLocalEmbeddingProvider(
  config: SemanticMemoryConfig
): Promise<EmbeddingProvider> {
  const modelPath = config.local?.modelPath?.trim() || DEFAULT_LOCAL_MODEL;
  const modelCacheDir = config.local?.modelCacheDir?.trim();

  const { getLlama, resolveModelFile, LlamaLogLevel } = await importNodeLlamaCpp();

  let llama: any = null;
  let embeddingModel: any = null;
  let embeddingContext: any = null;

  const ensureContext = async () => {
    if (!llama) {
      llama = await getLlama({ logLevel: LlamaLogLevel.error });
    }
    if (!embeddingModel) {
      const resolved = await resolveModelFile(modelPath, modelCacheDir || undefined);
      embeddingModel = await llama.loadModel({ modelPath: resolved });
    }
    if (!embeddingContext) {
      embeddingContext = await embeddingModel.createEmbeddingContext();
    }
    return embeddingContext;
  };

  return {
    id: "local",
    model: modelPath,
    embedQuery: async (text) => {
      const ctx = await ensureContext();
      const embedding = await ctx.getEmbeddingFor(text);
      return sanitizeAndNormalizeEmbedding(Array.from(embedding.vector));
    },
    embedBatch: async (texts) => {
      const ctx = await ensureContext();
      return Promise.all(
        texts.map(async (text) => {
          const embedding = await ctx.getEmbeddingFor(text);
          return sanitizeAndNormalizeEmbedding(Array.from(embedding.vector));
        })
      );
    },
  };
}

// =============================================================================
// Ollama Embeddings
// =============================================================================

export async function createOllamaEmbeddingProvider(
  config: SemanticMemoryConfig
): Promise<EmbeddingProvider> {
  const baseUrl = (
    config.ollama?.baseUrl ||
    config.baseUrl ||
    process.env.OLLAMA_HOST ||
    "http://ollama:11434"
  ).replace(/\/$/, "");
  const model = config.ollama?.model || config.model || "qwen3-embedding:0.6b";

  // Verify Ollama is reachable
  const healthRes = await fetch(`${baseUrl}/api/tags`).catch(() => null);
  if (!healthRes?.ok) {
    throw new Error(`Ollama not reachable at ${baseUrl}`);
  }

  return {
    id: "ollama",
    model,
    embedQuery: async (text) => {
      const res = await fetch(`${baseUrl}/api/embed`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, input: text }),
      });
      if (!res.ok) {
        throw new Error(`Ollama embed failed: ${res.status} ${await res.text()}`);
      }
      const data = (await res.json()) as { embeddings: number[][] };
      return data.embeddings[0];
    },
    embedBatch: async (texts) => {
      // Ollama /api/embed accepts input as string[] natively
      const res = await fetch(`${baseUrl}/api/embed`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, input: texts }),
      });
      if (!res.ok) {
        throw new Error(`Ollama batch embed failed: ${res.status} ${await res.text()}`);
      }
      const data = (await res.json()) as { embeddings: number[][] };
      return data.embeddings;
    },
  };
}

// =============================================================================
// Provider Factory
// =============================================================================

export async function createEmbeddingProvider(
  config: SemanticMemoryConfig
): Promise<EmbeddingProvider> {
  const provider = config.provider;

  if (provider === "openai") {
    return createOpenAiEmbeddingProvider(config);
  }

  if (provider === "gemini") {
    return createGeminiEmbeddingProvider(config);
  }

  if (provider === "ollama") {
    return createOllamaEmbeddingProvider(config);
  }

  if (provider === "local") {
    return createLocalEmbeddingProvider(config);
  }

  // Auto: try OpenAI → Gemini → Ollama → local
  const errors: string[] = [];

  try {
    return await createOpenAiEmbeddingProvider(config);
  } catch (err) {
    errors.push(`OpenAI: ${err instanceof Error ? err.message : String(err)}`);
  }

  try {
    return await createGeminiEmbeddingProvider(config);
  } catch (err) {
    errors.push(`Gemini: ${err instanceof Error ? err.message : String(err)}`);
  }

  try {
    return await createOllamaEmbeddingProvider(config);
  } catch (err) {
    errors.push(`Ollama: ${err instanceof Error ? err.message : String(err)}`);
  }

  try {
    return await createLocalEmbeddingProvider(config);
  } catch (err) {
    errors.push(`Local: ${err instanceof Error ? err.message : String(err)}`);
  }

  throw new Error(`No embedding provider available:\n${errors.join("\n")}`);
}
