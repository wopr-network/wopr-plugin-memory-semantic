import { createHash } from "node:crypto";
import type { ConfigSchema, ContextProvider, PluginManifest } from "@wopr-network/plugin-types";

export const pluginConfigSchema: ConfigSchema = {
  title: "Semantic Memory",
  description: "Configure semantic memory with embeddings for auto-recall and auto-capture",
  fields: [
    {
      name: "provider",
      type: "text",
      label: "Embedding Provider",
      description: "Which embedding provider to use (auto tries OpenAI → Gemini → Ollama → local)",
      default: "auto",
    },
    {
      name: "apiKey",
      type: "password",
      label: "API Key",
      description: "API key for the embedding provider (OpenAI or Gemini)",
      secret: true,
    },
    {
      name: "model",
      type: "text",
      label: "Embedding Model",
      description: "Model name for embeddings",
      default: "text-embedding-3-small",
    },
  ],
};

export const pluginManifest: PluginManifest = {
  name: "@wopr-network/wopr-plugin-memory-semantic",
  version: "1.0.0",
  description: "Semantic memory search with embeddings, auto-recall, and auto-capture",
  capabilities: ["memory", "semantic-search", "auto-recall", "auto-capture"],
  category: "memory",
  tags: ["memory", "semantic", "embeddings", "vector-search", "auto-recall"],
  icon: "🧠",
  provides: {
    capabilities: [
      {
        type: "memory",
        id: "semantic-memory",
        displayName: "Semantic Memory (Embeddings)",
      },
    ],
  },
  requires: {},
  lifecycle: {
    shutdownBehavior: "graceful",
    shutdownTimeoutMs: 15000,
  },
  configSchema: pluginConfigSchema,
};

export const memoryContextProvider: ContextProvider = {
  name: "memory-semantic",
  priority: 10,
  async getContext(_session: string): Promise<null> {
    // Actual injection happens via session:beforeInject event handler.
    // This registration makes the provider visible to the platform.
    return null;
  },
};

/** Generate deterministic ID from content to avoid duplicates */
export function contentHash(text: string): string {
  return createHash("sha256").update(text).digest("hex");
}
