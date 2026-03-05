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
      name: "model",
      type: "text",
      label: "Embedding Model",
      description: "Model name for embeddings",
      default: "text-embedding-3-small",
    },
    {
      name: "maxWriteBytes",
      type: "number",
      label: "Max Write Size (bytes)",
      description: "Maximum byte size for memory_write content (default: 1048576 = 1 MB)",
      default: 1048576,
    },
    {
      name: "autoRecallEnabled",
      type: "boolean",
      label: "Auto-Recall Enabled",
      description:
        "Automatically inject relevant memories into conversation context (maps to autoRecall.enabled at runtime)",
      default: true,
    },
    {
      name: "autoCaptureEnabled",
      type: "boolean",
      label: "Auto-Capture Enabled",
      description:
        "Automatically extract and store memories from conversations (maps to autoCapture.enabled at runtime)",
      default: true,
    },
    {
      name: "instanceId",
      type: "text",
      label: "Instance ID",
      description: "Unique identifier for multi-tenant memory isolation",
      default: "default",
    },
    {
      name: "searchMaxResults",
      type: "number",
      label: "Search Max Results",
      description:
        "Maximum number of results returned by semantic search (integer, 1–100, default: 10; non-integer values are rounded; maps to search.maxResults at runtime)",
      default: 10,
    },
    {
      name: "searchHybridWeight",
      type: "number",
      label: "Hybrid Search Vector Weight",
      description:
        "Weight of vector similarity vs text matching in hybrid search (0.0–1.0, default: 0.7; maps to search.hybridWeight at runtime)",
      default: 0.7,
      pattern: "^(0(\\.\\d+)?|1(\\.0*)?)$",
      patternError: "Must be a number between 0.0 and 1.0",
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
      {
        type: "semantic-search",
        id: "semantic-search",
        displayName: "Semantic Search",
      },
      {
        type: "auto-recall",
        id: "auto-recall",
        displayName: "Auto-Recall",
      },
      {
        type: "auto-capture",
        id: "auto-capture",
        displayName: "Auto-Capture",
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
