/**
 * WebMCP Memory Tools
 *
 * Browser-side WebMCP tool handlers for semantic memory search.
 * These tools are registered on the WebMCPRegistry when the
 * memory-semantic plugin is loaded, and unregistered when disabled.
 *
 * Pattern: browser WebMCP tool -> fetch(daemon REST API) -> response
 */

// -- Types (mirrors webmcp.ts from wopr-plugin-webui) --

export interface ParameterSchema {
  type: string;
  description?: string;
  required?: boolean;
}

export interface AuthContext {
  userId?: string;
  sessionId?: string;
  token?: string;
  [key: string]: unknown;
}

export type WebMCPHandler = (params: Record<string, unknown>, auth: AuthContext) => unknown | Promise<unknown>;

export interface WebMCPToolDeclaration {
  name: string;
  description: string;
  parameters?: Record<string, ParameterSchema>;
}

export interface WebMCPTool extends WebMCPToolDeclaration {
  handler: WebMCPHandler;
}

/**
 * Minimal registry interface — matches WebMCPRegistry from wopr-plugin-webui.
 * Only the methods needed for registration/unregistration.
 */
export interface WebMCPRegistryLike {
  register(tool: WebMCPTool): void;
  unregister(name: string): void;
}

// -- Internal helpers --

interface RequestOptions {
  method?: string;
  body?: string;
  headers?: Record<string, string>;
}

async function daemonRequest<T>(
  apiBase: string,
  path: string,
  auth: AuthContext,
  options?: RequestOptions,
): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...options?.headers,
  };
  if (auth.token) {
    headers.Authorization = `Bearer ${auth.token}`;
  }
  const res = await fetch(`${apiBase}${path}`, {
    ...options,
    headers,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: "Request failed" }));
    throw new Error((err as { error?: string }).error || `Request failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

// -- Manifest --

/** WebMCP tool declarations for the plugin manifest. */
export const WEBMCP_MANIFEST: WebMCPToolDeclaration[] = [
  {
    name: "searchMemory",
    description: "Search the bot's semantic memory for relevant information",
    parameters: {
      query: {
        type: "string",
        description: "The search query",
        required: true,
      },
      limit: {
        type: "number",
        description: "Maximum number of results to return (default: 10)",
        required: false,
      },
    },
  },
  {
    name: "listMemoryCollections",
    description: "List available memory collections",
  },
  {
    name: "getMemoryStats",
    description: "Get memory index statistics",
  },
];

// -- Tool registration --

/**
 * Register all 3 memory WebMCP tools on the given registry.
 *
 * @param registry - A WebMCPRegistry (or compatible) instance
 * @param apiBase  - Base URL of the WOPR daemon API (e.g. "/api" or "http://localhost:7437/api")
 */
export function registerMemoryTools(registry: WebMCPRegistryLike, apiBase = "/api"): void {
  // 1. searchMemory
  registry.register({
    name: "searchMemory",
    description: "Search the bot's semantic memory for relevant information",
    parameters: {
      query: {
        type: "string",
        description: "The search query",
        required: true,
      },
      limit: {
        type: "number",
        description: "Maximum number of results to return (default: 10)",
        required: false,
      },
    },
    handler: async (params: Record<string, unknown>, auth: AuthContext) => {
      const query = params.query as string;
      if (!query) {
        throw new Error("Parameter 'query' is required");
      }
      const limit = typeof params.limit === "number" && params.limit > 0 ? params.limit : 10;

      // Call the daemon's session inject endpoint with a structured search request.
      // The bot invokes the memory_search tool and returns results.
      const result = await daemonRequest<{
        session: string;
        response: string;
      }>(apiBase, "/sessions/default/inject", auth, {
        method: "POST",
        body: JSON.stringify({
          message: `Use the memory_search tool with query "${query}" and maxResults ${limit}. Return only the raw search results as JSON, no commentary.`,
          from: "webmcp",
        }),
      });

      return {
        query,
        response: result.response,
        session: result.session,
      };
    },
  });

  // 2. listMemoryCollections
  registry.register({
    name: "listMemoryCollections",
    description: "List available memory collections",
    handler: async (_params: Record<string, unknown>, auth: AuthContext) => {
      // List plugins and filter for memory-related ones to identify collections
      const data = await daemonRequest<{
        plugins: Array<{
          name: string;
          description: string | null;
          enabled: boolean;
          loaded: boolean;
        }>;
      }>(apiBase, "/plugins", auth);

      const memoryPlugins = data.plugins.filter(
        (p) => p.loaded && (p.name.includes("memory") || p.name.includes("semantic")),
      );

      return {
        collections: memoryPlugins.map((p) => ({
          name: p.name,
          description: p.description,
          enabled: p.enabled,
          loaded: p.loaded,
        })),
      };
    },
  });

  // 3. getMemoryStats
  registry.register({
    name: "getMemoryStats",
    description: "Get memory index statistics",
    handler: async (_params: Record<string, unknown>, auth: AuthContext) => {
      const pluginName = encodeURIComponent("memory-semantic");
      const data = await daemonRequest<{
        name: string;
        installed: boolean;
        enabled: boolean;
        loaded: boolean;
        version: string;
        source: string;
        manifest: {
          capabilities?: string[];
        } | null;
      }>(apiBase, `/plugins/${pluginName}/health`, auth);

      return {
        name: data.name,
        installed: data.installed,
        enabled: data.enabled,
        loaded: data.loaded,
        version: data.version,
        source: data.source,
        capabilities: data.manifest?.capabilities ?? [],
      };
    },
  });
}

/**
 * Unregister all 3 memory WebMCP tools from the given registry.
 */
export function unregisterMemoryTools(registry: WebMCPRegistryLike): void {
  for (const decl of WEBMCP_MANIFEST) {
    registry.unregister(decl.name);
  }
}
