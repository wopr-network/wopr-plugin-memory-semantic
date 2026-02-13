import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  type AuthContext,
  type WebMCPRegistryLike,
  type WebMCPTool,
  WEBMCP_MANIFEST,
  registerMemoryTools,
  unregisterMemoryTools,
} from "../../src/webmcp.js";

// Mock global fetch
const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

function mockJsonResponse(data: unknown, ok = true, status = 200) {
  return {
    ok,
    status,
    json: vi.fn().mockResolvedValue(data),
  };
}

/** Simple in-memory registry for testing. */
function createTestRegistry(): WebMCPRegistryLike & {
  tools: Map<string, WebMCPTool>;
  get(name: string): WebMCPTool | undefined;
  list(): string[];
} {
  const tools = new Map<string, WebMCPTool>();
  return {
    tools,
    register(tool: WebMCPTool) {
      tools.set(tool.name, tool);
    },
    unregister(name: string) {
      tools.delete(name);
    },
    get(name: string) {
      return tools.get(name);
    },
    list() {
      return Array.from(tools.keys());
    },
  };
}

/** Retrieve a tool from the registry, throwing if it is missing. */
function getTool(registry: ReturnType<typeof createTestRegistry>, name: string) {
  const tool = registry.get(name);
  if (!tool) throw new Error(`Tool "${name}" not registered`);
  return tool;
}

describe("WEBMCP_MANIFEST", () => {
  it("should declare 3 tools", () => {
    expect(WEBMCP_MANIFEST).toHaveLength(3);
  });

  it("should include searchMemory, listMemoryCollections, getMemoryStats", () => {
    const names = WEBMCP_MANIFEST.map((t) => t.name);
    expect(names).toContain("searchMemory");
    expect(names).toContain("listMemoryCollections");
    expect(names).toContain("getMemoryStats");
  });

  it("searchMemory should have query and limit parameters", () => {
    const tool = WEBMCP_MANIFEST.find((t) => t.name === "searchMemory");
    expect(tool?.parameters?.query?.type).toBe("string");
    expect(tool?.parameters?.query?.required).toBe(true);
    expect(tool?.parameters?.limit?.type).toBe("number");
    expect(tool?.parameters?.limit?.required).toBe(false);
  });
});

describe("registerMemoryTools", () => {
  let registry: ReturnType<typeof createTestRegistry>;
  const API_BASE = "/api";

  beforeEach(() => {
    registry = createTestRegistry();
    mockFetch.mockReset();
  });

  it("should register all 3 tools", () => {
    registerMemoryTools(registry, API_BASE);
    const names = registry.list();
    expect(names).toHaveLength(3);
    expect(names).toContain("searchMemory");
    expect(names).toContain("listMemoryCollections");
    expect(names).toContain("getMemoryStats");
  });

  it("should use default apiBase when not provided", () => {
    registerMemoryTools(registry);
    expect(registry.list()).toHaveLength(3);
  });

  describe("searchMemory", () => {
    it("should POST to /sessions/default/inject with search request", async () => {
      const response = { session: "default", response: "Found 2 results..." };
      mockFetch.mockResolvedValue(mockJsonResponse(response));
      registerMemoryTools(registry, API_BASE);

      const tool = getTool(registry, "searchMemory");
      const result = await tool.handler({ query: "authentication patterns" }, {});

      expect(mockFetch).toHaveBeenCalledWith(
        "/api/sessions/default/inject",
        expect.objectContaining({ method: "POST" }),
      );
      const body = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(body.message).toContain("authentication patterns");
      expect(body.message).toContain("memory_search");
      expect(body.from).toBe("webmcp");
      expect(result).toEqual({
        query: "authentication patterns",
        response: response.response,
        session: response.session,
      });
    });

    it("should include limit in the search request", async () => {
      mockFetch.mockResolvedValue(
        mockJsonResponse({ session: "default", response: "ok" }),
      );
      registerMemoryTools(registry, API_BASE);

      const tool = getTool(registry, "searchMemory");
      await tool.handler({ query: "test", limit: 5 }, {});

      const body = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(body.message).toContain("maxResults 5");
    });

    it("should default limit to 10 when not provided", async () => {
      mockFetch.mockResolvedValue(
        mockJsonResponse({ session: "default", response: "ok" }),
      );
      registerMemoryTools(registry, API_BASE);

      const tool = getTool(registry, "searchMemory");
      await tool.handler({ query: "test" }, {});

      const body = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(body.message).toContain("maxResults 10");
    });

    it("should throw when query parameter is missing", async () => {
      registerMemoryTools(registry, API_BASE);
      const tool = getTool(registry, "searchMemory");
      await expect(tool.handler({}, {})).rejects.toThrow(
        "Parameter 'query' is required",
      );
    });

    it("should include bearer token when auth.token is present", async () => {
      mockFetch.mockResolvedValue(
        mockJsonResponse({ session: "default", response: "ok" }),
      );
      registerMemoryTools(registry, API_BASE);

      const tool = getTool(registry, "searchMemory");
      const auth: AuthContext = { token: "my-secret-token" };
      await tool.handler({ query: "test" }, auth);

      const headers = mockFetch.mock.calls[0][1].headers;
      expect(headers.Authorization).toBe("Bearer my-secret-token");
    });

    it("should not include Authorization header when no token", async () => {
      mockFetch.mockResolvedValue(
        mockJsonResponse({ session: "default", response: "ok" }),
      );
      registerMemoryTools(registry, API_BASE);

      const tool = getTool(registry, "searchMemory");
      await tool.handler({ query: "test" }, {});

      const headers = mockFetch.mock.calls[0][1].headers;
      expect(headers.Authorization).toBeUndefined();
    });
  });

  describe("listMemoryCollections", () => {
    it("should GET /plugins and filter loaded memory-related plugins", async () => {
      const plugins = {
        plugins: [
          { name: "memory-semantic", description: "Semantic memory", enabled: true, loaded: true },
          { name: "discord", description: "Discord bot", enabled: true, loaded: true },
          { name: "memory-keyword", description: "Keyword memory", enabled: true, loaded: true },
        ],
      };
      mockFetch.mockResolvedValue(mockJsonResponse(plugins));
      registerMemoryTools(registry, API_BASE);

      const tool = getTool(registry, "listMemoryCollections");
      const result = (await tool.handler({}, {})) as { collections: Array<{ name: string; loaded: boolean }> };

      expect(mockFetch).toHaveBeenCalledWith("/api/plugins", expect.any(Object));
      expect(result.collections).toHaveLength(2);
      expect(result.collections[0].name).toBe("memory-semantic");
      expect(result.collections[1].name).toBe("memory-keyword");
    });

    it("should exclude unloaded memory plugins", async () => {
      const plugins = {
        plugins: [
          { name: "memory-semantic", description: "Semantic memory", enabled: true, loaded: true },
          { name: "memory-keyword", description: "Keyword memory", enabled: true, loaded: false },
        ],
      };
      mockFetch.mockResolvedValue(mockJsonResponse(plugins));
      registerMemoryTools(registry, API_BASE);

      const tool = getTool(registry, "listMemoryCollections");
      const result = (await tool.handler({}, {})) as { collections: Array<{ name: string }> };

      expect(result.collections).toHaveLength(1);
      expect(result.collections[0].name).toBe("memory-semantic");
    });

    it("should return empty collections when no memory plugins loaded", async () => {
      mockFetch.mockResolvedValue(
        mockJsonResponse({
          plugins: [
            { name: "discord", description: "Discord bot", enabled: true, loaded: true },
          ],
        }),
      );
      registerMemoryTools(registry, API_BASE);

      const tool = getTool(registry, "listMemoryCollections");
      const result = (await tool.handler({}, {})) as { collections: unknown[] };

      expect(result.collections).toHaveLength(0);
    });

    it("should include bearer token in auth header", async () => {
      mockFetch.mockResolvedValue(
        mockJsonResponse({ plugins: [] }),
      );
      registerMemoryTools(registry, API_BASE);

      const tool = getTool(registry, "listMemoryCollections");
      await tool.handler({}, { token: "tok-abc" });

      const headers = mockFetch.mock.calls[0][1].headers;
      expect(headers.Authorization).toBe("Bearer tok-abc");
    });
  });

  describe("getMemoryStats", () => {
    it("should GET /plugins/memory-semantic/health", async () => {
      const health = {
        name: "memory-semantic",
        installed: true,
        enabled: true,
        loaded: true,
        version: "1.0.0",
        source: "npm",
        manifest: { capabilities: ["vector-search", "auto-recall"] },
      };
      mockFetch.mockResolvedValue(mockJsonResponse(health));
      registerMemoryTools(registry, API_BASE);

      const tool = getTool(registry, "getMemoryStats");
      const result = (await tool.handler({}, {})) as {
        name: string;
        loaded: boolean;
        capabilities: string[];
      };

      expect(mockFetch).toHaveBeenCalledWith(
        "/api/plugins/memory-semantic/health",
        expect.any(Object),
      );
      expect(result.name).toBe("memory-semantic");
      expect(result.loaded).toBe(true);
      expect(result.capabilities).toEqual(["vector-search", "auto-recall"]);
    });

    it("should handle missing manifest gracefully", async () => {
      mockFetch.mockResolvedValue(
        mockJsonResponse({
          name: "memory-semantic",
          installed: true,
          enabled: false,
          loaded: false,
          version: "1.0.0",
          source: "npm",
          manifest: null,
        }),
      );
      registerMemoryTools(registry, API_BASE);

      const tool = getTool(registry, "getMemoryStats");
      const result = (await tool.handler({}, {})) as { capabilities: string[] };

      expect(result.capabilities).toEqual([]);
    });

    it("should include bearer token in auth header", async () => {
      mockFetch.mockResolvedValue(
        mockJsonResponse({
          name: "memory-semantic",
          installed: true,
          enabled: true,
          loaded: true,
          version: "1.0.0",
          source: "npm",
          manifest: null,
        }),
      );
      registerMemoryTools(registry, API_BASE);

      const tool = getTool(registry, "getMemoryStats");
      await tool.handler({}, { token: "tok-stats" });

      const headers = mockFetch.mock.calls[0][1].headers;
      expect(headers.Authorization).toBe("Bearer tok-stats");
    });
  });

  describe("error handling", () => {
    it("should throw on non-ok response with error from body", async () => {
      mockFetch.mockResolvedValue(
        mockJsonResponse({ error: "Plugin not found" }, false, 404),
      );
      registerMemoryTools(registry, API_BASE);

      const tool = getTool(registry, "getMemoryStats");
      await expect(tool.handler({}, {})).rejects.toThrow("Plugin not found");
    });

    it("should throw generic error when body has no error field", async () => {
      mockFetch.mockResolvedValue(mockJsonResponse({}, false, 500));
      registerMemoryTools(registry, API_BASE);

      const tool = getTool(registry, "getMemoryStats");
      await expect(tool.handler({}, {})).rejects.toThrow("Request failed (500)");
    });

    it("should throw generic error when body is not JSON", async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 502,
        json: vi.fn().mockRejectedValue(new Error("not json")),
      });
      registerMemoryTools(registry, API_BASE);

      const tool = getTool(registry, "getMemoryStats");
      await expect(tool.handler({}, {})).rejects.toThrow("Request failed");
    });
  });

  describe("custom apiBase", () => {
    it("should use custom apiBase for all requests", async () => {
      mockFetch.mockResolvedValue(mockJsonResponse({ plugins: [] }));
      registerMemoryTools(registry, "http://localhost:7437/api");

      const tool = getTool(registry, "listMemoryCollections");
      await tool.handler({}, {});

      expect(mockFetch).toHaveBeenCalledWith(
        "http://localhost:7437/api/plugins",
        expect.any(Object),
      );
    });
  });
});

describe("unregisterMemoryTools", () => {
  it("should remove all 3 tools from the registry", () => {
    const registry = createTestRegistry();
    registerMemoryTools(registry);

    expect(registry.list()).toHaveLength(3);

    unregisterMemoryTools(registry);

    expect(registry.list()).toHaveLength(0);
  });

  it("should not throw when tools are not registered", () => {
    const registry = createTestRegistry();
    expect(() => unregisterMemoryTools(registry)).not.toThrow();
  });
});
