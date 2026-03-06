import { describe, expect, it, vi } from "vitest";
import { registerIdentityTools, unregisterIdentityTools } from "../../src/identity-tools.js";

function createMockContext() {
  const tools = new Map<string, any>();
  return {
    registerTool: vi.fn((tool: any) => tools.set(tool.name, tool)),
    unregisterTool: vi.fn((name: string) => tools.delete(name)),
    log: { info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn() },
    session: {
      getContext: vi.fn().mockResolvedValue(null),
      setContext: vi.fn().mockResolvedValue(undefined),
      readConversationLog: vi.fn().mockResolvedValue([]),
    },
    _tools: tools,
  } as any;
}

describe("identity tools", () => {
  it("should register identity_get and identity_update tools", () => {
    const ctx = createMockContext();
    registerIdentityTools(ctx);

    expect(ctx.registerTool).toHaveBeenCalledTimes(2);
    const names = ctx.registerTool.mock.calls.map((c: any) => c[0].name);
    expect(names).toContain("identity_get");
    expect(names).toContain("identity_update");
  });

  it("should unregister identity tools", () => {
    const ctx = createMockContext();
    registerIdentityTools(ctx);
    unregisterIdentityTools(ctx);

    expect(ctx.unregisterTool).toHaveBeenCalledWith("identity_get");
    expect(ctx.unregisterTool).toHaveBeenCalledWith("identity_update");
  });

  it("should skip registration if registerTool is not available", () => {
    const ctx = createMockContext();
    delete ctx.registerTool;
    registerIdentityTools(ctx);
    expect(ctx.log.warn).toHaveBeenCalled();
  });

  it("identity_get should return 'No IDENTITY.md found' when no context exists", async () => {
    const ctx = createMockContext();
    registerIdentityTools(ctx);

    const tool = ctx._tools.get("identity_get");
    const result = await tool.handler({}, { sessionName: "test-session" });
    expect(result.content[0].text).toContain("No IDENTITY.md found");
  });

  it("identity_get should parse identity fields from IDENTITY.md", async () => {
    const ctx = createMockContext();
    ctx.session.getContext.mockImplementation(async (session: string, file: string) => {
      if (session === "test-session" && file === "IDENTITY.md") {
        return "# Identity\n- Name: TestBot\n- Creature: Cat\n- Vibe: chill\n- Emoji: 🐱";
      }
      return null;
    });
    registerIdentityTools(ctx);

    const tool = ctx._tools.get("identity_get");
    const result = await tool.handler({}, { sessionName: "test-session" });
    const parsed = JSON.parse(result.content[0].text);
    expect(parsed.parsed.name).toBe("TestBot");
    expect(parsed.parsed.creature).toBe("Cat");
    expect(parsed.source).toBe("session");
  });

  it("identity_get should fall back to global context", async () => {
    const ctx = createMockContext();
    ctx.session.getContext.mockImplementation(async (session: string, file: string) => {
      if (session === "__global__" && file === "IDENTITY.md") {
        return "- Name: GlobalBot";
      }
      return null;
    });
    registerIdentityTools(ctx);

    const tool = ctx._tools.get("identity_get");
    const result = await tool.handler({}, { sessionName: "test-session" });
    const parsed = JSON.parse(result.content[0].text);
    expect(parsed.parsed.name).toBe("GlobalBot");
    expect(parsed.source).toBe("global");
  });

  it("identity_update should update IDENTITY.md via session context", async () => {
    const ctx = createMockContext();
    registerIdentityTools(ctx);

    const tool = ctx._tools.get("identity_update");
    const result = await tool.handler(
      { name: "NewBot", creature: "Dog" },
      { sessionName: "test-session" },
    );
    expect(result.content[0].text).toContain("Identity updated");
    expect(ctx.session.setContext).toHaveBeenCalledWith(
      "test-session",
      "IDENTITY.md",
      expect.stringContaining("Name: NewBot"),
      "session",
    );
  });
});
