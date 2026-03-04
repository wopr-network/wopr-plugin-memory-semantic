import { describe, it, expect, vi, beforeEach } from "vitest";
import { mockLogger } from "./helpers.js";
import { createSessionDestroyHandler } from "../../../src/core-memory/session-hook.js";

function mockSessionApi() {
  return {
    getContext: vi.fn().mockResolvedValue(null),
    setContext: vi.fn().mockResolvedValue(undefined),
    readConversationLog: vi.fn().mockResolvedValue([]),
  };
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("createSessionDestroyHandler", () => {
  it("returns a function", async () => {
    const log = mockLogger();
    const handler = await createSessionDestroyHandler({
      sessionsDir: "/sessions",
      log,
    });
    expect(typeof handler).toBe("function");
  });

  it("warns and returns when sessionApi is not provided", async () => {
    const log = mockLogger();
    const handler = await createSessionDestroyHandler({
      sessionsDir: "/sessions",
      log,
    });

    await handler("test-session", "timeout");
    expect(vi.mocked(log.warn)).toHaveBeenCalledWith(
      expect.stringContaining("ctx.session not available"),
    );
  });

  it("returns early for empty conversation", async () => {
    const log = mockLogger();
    const sessionApi = mockSessionApi();
    sessionApi.readConversationLog.mockResolvedValue([]);

    const handler = await createSessionDestroyHandler({
      sessionsDir: "/sessions",
      log,
      sessionApi: sessionApi as any,
    });

    await handler("test-session", "timeout");
    expect(vi.mocked(sessionApi.setContext)).not.toHaveBeenCalled();
  });

  it("saves conversation to date-based file", async () => {
    const log = mockLogger();
    const sessionApi = mockSessionApi();
    sessionApi.readConversationLog.mockResolvedValue([
      { from: "Alice", content: "Hello WOPR", type: "message" },
      { from: "WOPR", content: "Hello Alice", type: "response" },
    ]);

    const handler = await createSessionDestroyHandler({
      sessionsDir: "/sessions",
      log,
      sessionApi: sessionApi as any,
    });

    await handler("test-session", "timeout");

    expect(vi.mocked(sessionApi.setContext)).toHaveBeenCalledWith(
      "test-session",
      expect.stringMatching(/memory\/\d{4}-\d{2}-\d{2}\.md/),
      expect.stringContaining("Alice"),
      "session",
    );
    expect(vi.mocked(log.info)).toHaveBeenCalledWith(
      expect.stringContaining("[session-hook]"),
    );
  });

  it("appends to existing content", async () => {
    const log = mockLogger();
    const sessionApi = mockSessionApi();
    sessionApi.getContext.mockResolvedValue("# Existing content\n\n");
    sessionApi.readConversationLog.mockResolvedValue([
      { from: "Alice", content: "New message", type: "message" },
    ]);

    const handler = await createSessionDestroyHandler({
      sessionsDir: "/sessions",
      log,
      sessionApi: sessionApi as any,
    });

    await handler("test-session", "timeout");

    const setCall = vi.mocked(sessionApi.setContext).mock.calls[0];
    expect(setCall).toBeDefined();
    const content = setCall![2] as string;
    expect(content).toContain("# Existing content");
    expect(content).toContain("New message");
  });

  it("skips context and middleware entries", async () => {
    const log = mockLogger();
    const sessionApi = mockSessionApi();
    sessionApi.readConversationLog.mockResolvedValue([
      { from: "system", content: "system context", type: "context" },
      { from: "system", content: "middleware info", type: "middleware" },
      { from: "Alice", content: "Real message", type: "message" },
    ]);

    const handler = await createSessionDestroyHandler({
      sessionsDir: "/sessions",
      log,
      sessionApi: sessionApi as any,
    });

    await handler("test-session", "timeout");

    const setCall = vi.mocked(sessionApi.setContext).mock.calls[0];
    expect(setCall![2]).not.toContain("system context");
    expect(setCall![2]).not.toContain("middleware info");
    expect(setCall![2]).toContain("Real message");
  });

  it("skips system sender messages", async () => {
    const log = mockLogger();
    const sessionApi = mockSessionApi();
    sessionApi.readConversationLog.mockResolvedValue([
      { from: "system", content: "internal", type: "message" },
    ]);

    const handler = await createSessionDestroyHandler({
      sessionsDir: "/sessions",
      log,
      sessionApi: sessionApi as any,
    });

    await handler("test-session", "timeout");
    expect(vi.mocked(sessionApi.setContext)).not.toHaveBeenCalled();
  });

  it("logs warning on setContext failure", async () => {
    const log = mockLogger();
    const sessionApi = mockSessionApi();
    sessionApi.readConversationLog.mockResolvedValue([
      { from: "Alice", content: "Hello", type: "message" },
    ]);
    sessionApi.setContext.mockRejectedValue(new Error("write failed"));

    const handler = await createSessionDestroyHandler({
      sessionsDir: "/sessions",
      log,
      sessionApi: sessionApi as any,
    });

    await handler("test-session", "timeout");
    expect(vi.mocked(log.warn)).toHaveBeenCalledWith(
      expect.stringContaining("[session-hook]"),
    );
  });
});
