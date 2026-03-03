import { describe, it, expect, vi } from "vitest";
import { createSessionDestroyHandler } from "../../src/core-memory/session-hook.js";

describe("createSessionDestroyHandler (SQL)", () => {
  const mockLog = {
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
    debug: vi.fn(),
  };

  it("reads conversation from SQL and writes summary via setContext", async () => {
    const mockSessionApi = {
      getContext: vi.fn().mockResolvedValue(null),
      setContext: vi.fn(),
      readConversationLog: vi.fn().mockResolvedValue([
        { ts: 1000, from: "Alice", content: "Hello", type: "message" as const },
        { ts: 2000, from: "WOPR", content: "Hi there", type: "response" as const },
      ]),
    };

    const handler = await createSessionDestroyHandler({
      sessionsDir: "/tmp/sessions",
      log: mockLog,
      sessionApi: mockSessionApi,
    });

    await handler("test-session", "user-requested");

    // Should have called readConversationLog
    expect(mockSessionApi.readConversationLog).toHaveBeenCalledWith("test-session");

    // Should have called setContext to write the memory summary
    expect(mockSessionApi.setContext).toHaveBeenCalledWith(
      "test-session",
      expect.stringMatching(/^memory\/\d{4}-\d{2}-\d{2}\.md$/),
      expect.stringContaining("**Alice**: Hello"),
      "session",
    );
  });

  it("does nothing when no messages exist", async () => {
    const mockSessionApi = {
      getContext: vi.fn(),
      setContext: vi.fn(),
      readConversationLog: vi.fn().mockResolvedValue([]),
    };

    const handler = await createSessionDestroyHandler({
      sessionsDir: "/tmp/sessions",
      log: mockLog,
      sessionApi: mockSessionApi,
    });

    await handler("empty-session", "timeout");
    expect(mockSessionApi.setContext).not.toHaveBeenCalled();
  });

  it("falls back gracefully when sessionApi is undefined", async () => {
    const handler = await createSessionDestroyHandler({
      sessionsDir: "/tmp/sessions",
      log: mockLog,
      sessionApi: undefined,
    });

    // Should not throw
    await handler("no-api-session", "timeout");
    expect(mockLog.warn).toHaveBeenCalledWith(
      expect.stringContaining("ctx.session not available"),
    );
  });
});
