import { describe, it, expect, vi } from "vitest";
import { listSessionNames, buildSessionEntryFromSql, getRecentSessionContentFromSql } from "../../src/core-memory/session-files.js";

describe("session-files SQL", () => {
  const mockSessionApi = {
    getContext: vi.fn(),
    setContext: vi.fn(),
    readConversationLog: vi.fn(),
  };

  it("listSessionNames queries SQL for active sessions", async () => {
    const mockStorage = {
      raw: vi.fn().mockResolvedValue([
        { name: "session-1" },
        { name: "session-2" },
      ]),
    };
    const names = await listSessionNames(mockStorage as any);
    expect(names).toEqual(["session-1", "session-2"]);
    expect(mockStorage.raw).toHaveBeenCalledWith(
      expect.stringContaining("SELECT"),
      expect.anything(),
    );
  });

  it("buildSessionEntryFromSql formats ConversationEntry[] into SessionFileEntry", async () => {
    mockSessionApi.readConversationLog.mockResolvedValue([
      { ts: 1000, from: "Alice", content: "Hello", type: "message" },
      { ts: 2000, from: "WOPR", content: "Hi", type: "response" },
    ]);

    const entry = await buildSessionEntryFromSql("test-session", mockSessionApi);
    expect(entry).not.toBeNull();
    expect(entry!.content).toContain("User: Hello");
    expect(entry!.content).toContain("Assistant: Hi");
    expect(entry!.path).toBe("sessions/test-session");
  });

  it("buildSessionEntryFromSql returns null for empty conversation", async () => {
    mockSessionApi.readConversationLog.mockResolvedValue([]);
    const entry = await buildSessionEntryFromSql("empty", mockSessionApi);
    expect(entry).toBeNull();
  });

  it("getRecentSessionContentFromSql returns last N messages", async () => {
    mockSessionApi.readConversationLog.mockResolvedValue([
      { ts: 1000, from: "Alice", content: "msg1", type: "message" },
      { ts: 2000, from: "WOPR", content: "msg2", type: "response" },
      { ts: 3000, from: "Alice", content: "msg3", type: "message" },
    ]);

    const content = await getRecentSessionContentFromSql("test", mockSessionApi, 2);
    expect(content).toContain("msg2");
    expect(content).toContain("msg3");
    expect(content).not.toContain("msg1");
  });
});
