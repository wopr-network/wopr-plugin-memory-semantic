import { describe, it, expect, vi } from "vitest";
import { listSessionNames, buildSessionEntryFromSql, getRecentSessionContentFromSql } from "../../src/core-memory/session-files.js";

describe("session-files SQL", () => {
  const mockLog = {
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
    debug: vi.fn(),
  };

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
    const names = await listSessionNames(mockStorage as any, mockLog as any);
    expect(names).toEqual(["session-1", "session-2"]);
    expect(mockStorage.raw).toHaveBeenCalledWith(
      expect.stringContaining("SELECT"),
      expect.anything(),
    );
  });

  it("listSessionNames logs warning on SQL failure", async () => {
    mockLog.warn.mockClear();
    const mockStorage = {
      raw: vi.fn().mockRejectedValue(new Error("db error")),
    };
    const names = await listSessionNames(mockStorage as any, mockLog as any);
    expect(names).toEqual([]);
    expect(mockLog.warn).toHaveBeenCalledWith(
      expect.stringContaining("[session-files]"),
      expect.any(Error),
    );
  });

  it("buildSessionEntryFromSql formats ConversationEntry[] into SessionFileEntry", async () => {
    mockSessionApi.readConversationLog.mockResolvedValue([
      { ts: 1000, from: "Alice", content: "Hello", type: "message" },
      { ts: 2000, from: "WOPR", content: "Hi", type: "response" },
    ]);

    const entry = await buildSessionEntryFromSql("test-session", mockSessionApi, mockLog as any);
    expect(entry).not.toBeNull();
    expect(entry!.content).toContain("User: Hello");
    expect(entry!.content).toContain("Assistant: Hi");
    expect(entry!.path).toBe("sessions/test-session");
  });

  it("buildSessionEntryFromSql returns null for empty conversation", async () => {
    mockSessionApi.readConversationLog.mockResolvedValue([]);
    const entry = await buildSessionEntryFromSql("empty", mockSessionApi, mockLog as any);
    expect(entry).toBeNull();
  });

  it("buildSessionEntryFromSql logs warning on failure", async () => {
    mockLog.warn.mockClear();
    mockSessionApi.readConversationLog.mockRejectedValue(new Error("db error"));
    const entry = await buildSessionEntryFromSql("broken", mockSessionApi, mockLog as any);
    expect(entry).toBeNull();
    expect(mockLog.warn).toHaveBeenCalledWith(
      expect.stringContaining("[session-files]"),
      expect.any(Error),
    );
  });

  it("getRecentSessionContentFromSql returns last N messages", async () => {
    mockSessionApi.readConversationLog.mockResolvedValue([
      { ts: 1000, from: "Alice", content: "msg1", type: "message" },
      { ts: 2000, from: "WOPR", content: "msg2", type: "response" },
      { ts: 3000, from: "Alice", content: "msg3", type: "message" },
    ]);

    const content = await getRecentSessionContentFromSql("test", mockSessionApi, mockLog as any, 2);
    expect(content).toContain("msg2");
    expect(content).toContain("msg3");
    expect(content).not.toContain("msg1");
  });

  it("getRecentSessionContentFromSql logs warning on failure", async () => {
    mockLog.warn.mockClear();
    mockSessionApi.readConversationLog.mockRejectedValue(new Error("db error"));
    const content = await getRecentSessionContentFromSql("broken", mockSessionApi, mockLog as any);
    expect(content).toBeNull();
    expect(mockLog.warn).toHaveBeenCalledWith(
      expect.stringContaining("[session-files]"),
      expect.any(Error),
    );
  });
});
