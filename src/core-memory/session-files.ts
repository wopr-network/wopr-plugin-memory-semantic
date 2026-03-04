// Session file indexing - adapted from OpenClaw for WOPR
import fs from "node:fs/promises";
import path from "node:path";
import type { PluginLogger, StorageApi } from "@wopr-network/plugin-types";
import type { SessionApi } from "../types.js";
import { hashText } from "./internal.js";

// SESSIONS_DIR removed - passed as parameter instead

export type SessionFileEntry = {
  path: string;
  absPath: string;
  mtimeMs: number;
  size: number;
  hash: string;
  content: string;
};

export async function listSessionFiles(sessionsDir: string): Promise<string[]> {
  try {
    const entries = await fs.readdir(sessionsDir, { withFileTypes: true });
    return entries
      .filter((entry) => entry.isFile())
      .map((entry) => entry.name)
      .filter((name) => name.endsWith(".conversation.jsonl"))
      .map((name) => path.join(sessionsDir, name));
  } catch {
    return [];
  }
}

export function sessionPathForFile(absPath: string): string {
  return path.join("sessions", path.basename(absPath)).replace(/\\/g, "/");
}

function normalizeSessionText(value: string): string {
  return value
    .replace(/\s*\n+\s*/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export function extractSessionText(content: unknown): string | null {
  if (typeof content === "string") {
    const normalized = normalizeSessionText(content);
    return normalized ? normalized : null;
  }
  if (!Array.isArray(content)) {
    return null;
  }
  const parts: string[] = [];
  for (const block of content) {
    if (!block || typeof block !== "object") {
      continue;
    }
    const record = block as { type?: unknown; text?: unknown };
    if (record.type !== "text" || typeof record.text !== "string") {
      continue;
    }
    const normalized = normalizeSessionText(record.text);
    if (normalized) {
      parts.push(normalized);
    }
  }
  if (parts.length === 0) {
    return null;
  }
  return parts.join(" ");
}

export async function buildSessionEntry(absPath: string): Promise<SessionFileEntry | null> {
  try {
    const stat = await fs.stat(absPath);
    const raw = await fs.readFile(absPath, "utf-8");
    const lines = raw.split("\n");
    const collected: string[] = [];
    for (const line of lines) {
      if (!line.trim()) {
        continue;
      }
      let record: unknown;
      try {
        record = JSON.parse(line);
      } catch {
        continue;
      }
      if (!record || typeof record !== "object") {
        continue;
      }
      const entry = record as {
        type?: unknown;
        from?: unknown;
        content?: unknown;
        message?: { role?: unknown; content?: unknown };
      };

      // WOPR format: type="message"|"response", from="Username"|"WOPR", content="..."
      if (
        (entry.type === "message" || entry.type === "response") &&
        typeof entry.from === "string" &&
        typeof entry.content === "string"
      ) {
        const text = extractSessionText(entry.content);
        if (text) {
          // Determine if this is user or assistant based on type
          const label = entry.type === "response" ? "Assistant" : "User";
          collected.push(`${label}: ${text}`);
        }
        continue;
      }

      // OpenClaw/Claude format: type="message", message={role, content}
      if (entry.type === "message" && entry.message) {
        const msg = entry.message;
        if (typeof msg.role === "string" && (msg.role === "user" || msg.role === "assistant")) {
          const text = extractSessionText(msg.content);
          if (text) {
            const label = msg.role === "user" ? "User" : "Assistant";
            collected.push(`${label}: ${text}`);
          }
        }
      }
    }
    const content = collected.join("\n");
    return {
      path: sessionPathForFile(absPath),
      absPath,
      mtimeMs: stat.mtimeMs,
      size: stat.size,
      hash: hashText(content),
      content,
    };
  } catch {
    return null;
  }
}

/**
 * Read recent messages from session file for summary/slug generation
 */
export async function getRecentSessionContent(
  sessionFilePath: string,
  messageCount: number = 15,
): Promise<string | null> {
  try {
    const content = await fs.readFile(sessionFilePath, "utf-8");
    const lines = content.trim().split("\n");

    const allMessages: string[] = [];
    for (const line of lines) {
      try {
        const entry = JSON.parse(line);

        // WOPR format: type="message"|"response", from="Username"|"WOPR", content="..."
        if (
          (entry.type === "message" || entry.type === "response") &&
          typeof entry.from === "string" &&
          typeof entry.content === "string"
        ) {
          const text = entry.content;
          if (text && !text.startsWith("/")) {
            const role = entry.type === "response" ? "assistant" : "user";
            allMessages.push(`${role}: ${text}`);
          }
          continue;
        }

        // OpenClaw/Claude format: type="message", message={role, content}
        if (entry.type === "message" && entry.message) {
          const msg = entry.message;
          const role = msg.role;
          if ((role === "user" || role === "assistant") && msg.content) {
            const text = Array.isArray(msg.content)
              ? msg.content.find((c: { type?: string; text?: string }) => c.type === "text")?.text
              : msg.content;
            if (text && !text.startsWith("/")) {
              allMessages.push(`${role}: ${text}`);
            }
          }
        }
      } catch {
        // Skip invalid JSON lines
      }
    }

    const recentMessages = allMessages.slice(-messageCount);
    return recentMessages.join("\n");
  } catch {
    return null;
  }
}

// --- SQL-backed functions (WOP-1535) ---

/**
 * List active session names from SQL (replaces filesystem scan for .conversation.jsonl)
 */
export async function listSessionNames(storage: StorageApi, log: PluginLogger): Promise<string[]> {
  try {
    const rows = (await storage.raw(`SELECT name FROM sessions WHERE status = ?`, ["active"])) as Array<{
      name: string;
    }>;
    return rows.map((r) => r.name);
  } catch (error) {
    log.warn("[session-files] Failed to list active SQL-backed session names", error);
    return [];
  }
}

/**
 * Build a SessionFileEntry from SQL conversation data (replaces buildSessionEntry that read .jsonl files)
 */
export async function buildSessionEntryFromSql(
  sessionName: string,
  sessionApi: SessionApi,
  log: PluginLogger,
): Promise<SessionFileEntry | null> {
  try {
    const entries = await sessionApi.readConversationLog(sessionName);
    if (entries.length === 0) return null;

    const collected: string[] = [];
    for (const entry of entries) {
      if (entry.type === "context" || entry.type === "middleware") continue;
      if (entry.from === "system") continue;
      const text = extractSessionText(entry.content);
      if (text) {
        const label = entry.from === "WOPR" ? "Assistant" : "User";
        collected.push(`${label}: ${text}`);
      }
    }

    if (collected.length === 0) return null;
    const content = collected.join("\n");

    return {
      path: `sessions/${sessionName}`,
      absPath: `sql://sessions/${sessionName}`,
      mtimeMs: Date.now(),
      size: content.length,
      hash: hashText(content),
      content,
    };
  } catch (error) {
    log.warn(`[session-files] Failed to build session entry from SQL for "${sessionName}"`, error);
    return null;
  }
}

/**
 * Get recent session content from SQL (replaces getRecentSessionContent that read .jsonl files)
 */
export async function getRecentSessionContentFromSql(
  sessionName: string,
  sessionApi: SessionApi,
  log: PluginLogger,
  messageCount: number = 15,
): Promise<string | null> {
  try {
    const entries = await sessionApi.readConversationLog(sessionName);
    if (entries.length === 0) return null;

    const allMessages: string[] = [];
    for (const entry of entries) {
      if (entry.type === "context" || entry.type === "middleware") continue;
      if (entry.from === "system") continue;
      if (entry.content && !entry.content.startsWith("/")) {
        const role = entry.from === "WOPR" ? "assistant" : "user";
        allMessages.push(`${role}: ${entry.content}`);
      }
    }

    const recentMessages = allMessages.slice(-messageCount);
    return recentMessages.length > 0 ? recentMessages.join("\n") : null;
  } catch (error) {
    log.warn(`[session-files] Failed to get recent session content from SQL for "${sessionName}"`, error);
    return null;
  }
}
