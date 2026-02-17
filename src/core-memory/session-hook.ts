// Auto-save session conversations to memory/YYYY-MM-DD.md on session:destroy
import fs from "node:fs/promises";
import path from "node:path";
import type { PluginLogger } from "@wopr-network/plugin-types";

export async function createSessionDestroyHandler(params: {
  sessionsDir: string;
  log: PluginLogger;
}): Promise<(sessionName: string, reason: string) => Promise<void>> {
  return async (sessionName: string, _reason: string) => {
    try {
      const sessionDir = path.join(params.sessionsDir, sessionName);
      const conversationPath = path.join(sessionDir, `${sessionName}.conversation.jsonl`);

      // Check if conversation log exists
      try {
        await fs.access(conversationPath);
      } catch {
        return; // No conversation log to save
      }

      // Read conversation log
      const content = await fs.readFile(conversationPath, "utf-8");
      const lines = content.trim().split("\n");

      // Extract messages
      const messages: Array<{ role: string; text: string }> = [];
      for (const line of lines) {
        try {
          const entry = JSON.parse(line);

          // WOPR format: type="message"|"response", from="Username"|"WOPR", content="..."
          if (
            (entry.type === "message" || entry.type === "response") &&
            typeof entry.from === "string" &&
            typeof entry.content === "string"
          ) {
            const role = entry.type === "response" ? "WOPR" : entry.from;
            messages.push({ role, text: entry.content });
            continue;
          }

          // OpenClaw/Claude format: type="message", message={role, content}
          if (entry.type === "message" && entry.message) {
            const msg = entry.message;
            if (typeof msg.role === "string" && (msg.role === "user" || msg.role === "assistant")) {
              const text = Array.isArray(msg.content)
                ? msg.content.find((c: { type?: string; text?: string }) => c.type === "text")?.text
                : msg.content;
              if (text) {
                const role = msg.role === "user" ? "User" : "Assistant";
                messages.push({ role, text });
              }
            }
          }
        } catch {
          // Skip invalid JSON lines
        }
      }

      if (messages.length === 0) {
        return; // No messages to save
      }

      // Save to today's memory file
      const memoryDir = path.join(sessionDir, "memory");
      await fs.mkdir(memoryDir, { recursive: true });

      const today = new Date().toISOString().split("T")[0];
      const memoryFile = path.join(memoryDir, `${today}.md`);

      // Format conversation
      const formattedMessages = messages.map((msg) => `**${msg.role}**: ${msg.text}`).join("\n\n");

      const header = `## Session: ${sessionName}\n\n`;
      const footer = `\n\n---\n\n`;

      // Append to existing file or create new
      try {
        const existing = await fs.readFile(memoryFile, "utf-8");
        await fs.writeFile(memoryFile, existing + header + formattedMessages + footer);
      } catch {
        await fs.writeFile(memoryFile, header + formattedMessages + footer);
      }

      params.log.info(`[session-hook] Saved session ${sessionName} to ${today}.md`);
    } catch (err) {
      params.log.warn(
        `[session-hook] Failed to save session ${sessionName}: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
  };
}
