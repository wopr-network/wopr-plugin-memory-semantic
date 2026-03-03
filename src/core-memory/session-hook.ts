// Auto-save session conversations to memory/YYYY-MM-DD.md on session:destroy
import type { PluginLogger } from "@wopr-network/plugin-types";
import type { SessionApi } from "../types.js";

export async function createSessionDestroyHandler(params: {
  sessionsDir: string;
  log: PluginLogger;
  sessionApi?: SessionApi;
}): Promise<(sessionName: string, reason: string) => Promise<void>> {
  return async (sessionName: string, _reason: string) => {
    try {
      if (!params.sessionApi) {
        params.log.warn(`[session-hook] ctx.session not available — skipping memory save for ${sessionName}`);
        return;
      }

      // Read conversation from SQL
      const entries = await params.sessionApi.readConversationLog(sessionName);

      if (entries.length === 0) {
        return; // No messages to save
      }

      // Extract messages (skip system/context entries)
      const messages: Array<{ role: string; text: string }> = [];
      for (const entry of entries) {
        if (entry.type === "context" || entry.type === "middleware") continue;

        let role: string;
        if (entry.from === "system") {
          continue; // skip system messages
        } else if (entry.from === "WOPR") {
          role = "WOPR";
        } else {
          role = entry.from;
        }
        messages.push({ role, text: entry.content });
      }

      if (messages.length === 0) {
        return;
      }

      // Format conversation
      const formattedMessages = messages.map((msg) => `**${msg.role}**: ${msg.text}`).join("\n\n");
      const header = `## Session: ${sessionName}\n\n`;
      const footer = `\n\n---\n\n`;
      const today = new Date().toISOString().split("T")[0];
      const filename = `memory/${today}.md`;

      // Read existing content and append, or create new
      const existing = await params.sessionApi.getContext(sessionName, filename);
      const content = existing ? existing + header + formattedMessages + footer : header + formattedMessages + footer;

      await params.sessionApi.setContext(sessionName, filename, content, "session");

      params.log.info(`[session-hook] Saved session ${sessionName} to ${today}.md`);
    } catch (err) {
      params.log.warn(
        `[session-hook] Failed to save session ${sessionName}: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
  };
}
