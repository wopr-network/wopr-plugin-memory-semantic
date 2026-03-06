/**
 * Identity tools: identity_get, identity_update
 * Moved from core (WOP-1773) into the memory-semantic plugin (WOP-1774).
 * Uses ctx.session.getContext/setContext instead of core session-context-repository.
 */

import type { WOPRPluginContext } from "@wopr-network/plugin-types";

interface ContextWithTools extends WOPRPluginContext {
  registerTool?: unknown;
  unregisterTool?: unknown;
}

interface SessionApi {
  getContext(sessionName: string, filename: string): Promise<string | null>;
  setContext(sessionName: string, filename: string, content: string, source: "global" | "session"): Promise<void>;
}

interface PluginContextWithSession extends WOPRPluginContext {
  session?: SessionApi;
}

const IDENTITY_TOOL_NAMES = ["identity_get", "identity_update"];

export function registerIdentityTools(ctx: WOPRPluginContext): void {
  if (typeof (ctx as ContextWithTools).registerTool !== "function") {
    ctx.log.warn("[memory-semantic] ctx.registerTool not available — identity tools will not be registered.");
    return;
  }

  const sessionApi = (ctx as PluginContextWithSession).session;
  if (!sessionApi) {
    ctx.log.warn("[memory-semantic] ctx.session not available — identity tools will not be registered.");
    return;
  }

  const api = ctx as ContextWithTools & { registerTool: (...args: unknown[]) => unknown };

  // identity_get
  api.registerTool({
    name: "identity_get",
    description: "Get current identity from IDENTITY.md. Checks session-specific first, then global.",
    inputSchema: {
      type: "object",
      properties: {},
    },
    handler: async (_args: Record<string, unknown>, context: any) => {
      const sessionName = context.sessionName || "default";

      // Try session-specific first
      let content = await sessionApi.getContext(sessionName, "IDENTITY.md");
      let isGlobal = false;

      // Fall back to global
      if (content === null) {
        content = await sessionApi.getContext("__global__", "IDENTITY.md");
        isGlobal = content !== null;
      }

      if (content === null) {
        return { content: [{ type: "text", text: "No IDENTITY.md found." }] };
      }

      const identity: Record<string, string> = {};
      const nameMatch = content.match(/[-*]\s*Name:\s*(.+)/i);
      const creatureMatch = content.match(/[-*]\s*Creature:\s*(.+)/i);
      const vibeMatch = content.match(/[-*]\s*Vibe:\s*(.+)/i);
      const emojiMatch = content.match(/[-*]\s*Emoji:\s*(.+)/i);
      if (nameMatch) identity.name = nameMatch[1].trim();
      if (creatureMatch) identity.creature = creatureMatch[1].trim();
      if (vibeMatch) identity.vibe = vibeMatch[1].trim();
      if (emojiMatch) identity.emoji = emojiMatch[1].trim();

      return {
        content: [
          {
            type: "text",
            text: JSON.stringify({ parsed: identity, raw: content, source: isGlobal ? "global" : "session" }, null, 2),
          },
        ],
      };
    },
  });

  // identity_update
  api.registerTool({
    name: "identity_update",
    description: "Update fields in IDENTITY.md.",
    inputSchema: {
      type: "object",
      properties: {
        name: { type: "string", description: "Agent name" },
        creature: { type: "string", description: "Entity type" },
        vibe: { type: "string", description: "Personality vibe" },
        emoji: { type: "string", description: "Identity emoji" },
        section: { type: "string", description: "Custom section name" },
        sectionContent: { type: "string", description: "Content for custom section" },
      },
    },
    handler: async (
      args: {
        name?: string;
        creature?: string;
        vibe?: string;
        emoji?: string;
        section?: string;
        sectionContent?: string;
      },
      context: any,
    ) => {
      const sessionName = context.sessionName || "default";
      const { name, creature, vibe, emoji, section, sectionContent } = args;

      // Read current content (session-specific, not global)
      let content = await sessionApi.getContext(sessionName, "IDENTITY.md");
      if (content === null) {
        content = "# IDENTITY.md - Agent Identity\n\n";
      }

      const updates: string[] = [];
      if (name) {
        content = content.replace(/[-*]\s*Name:\s*.+/i, `- Name: ${name}`);
        if (!content.includes("Name:")) content += `- Name: ${name}\n`;
        updates.push(`name: ${name}`);
      }
      if (creature) {
        content = content.replace(/[-*]\s*Creature:\s*.+/i, `- Creature: ${creature}`);
        if (!content.includes("Creature:")) content += `- Creature: ${creature}\n`;
        updates.push(`creature: ${creature}`);
      }
      if (vibe) {
        content = content.replace(/[-*]\s*Vibe:\s*.+/i, `- Vibe: ${vibe}`);
        if (!content.includes("Vibe:")) content += `- Vibe: ${vibe}\n`;
        updates.push(`vibe: ${vibe}`);
      }
      if (emoji) {
        content = content.replace(/[-*]\s*Emoji:\s*.+/i, `- Emoji: ${emoji}`);
        if (!content.includes("Emoji:")) content += `- Emoji: ${emoji}\n`;
        updates.push(`emoji: ${emoji}`);
      }
      if (section && sectionContent) {
        const safeSection = section.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
        const sectionRegex = new RegExp(`## ${safeSection}[\\s\\S]*?(?=\\n## |$)`, "i");
        const newSection = `## ${section}\n\n${sectionContent}\n`;
        if (content.match(sectionRegex)) content = content.replace(sectionRegex, newSection);
        else content += `\n${newSection}`;
        updates.push(`section: ${section}`);
      }

      if (updates.length === 0) {
        return { content: [{ type: "text", text: "No fields provided to update." }], isError: true };
      }

      await sessionApi.setContext(sessionName, "IDENTITY.md", content, "session");
      return { content: [{ type: "text", text: `Identity updated: ${updates.join(", ")}` }] };
    },
  });

  ctx.log.info("[memory-semantic] Registered 2 identity tools");
}

export function unregisterIdentityTools(ctx: WOPRPluginContext): void {
  if (typeof (ctx as ContextWithTools).unregisterTool !== "function") {
    ctx.log.warn("[memory-semantic] ctx.unregisterTool not available — identity tools cannot be unregistered.");
    return;
  }
  const api = ctx as ContextWithTools & { unregisterTool: (name: string) => void };
  for (const name of IDENTITY_TOOL_NAMES) {
    try {
      api.unregisterTool(name);
    } catch {
      /* tool may not have been registered */
    }
  }
  ctx.log.info("[memory-semantic] Unregistered identity tools");
}
