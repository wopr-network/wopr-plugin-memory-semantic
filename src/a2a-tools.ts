/**
 * A2A Memory Tools - Moved from core to memory-semantic plugin
 *
 * These tools provide memory operations for agents via MCP/A2A.
 */

import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { WOPRPluginContext } from "@wopr-network/plugin-types";
import { MemoryIndexManager } from "./core-memory/manager.js";
import { parseTemporalFilter } from "./core-memory/types.js";

// Helper to resolve memory files
function resolveMemoryFile(
  sessionDir: string,
  filename: string,
  globalMemoryDir: string,
): { path: string; exists: boolean; isGlobal: boolean } {
  const globalPath = join(globalMemoryDir, filename);
  if (existsSync(globalPath)) {
    return { path: globalPath, exists: true, isGlobal: true };
  }
  const sessionPath = join(sessionDir, "memory", filename);
  if (existsSync(sessionPath)) {
    return { path: sessionPath, exists: true, isGlobal: false };
  }
  return { path: sessionPath, exists: false, isGlobal: false };
}

function resolveRootFile(
  sessionDir: string,
  filename: string,
  globalIdentityDir: string,
): { path: string; exists: boolean; isGlobal: boolean } {
  const globalPath = join(globalIdentityDir, filename);
  if (existsSync(globalPath)) {
    return { path: globalPath, exists: true, isGlobal: true };
  }
  const sessionPath = join(sessionDir, filename);
  if (existsSync(sessionPath)) {
    return { path: sessionPath, exists: true, isGlobal: false };
  }
  return { path: sessionPath, exists: false, isGlobal: false };
}

function listAllMemoryFiles(sessionDir: string, globalMemoryDir: string): string[] {
  const files = new Set<string>();
  if (existsSync(globalMemoryDir)) {
    for (const f of readdirSync(globalMemoryDir)) {
      if (f.endsWith(".md")) files.add(f);
    }
  }
  const sessionMemoryDir = join(sessionDir, "memory");
  if (existsSync(sessionMemoryDir)) {
    for (const f of readdirSync(sessionMemoryDir)) {
      if (f.endsWith(".md")) files.add(f);
    }
  }
  return [...files];
}

/**
 * Register A2A memory tools with the plugin context
 */
export function registerMemoryTools(
  ctx: WOPRPluginContext,
  memoryManager: MemoryIndexManager,
): void {
  const GLOBAL_IDENTITY_DIR = process.env.WOPR_GLOBAL_IDENTITY || "/data/identity";
  const GLOBAL_MEMORY_DIR = join(GLOBAL_IDENTITY_DIR, "memory");
  const SESSIONS_DIR = join(process.env.WOPR_HOME || "", "sessions");

  // Helper to get session dir from context
  const getSessionDir = (sessionName: string) => join(SESSIONS_DIR, sessionName);

  // memory_read tool
  ctx.registerTool({
    name: "memory_read",
    description:
      "Read a memory file. Checks global identity first, then session-specific. Supports daily logs, SELF.md, or topic files.",
    inputSchema: {
      type: "object",
      properties: {
        file: { type: "string", description: "Filename to read (e.g., 'SELF.md', '2026-01-24.md')" },
        from: { type: "number", description: "Starting line number (1-indexed)" },
        lines: { type: "number", description: "Number of lines to read" },
        days: { type: "number", description: "For daily logs: read last N days (default: 7)" },
      },
    },
    handler: async (args: { file?: string; from?: number; lines?: number; days?: number }, context) => {
      const { file, days = 7, from, lines: lineCount } = args;
      const sessionName = context.sessionName || "default";
      const sessionDir = getSessionDir(sessionName);

      if (!file) {
        const files: string[] = listAllMemoryFiles(sessionDir, GLOBAL_MEMORY_DIR);
        for (const f of ["SOUL.md", "IDENTITY.md", "MEMORY.md", "USER.md"]) {
          const resolved = resolveRootFile(sessionDir, f, GLOBAL_IDENTITY_DIR);
          if (resolved.exists && !files.includes(f)) files.push(f);
        }
        return {
          content: [
            {
              type: "text",
              text: files.length > 0 ? `Available memory files:\n${files.join("\n")}` : "No memory files found.",
            },
          ],
        };
      }

      if (file === "recent" || file === "daily") {
        const dailyFiles: { name: string; path: string }[] = [];
        if (existsSync(GLOBAL_MEMORY_DIR)) {
          for (const f of readdirSync(GLOBAL_MEMORY_DIR).filter((f: string) => f.match(/^\d{4}-\d{2}-\d{2}\.md$/))) {
            dailyFiles.push({ name: f, path: join(GLOBAL_MEMORY_DIR, f) });
          }
        }
        const sessionMemoryDir = join(sessionDir, "memory");
        if (existsSync(sessionMemoryDir)) {
          readdirSync(sessionMemoryDir)
            .filter((f: string) => f.match(/^\d{4}-\d{2}-\d{2}\.md$/))
            .forEach((f: string) => {
              const idx = dailyFiles.findIndex((d) => d.name === f);
              if (idx >= 0) dailyFiles[idx].path = join(sessionMemoryDir, f);
              else dailyFiles.push({ name: f, path: join(sessionMemoryDir, f) });
            });
        }
        dailyFiles.sort((a, b) => a.name.localeCompare(b.name));
        const recent = dailyFiles.slice(-days);
        if (recent.length === 0) return { content: [{ type: "text", text: "No daily memory files yet." }] };
        const contents = recent
          .map(({ name, path }) => {
            const content = readFileSync(path, "utf-8");
            return `## ${name.replace(".md", "")}\n\n${content}`;
          })
          .join("\n\n---\n\n");
        return { content: [{ type: "text", text: contents }] };
      }

      const rootFiles = ["SOUL.md", "IDENTITY.md", "MEMORY.md", "USER.md", "AGENTS.md"];
      let filePath: string;
      if (rootFiles.includes(file)) {
        const resolved = resolveRootFile(sessionDir, file, GLOBAL_IDENTITY_DIR);
        if (!resolved.exists) return { content: [{ type: "text", text: `File not found: ${file}` }], isError: true };
        filePath = resolved.path;
      } else {
        const resolved = resolveMemoryFile(sessionDir, file, GLOBAL_MEMORY_DIR);
        if (!resolved.exists) return { content: [{ type: "text", text: `File not found: ${file}` }], isError: true };
        filePath = resolved.path;
      }

      const content = readFileSync(filePath, "utf-8");
      if (from !== undefined && from > 0) {
        const allLines = content.split("\n");
        const startIdx = Math.max(0, from - 1);
        const endIdx = lineCount !== undefined ? Math.min(allLines.length, startIdx + lineCount) : allLines.length;
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(
                {
                  path: file,
                  from: startIdx + 1,
                  to: endIdx,
                  totalLines: allLines.length,
                  text: allLines.slice(startIdx, endIdx).join("\n"),
                },
                null,
                2,
              ),
            },
          ],
        };
      }
      return { content: [{ type: "text", text: content }] };
    },
  });

  // memory_write tool
  ctx.registerTool({
    name: "memory_write",
    description: "Write to a memory file. Creates memory/ directory if needed.",
    inputSchema: {
      type: "object",
      properties: {
        file: { type: "string", description: "Filename (e.g., 'today' for today's log, 'SELF.md')" },
        content: { type: "string", description: "Content to write or append" },
        append: { type: "boolean", description: "If true, append instead of replacing" },
      },
      required: ["file", "content"],
    },
    handler: async (args: { file: string; content: string; append?: boolean }, context) => {
      const { file, content, append } = args;
      const sessionName = context.sessionName || "default";
      const sessionDir = getSessionDir(sessionName);
      const memoryDir = join(sessionDir, "memory");
      if (!existsSync(memoryDir)) mkdirSync(memoryDir, { recursive: true });

      let filename = file;
      if (file === "today") filename = `${new Date().toISOString().split("T")[0]}.md`;

      const rootFiles = ["SOUL.md", "IDENTITY.md", "MEMORY.md", "USER.md", "AGENTS.md"];
      const filePath = rootFiles.includes(filename) ? join(sessionDir, filename) : join(memoryDir, filename);
      const shouldAppend = append !== undefined ? append : filename.match(/^\d{4}-\d{2}-\d{2}\.md$/);

      if (shouldAppend && existsSync(filePath)) {
        const existing = readFileSync(filePath, "utf-8");
        writeFileSync(filePath, `${existing}\n\n${content}`);
      } else {
        writeFileSync(filePath, content);
      }

      return { content: [{ type: "text", text: `${shouldAppend ? "Appended to" : "Wrote"} ${filename}` }] };
    },
  });

  // memory_search tool
  ctx.registerTool({
    name: "memory_search",
    description:
      "Search memory files. Uses FTS5 keyword search by default; semantic/vector search available via wopr-plugin-memory-semantic. Supports temporal filtering.",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Search query" },
        maxResults: { type: "number", description: "Maximum results (default: 10)" },
        minScore: { type: "number", description: "Minimum relevance score 0-1 (default: 0.35)" },
        temporal: {
          type: "string",
          description: 'Time filter: relative ("24h", "7d") or date range ("2026-01-01", "2026-01-01 to 2026-01-05")',
        },
      },
      required: ["query"],
    },
    handler: async (args: { query: string; maxResults?: number; minScore?: number; temporal?: string }, _context) => {
      const { query, maxResults = 10, minScore = 0.35, temporal: temporalExpr } = args;
      const parsedTemporal = temporalExpr ? parseTemporalFilter(temporalExpr) : null;
      if (temporalExpr && !parsedTemporal) {
        return {
          content: [
            {
              type: "text",
              text: `Invalid temporal filter "${temporalExpr}". Examples: "24h", "7d", "last 3 days", "2026-01-01"`,
            },
          ],
        };
      }
      const temporal = parsedTemporal ?? undefined;

      try {
        // Use the memory manager to search
        const results = await memoryManager.search(query, { maxResults, minScore, temporal });

        if (results.length === 0) {
          const temporalNote = temporalExpr ? ` within time range "${temporalExpr}"` : "";
          return { content: [{ type: "text", text: `No matches found for "${query}"${temporalNote}` }] };
        }

        const formatted = results
          .map((r, i) => `[${i + 1}] ${r.source}/${r.path}:${r.startLine}-${r.endLine} (score: ${r.score.toFixed(2)})\n${r.snippet}`)
          .join("\n\n---\n\n");
        const temporalNote = temporalExpr ? ` (filtered by: ${temporalExpr})` : "";

        return {
          content: [{ type: "text", text: `Found ${results.length} results${temporalNote}:\n\n${formatted}` }],
        };
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        ctx.log.warn(`Memory search failed: ${message}`);
        return {
          content: [{ type: "text", text: `Search failed: ${message}` }],
          isError: true,
        };
      }
    },
  });

  // memory_get tool
  ctx.registerTool({
    name: "memory_get",
    description: "Read a snippet from memory files with optional line range.",
    inputSchema: {
      type: "object",
      properties: {
        path: { type: "string", description: "Relative path from search results" },
        from: { type: "number", description: "Starting line number (1-indexed)" },
        lines: { type: "number", description: "Number of lines to read" },
      },
      required: ["path"],
    },
    handler: async (args: { path: string; from?: number; lines?: number }, context) => {
      const { path: relPath, from, lines: lineCount } = args;
      const sessionName = context.sessionName || "default";
      const sessionDir = getSessionDir(sessionName);
      const memoryDir = join(sessionDir, "memory");

      let filePath = join(sessionDir, relPath);
      if (!existsSync(filePath)) filePath = join(memoryDir, relPath);
      if (!existsSync(filePath))
        return { content: [{ type: "text", text: `File not found: ${relPath}` }], isError: true };

      const content = readFileSync(filePath, "utf-8");
      const allLines = content.split("\n");

      if (from !== undefined && from > 0) {
        const startIdx = Math.max(0, from - 1);
        const endIdx = lineCount !== undefined ? Math.min(allLines.length, startIdx + lineCount) : allLines.length;
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(
                {
                  path: relPath,
                  from: startIdx + 1,
                  to: endIdx,
                  totalLines: allLines.length,
                  text: allLines.slice(startIdx, endIdx).join("\n"),
                },
                null,
                2,
              ),
            },
          ],
        };
      }

      return {
        content: [
          {
            type: "text",
            text: JSON.stringify({ path: relPath, totalLines: allLines.length, text: content }, null, 2),
          },
        ],
      };
    },
  });

  // self_reflect tool
  ctx.registerTool({
    name: "self_reflect",
    description: "Add a reflection to SELF.md (private journal). Use for tattoos and daily reflections.",
    inputSchema: {
      type: "object",
      properties: {
        reflection: { type: "string", description: "The reflection to record" },
        tattoo: { type: "string", description: "A persistent identity marker" },
        section: { type: "string", description: "Section header (default: today's date)" },
      },
    },
    handler: async (args: { reflection?: string; tattoo?: string; section?: string }, context) => {
      const { reflection, tattoo, section } = args;
      if (!reflection && !tattoo) {
        return { content: [{ type: "text", text: "Provide 'reflection' or 'tattoo'" }], isError: true };
      }

      const sessionName = context.sessionName || "default";
      const sessionDir = getSessionDir(sessionName);
      const memoryDir = join(sessionDir, "memory");
      const selfPath = join(memoryDir, "SELF.md");

      if (!existsSync(memoryDir)) mkdirSync(memoryDir, { recursive: true });
      if (!existsSync(selfPath)) writeFileSync(selfPath, "# SELF.md — Private Reflections\n\n");

      const existing = readFileSync(selfPath, "utf-8");
      const today = new Date().toISOString().split("T")[0];

      if (tattoo) {
        const lines = existing.split("\n");
        const tattooSection = lines.findIndex((l: string) => l.includes("## Tattoos"));
        if (tattooSection === -1) {
          const titleLine = lines.findIndex((l: string) => l.startsWith("# "));
          writeFileSync(
            selfPath,
            [...lines.slice(0, titleLine + 1), `\n## Tattoos\n\n- "${tattoo}"\n`, ...lines.slice(titleLine + 1)].join(
              "\n",
            ),
          );
        } else {
          const beforeTattoo = lines.slice(0, tattooSection + 1);
          const afterTattoo = lines.slice(tattooSection + 1);
          const insertPoint = afterTattoo.findIndex((l: string) => l.startsWith("## "));
          if (insertPoint === -1) afterTattoo.push(`- "${tattoo}"`);
          else afterTattoo.splice(insertPoint, 0, `- "${tattoo}"`);
          writeFileSync(selfPath, [...beforeTattoo, ...afterTattoo].join("\n"));
        }
        return { content: [{ type: "text", text: `Tattoo added: "${tattoo}"` }] };
      }

      if (reflection) {
        const sectionHeader = section || today;
        writeFileSync(selfPath, `${existing}\n---\n\n## ${sectionHeader}\n\n${reflection}\n`);
        return { content: [{ type: "text", text: `Reflection added under "${sectionHeader}"` }] };
      }

      return { content: [{ type: "text", text: "Nothing to add" }] };
    },
  });

  ctx.log.info("[memory-semantic] Registered 5 A2A memory tools");
}
