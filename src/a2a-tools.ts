/**
 * A2A Memory Tools - Moved from core to memory-semantic plugin
 *
 * These tools provide memory operations for agents via MCP/A2A.
 */

import { existsSync, lstatSync, mkdirSync, readdirSync, readFileSync, realpathSync, writeFileSync } from "node:fs";
import fs from "node:fs/promises";
import { isAbsolute, join, resolve, sep } from "node:path";
import type { WOPRPluginContext } from "@wopr-network/plugin-types";
import type { MemoryIndexManager } from "./core-memory/manager.js";
import { parseTemporalFilter } from "./core-memory/types.js";

/** Duck-type interface for plugin contexts that expose tool registration. */
interface ContextWithTools extends WOPRPluginContext {
  registerTool?: unknown;
  unregisterTool?: unknown;
}

/** Maximum allowed byte length for self_reflect content fields. */
const SELF_REFLECT_MAX_BYTES = 65_536; // 64 KB

/** Maximum allowed byte length for the self_reflect section header. */
const SELF_REFLECT_SECTION_MAX_BYTES = 256;

/** Upper bound for search results to prevent FTS5 full-scan DoS. */
export const MAX_SEARCH_RESULTS = 100;

/** Maximum allowed byte length for memory_write content. */
export const MEMORY_WRITE_MAX_BYTES = 1_048_576; // 1 MB

/** Separator inserted between existing content and new content on append. */
const MEMORY_APPEND_SEPARATOR = "\n\n";
const MEMORY_APPEND_SEPARATOR_BYTES = Buffer.byteLength(MEMORY_APPEND_SEPARATOR, "utf-8");

/** Thrown when a path escapes its allowed base directory. */
export class PathTraversalError extends Error {
  constructor(message = "Path outside allowed directory") {
    super(message);
    this.name = "PathTraversalError";
  }
}

/** Regex for safe session names: alphanumeric, hyphens, underscores, 1-64 chars */
const SAFE_SESSION_NAME = /^[a-zA-Z0-9_-]{1,64}$/;

/** Windows reserved device names (case-insensitive) */
const WINDOWS_RESERVED = /^(con|prn|aux|nul|com[0-9]|lpt[0-9])$/i;

/**
 * Validate that a session name is a safe filesystem identifier.
 * Rejects null bytes, path separators, special characters, Windows reserved
 * names, empty strings, and names longer than 64 characters.
 */
export function validateSessionName(name: string): void {
  if (!SAFE_SESSION_NAME.test(name) || WINDOWS_RESERVED.test(name)) {
    throw new PathTraversalError(
      `Invalid session name. Must match ${SAFE_SESSION_NAME} and not be a Windows reserved name.`,
    );
  }
}

/**
 * Throw if filePath escapes baseDir.
 * Uses realpathSync to resolve symlinks so symlink-based escapes are also caught.
 * For not-yet-existing paths, resolves the parent directory then appends the filename.
 */
function assertWithinBase(baseDir: string, filePath: string): void {
  const resolvedBase = resolve(baseDir);
  const baseReal = existsSync(resolvedBase) ? realpathSync(resolvedBase) : resolvedBase;

  const resolvedTarget = resolve(filePath);

  // Reject symlinks that exist
  if (existsSync(resolvedTarget)) {
    const stat = lstatSync(resolvedTarget);
    if (stat.isSymbolicLink()) {
      throw new PathTraversalError();
    }
  }

  let targetReal: string;
  if (existsSync(resolvedTarget)) {
    targetReal = realpathSync(resolvedTarget);
  } else {
    // File doesn't exist yet — resolve the parent to catch symlinked parent dirs
    const parentDir = resolve(resolvedTarget, "..");
    const parentReal = existsSync(parentDir) ? realpathSync(parentDir) : parentDir;
    const filename = resolvedTarget.split(sep).pop() ?? "";
    targetReal = join(parentReal, filename);
  }

  if (targetReal !== baseReal && !targetReal.startsWith(baseReal + sep)) {
    throw new PathTraversalError();
  }
}

/**
 * Resolve a memory filename to an absolute path.
 * Checks global identity memory first, then falls back to session-specific memory.
 * Throws PathTraversalError if filename is absolute or escapes its base directory.
 */
function resolveMemoryFile(
  sessionDir: string,
  filename: string,
  globalMemoryDir: string,
): { path: string; exists: boolean; isGlobal: boolean } {
  if (isAbsolute(filename)) throw new PathTraversalError();
  const globalPath = join(globalMemoryDir, filename);
  assertWithinBase(globalMemoryDir, globalPath);
  if (existsSync(globalPath)) {
    return { path: globalPath, exists: true, isGlobal: true };
  }
  const sessionPath = join(sessionDir, "memory", filename);
  assertWithinBase(join(sessionDir, "memory"), sessionPath);
  if (existsSync(sessionPath)) {
    return { path: sessionPath, exists: true, isGlobal: false };
  }
  return { path: sessionPath, exists: false, isGlobal: false };
}

/**
 * Resolve a root-level identity filename (e.g. SOUL.md, IDENTITY.md) to an absolute path.
 * Checks global identity directory first, then falls back to the session root.
 * Throws PathTraversalError if filename is absolute or escapes its base directory.
 */
function resolveRootFile(
  sessionDir: string,
  filename: string,
  globalIdentityDir: string,
): { path: string; exists: boolean; isGlobal: boolean } {
  if (isAbsolute(filename)) throw new PathTraversalError();
  const globalPath = join(globalIdentityDir, filename);
  assertWithinBase(globalIdentityDir, globalPath);
  if (existsSync(globalPath)) {
    return { path: globalPath, exists: true, isGlobal: true };
  }
  const sessionPath = join(sessionDir, filename);
  assertWithinBase(sessionDir, sessionPath);
  if (existsSync(sessionPath)) {
    return { path: sessionPath, exists: true, isGlobal: false };
  }
  return { path: sessionPath, exists: false, isGlobal: false };
}

/**
 * Return a deduplicated list of all .md filenames available in the global memory
 * directory and the session-specific memory directory. Session files shadow global ones.
 */
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
 * Discover session directories that contain a memory/ subdirectory.
 * Moved from core src/memory/index.ts (WOP-1774).
 */
export async function discoverSessionMemoryDirs(): Promise<string[]> {
  const sessionsBase = join(process.env.WOPR_HOME || "", "sessions");
  const dirs: string[] = [];
  try {
    const entries = await fs.readdir(sessionsBase, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      const sessionDir = join(sessionsBase, entry.name);
      const memDir = join(sessionDir, "memory");
      try {
        const stat = await fs.stat(memDir);
        if (stat.isDirectory()) dirs.push(sessionDir);
      } catch {
        /* memory dir does not exist */
      }
    }
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code !== "ENOENT") throw err;
    /* sessions dir does not exist — return empty */
  }
  return dirs;
}

/**
 * Register A2A memory tools with the plugin context
 */
export function registerMemoryTools(
  ctx: WOPRPluginContext,
  memoryManager: MemoryIndexManager,
  instanceId?: string,
): void {
  // A2A tools require registerTool method (not yet in @wopr-network/plugin-types v0.2.0)
  if (typeof (ctx as ContextWithTools).registerTool !== "function") {
    ctx.log.warn(
      "[memory-semantic] ctx.registerTool not available — A2A memory tools will not be registered. " +
        "Upgrade @wopr-network/plugin-types or ensure the host provides registerTool.",
    );
    return;
  }

  const GLOBAL_IDENTITY_DIR = process.env.WOPR_GLOBAL_IDENTITY || "/data/identity";
  const GLOBAL_MEMORY_DIR = join(GLOBAL_IDENTITY_DIR, "memory");
  const SESSIONS_DIR = join(process.env.WOPR_HOME || "", "sessions");

  // Helper to get session dir from context — validates sessionName cannot escape SESSIONS_DIR
  const getSessionDir = (sessionName: string) => {
    validateSessionName(sessionName);
    const dir = join(SESSIONS_DIR, sessionName);
    assertWithinBase(SESSIONS_DIR, dir);
    return dir;
  };

  // Files that live at the session root (not inside memory/)
  const ROOT_FILES = ["SOUL.md", "IDENTITY.md", "MEMORY.md", "USER.md", "AGENTS.md"];

  // Type assertion after guard check
  const api = ctx as ContextWithTools & { registerTool: (...args: unknown[]) => unknown };

  // memory_read tool
  api.registerTool({
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
    handler: async (args: { file?: string; from?: number; lines?: number; days?: number }, context: any) => {
      try {
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
          // Helper: collect date-named .md files from a base dir, skipping symlinks
          const collectDailyFromDir = (baseDir: string, dest: typeof dailyFiles) => {
            if (!existsSync(baseDir)) return;
            for (const f of readdirSync(baseDir).filter((f: string) => /^\d{4}-\d{2}-\d{2}\.md$/.test(f))) {
              const fullPath = join(baseDir, f);
              try {
                assertWithinBase(baseDir, fullPath);
                const stat = lstatSync(fullPath);
                if (stat.isSymbolicLink() || !stat.isFile()) continue;
              } catch {
                continue;
              }
              dest.push({ name: f, path: fullPath });
            }
          };
          collectDailyFromDir(GLOBAL_MEMORY_DIR, dailyFiles);
          const sessionMemoryDir = join(sessionDir, "memory");
          const sessionDaily: typeof dailyFiles = [];
          collectDailyFromDir(sessionMemoryDir, sessionDaily);
          for (const entry of sessionDaily) {
            const idx = dailyFiles.findIndex((d) => d.name === entry.name);
            if (idx >= 0) dailyFiles[idx] = entry;
            else dailyFiles.push(entry);
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

        let filePath: string;
        if (ROOT_FILES.includes(file)) {
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
      } catch (err) {
        if (err instanceof PathTraversalError) {
          return { content: [{ type: "text", text: err.message }], isError: true };
        }
        throw err;
      }
    },
  });

  // memory_write tool
  api.registerTool({
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
    handler: async (args: { file: string; content: string; append?: boolean }, context: any) => {
      try {
        const { file, content, append } = args;

        // Enforce content size limit to prevent disk exhaustion
        const maxBytes = context.config?.maxWriteBytes ?? MEMORY_WRITE_MAX_BYTES;
        const contentBytes = Buffer.byteLength(content, "utf-8");
        if (contentBytes > maxBytes) {
          return {
            content: [
              { type: "text", text: `Content exceeds maximum size of ${maxBytes} bytes (got ${contentBytes})` },
            ],
            isError: true,
          };
        }

        const sessionName = context.sessionName || "default";
        const sessionDir = getSessionDir(sessionName);
        const memoryDir = join(sessionDir, "memory");
        if (!existsSync(memoryDir)) mkdirSync(memoryDir, { recursive: true });

        let filename = file;
        if (file === "today") filename = `${new Date().toISOString().split("T")[0]}.md`;

        const filePath = ROOT_FILES.includes(filename) ? join(sessionDir, filename) : join(memoryDir, filename);
        assertWithinBase(ROOT_FILES.includes(filename) ? sessionDir : memoryDir, filePath);

        // Guard: reject symlinks at write target (closes TOCTOU window after assertWithinBase).
        // Use lstatSync (not existsSync) so dangling symlinks are also caught — existsSync
        // follows the link and returns false for dangling targets, allowing a bypass.
        try {
          const stat = lstatSync(filePath);
          if (stat.isSymbolicLink()) {
            throw new PathTraversalError();
          }
        } catch (err) {
          if ((err as NodeJS.ErrnoException).code !== "ENOENT") throw err;
          // ENOENT means the path doesn't exist yet — fine, we're creating it
        }

        const shouldAppend = append !== undefined ? append : filename.match(/^\d{4}-\d{2}-\d{2}\.md$/);

        if (shouldAppend && existsSync(filePath)) {
          const existingBytes = lstatSync(filePath).size;
          const combinedBytes = existingBytes + MEMORY_APPEND_SEPARATOR_BYTES + contentBytes;
          if (combinedBytes > maxBytes) {
            return {
              content: [
                {
                  type: "text",
                  text: `Appended content exceeds maximum size of ${maxBytes} bytes (would be ${combinedBytes})`,
                },
              ],
              isError: true,
            };
          }
          writeFileSync(filePath, `${MEMORY_APPEND_SEPARATOR}${content}`, { flag: "a" });
        } else {
          writeFileSync(filePath, content);
        }

        return { content: [{ type: "text", text: `${shouldAppend ? "Appended to" : "Wrote"} ${filename}` }] };
      } catch (err) {
        if (err instanceof PathTraversalError) {
          return { content: [{ type: "text", text: err.message }], isError: true };
        }
        throw err;
      }
    },
  });

  // memory_search tool
  api.registerTool({
    name: "memory_search",
    description:
      "Search memory files. Uses FTS5 keyword search by default; semantic/vector search available via wopr-plugin-memory-semantic. Supports temporal filtering.",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Search query" },
        maxResults: {
          type: "number",
          description: "Maximum results (default: 10, min: 1, max: 100)",
          minimum: 1,
          maximum: MAX_SEARCH_RESULTS,
        },
        minScore: { type: "number", description: "Minimum relevance score 0-1 (default: 0.35)" },
        temporal: {
          type: "string",
          description: 'Time filter: relative ("24h", "7d") or date range ("2026-01-01", "2026-01-01 to 2026-01-05")',
        },
      },
      required: ["query"],
    },
    handler: async (
      args: { query: string; maxResults?: number; minScore?: number; temporal?: string },
      _context: any,
    ) => {
      const { query, maxResults: rawMax = 10, minScore = 0.35, temporal: temporalExpr } = args;
      const safeRaw = Number.isFinite(rawMax) && rawMax > 0 ? Math.floor(rawMax) : 10;
      const maxResults = Math.min(safeRaw, MAX_SEARCH_RESULTS);
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
        // Use the memory manager to search — scoped to this instance
        const results = await memoryManager.search(query, { maxResults, minScore, temporal, instanceId });

        if (results.length === 0) {
          const temporalNote = temporalExpr ? ` within time range "${temporalExpr}"` : "";
          return { content: [{ type: "text", text: `No matches found for "${query}"${temporalNote}` }] };
        }

        const formatted = results
          .map(
            (r, i) =>
              `[${i + 1}] ${r.source}/${r.path}:${r.startLine}-${r.endLine} (score: ${r.score.toFixed(2)})\n${r.snippet}`,
          )
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
  api.registerTool({
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
    handler: async (args: { path: string; from?: number; lines?: number }, context: any) => {
      try {
        const { path: relPath, from, lines: lineCount } = args;
        const sessionName = context.sessionName || "default";
        const sessionDir = getSessionDir(sessionName);
        const memoryDir = join(sessionDir, "memory");

        let filePath = join(sessionDir, relPath);
        assertWithinBase(sessionDir, filePath);
        if (!existsSync(filePath)) {
          filePath = join(memoryDir, relPath);
          assertWithinBase(memoryDir, filePath);
        }
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
      } catch (err) {
        if (err instanceof PathTraversalError) {
          return { content: [{ type: "text", text: err.message }], isError: true };
        }
        throw err;
      }
    },
  });

  // self_reflect tool
  api.registerTool({
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
    handler: async (args: { reflection?: string; tattoo?: string; section?: string }, context: any) => {
      try {
        const { reflection, tattoo, section } = args;
        if (!reflection && !tattoo) {
          return { content: [{ type: "text", text: "Provide 'reflection' or 'tattoo'" }], isError: true };
        }

        // Enforce content size limit to prevent disk exhaustion
        if (reflection && Buffer.byteLength(reflection, "utf-8") > SELF_REFLECT_MAX_BYTES) {
          return {
            content: [
              { type: "text", text: `reflection exceeds maximum allowed size of ${SELF_REFLECT_MAX_BYTES} bytes` },
            ],
            isError: true,
          };
        }
        if (tattoo && Buffer.byteLength(tattoo, "utf-8") > SELF_REFLECT_MAX_BYTES) {
          return {
            content: [{ type: "text", text: `tattoo exceeds maximum allowed size of ${SELF_REFLECT_MAX_BYTES} bytes` }],
            isError: true,
          };
        }
        if (section && Buffer.byteLength(section, "utf-8") > SELF_REFLECT_SECTION_MAX_BYTES) {
          return {
            content: [
              {
                type: "text",
                text: `section exceeds maximum allowed size of ${SELF_REFLECT_SECTION_MAX_BYTES} bytes`,
              },
            ],
            isError: true,
          };
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
      } catch (err) {
        if (err instanceof PathTraversalError) {
          return { content: [{ type: "text", text: err.message }], isError: true };
        }
        throw err;
      }
    },
  });

  ctx.log.info("[memory-semantic] Registered 5 A2A memory tools");
}

const TOOL_NAMES = ["memory_read", "memory_write", "memory_search", "memory_get", "self_reflect"];

/**
 * Unregister all A2A memory tools from the plugin context.
 * Requires ctx to have an unregisterTool method.
 */
export function unregisterMemoryTools(ctx: WOPRPluginContext): void {
  if (typeof (ctx as ContextWithTools).unregisterTool !== "function") {
    ctx.log.warn("[memory-semantic] ctx.unregisterTool not available — A2A memory tools cannot be unregistered.");
    return;
  }
  const api = ctx as ContextWithTools & { unregisterTool: (name: string) => void };
  for (const name of TOOL_NAMES) {
    try {
      api.unregisterTool(name);
    } catch {
      /* tool may not have been registered */
    }
  }
  ctx.log.info("[memory-semantic] Unregistered A2A memory tools");
}
