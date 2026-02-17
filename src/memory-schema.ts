/**
 * Memory plugin storage schema
 * Registers files, chunks, meta tables with the Storage API
 */

import { z } from "zod";
import type { StorageApi, PluginSchema } from "@wopr-network/plugin-types";

/**
 * Extended schema with optional migrate function for v0 → v1 migration
 */
export interface MigratablePluginSchema extends PluginSchema {
  migrate?: (fromVersion: number, toVersion: number, storage: StorageApi) => Promise<void>;
}

export const MEMORY_NAMESPACE = "memory";
export const MEMORY_SCHEMA_VERSION = 1;

// Use synthetic ID instead of composite PK (path, source) for Storage API compatibility
const filesSchema = z.object({
  id: z.string(), // sha256(path + ":" + source)
  path: z.string(),
  source: z.string(),
  hash: z.string(),
  mtime: z.number().int(),
  size: z.number().int(),
});

const chunksSchema = z.object({
  id: z.string(),
  path: z.string(),
  source: z.string(),
  start_line: z.number().int(),
  end_line: z.number().int(),
  hash: z.string(),
  model: z.string(),
  text: z.string(),
  updated_at: z.number().int(),
  // embedding is managed separately via ALTER TABLE (BLOB column)
});

const metaSchema = z.object({
  key: z.string(),
  value: z.string(),
});

export const memoryPluginSchema: MigratablePluginSchema = {
  namespace: MEMORY_NAMESPACE,
  version: MEMORY_SCHEMA_VERSION,
  tables: {
    files: {
      schema: filesSchema,
      primaryKey: "id",
      indexes: [
        { fields: ["path", "source"], unique: true },
        { fields: ["source"] },
      ],
    },
    chunks: {
      schema: chunksSchema,
      primaryKey: "id",
      indexes: [
        { fields: ["path"] },
        { fields: ["source"] },
      ],
    },
    meta: {
      schema: metaSchema,
      primaryKey: "key",
    },
  },
  migrate: async (fromVersion: number, toVersion: number, storage: StorageApi) => {
    // v0 → v1: Import data from old index.sqlite into wopr.sqlite
    if (fromVersion === 0 && toVersion === 1) {
      await migrateFromLegacyIndexSqlite(storage);
    }
  },
};

/**
 * Migrate data from legacy $WOPR_HOME/memory/index.sqlite to the new Storage API
 */
async function migrateFromLegacyIndexSqlite(storage: StorageApi): Promise<void> {
  const { existsSync } = await import("node:fs");
  const { join } = await import("node:path");
  const { renameSync } = await import("node:fs");

  const woprHome = process.env.WOPR_HOME;
  if (!woprHome) return;

  const legacyDbPath = join(woprHome, "memory", "index.sqlite");
  if (!existsSync(legacyDbPath)) return;

  // Use ATTACH DATABASE to read from old DB and copy to new tables
  try {
    await storage.raw(`ATTACH DATABASE ? AS legacy`, [legacyDbPath]);

    // Migrate files table (add synthetic ID)
    await storage.raw(`
      INSERT OR IGNORE INTO memory_files (id, path, source, hash, mtime, size)
      SELECT
        lower(hex(randomblob(16))),
        path, source, hash, mtime, size
      FROM legacy.files
    `);

    // Migrate chunks table
    await storage.raw(`
      INSERT OR IGNORE INTO memory_chunks (id, path, source, start_line, end_line, hash, model, text, updated_at)
      SELECT id, path, source, start_line, end_line, hash, model, text, updated_at
      FROM legacy.chunks
    `);

    // Migrate FTS5 content
    await storage
      .raw(
        `
      INSERT OR IGNORE INTO memory_chunks_fts (text, id, path, source, model, start_line, end_line)
      SELECT text, id, path, source, model, start_line, end_line
      FROM legacy.chunks_fts
    `,
      )
      .catch(() => {
        /* old FTS5 may not exist */
      });

    // Migrate meta table
    await storage.raw(`
      INSERT OR IGNORE INTO memory_meta (key, value)
      SELECT key, value FROM legacy.meta
    `);

    // Migrate embedding column if it exists
    await storage
      .raw(
        `
      UPDATE memory_chunks SET embedding = (
        SELECT legacy.chunks.embedding FROM legacy.chunks
        WHERE legacy.chunks.id = memory_chunks.id
      )
      WHERE EXISTS (
        SELECT 1 FROM legacy.chunks
        WHERE legacy.chunks.id = memory_chunks.id
        AND legacy.chunks.embedding IS NOT NULL
      )
    `,
      )
      .catch(() => {
        /* embedding column may not exist in legacy */
      });

    await storage.raw(`DETACH DATABASE legacy`);

    // Rename old DB to mark as migrated (don't delete — safety net)
    renameSync(legacyDbPath, `${legacyDbPath}.migrated`);
  } catch (err) {
    // If ATTACH fails, try manual approach or log and skip
    try {
      await storage.raw(`DETACH DATABASE legacy`);
    } catch {}
    throw err;
  }
}
