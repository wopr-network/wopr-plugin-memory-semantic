# wopr-plugin-memory-semantic

Semantic memory plugin for WOPR — embeddings-based auto-recall and auto-capture for conversations.

## Commands

```bash
npm run build     # tsc
npm run check     # biome check + tsc --noEmit (run before committing)
npm run format    # biome format --write src/
npm test          # vitest run
```

## Architecture

```
src/
  index.ts        # Plugin entry — wires all memory components
  embeddings.ts   # Embedding generation (calls embedding provider)
  capture.ts      # Auto-capture: extracts memories from conversations
  recall.ts       # Auto-recall: injects relevant memories into context
  search.ts       # Semantic search over stored memories
  tool-enhancer.ts  # Wraps LLM tool calls with memory context
  webmcp.ts       # WebMCP endpoint for memory operations
  types.ts        # Plugin-local types
```

## Key Details

- Implements the `memory` capability from `@wopr-network/plugin-types`
- Requires an embedding model — either local (e.g. Ollama) or remote (OpenAI embeddings)
- Memories stored locally in SQLite via the plugin's storage allocation
- **Auto-capture**: after each conversation turn, extracts key facts and stores them as embeddings
- **Auto-recall**: before each LLM call, retrieves top-K relevant memories and prepends to context
- `tool-enhancer.ts` wraps tool results with memory context injection
- **Gotcha**: Embedding dimensions must be consistent — changing embedding models requires re-indexing all stored memories

## Plugin Contract

Imports only from `@wopr-network/plugin-types`. Never import from `@wopr-network/wopr` core.

## Issue Tracking

All issues in **Linear** (team: WOPR). Issue descriptions start with `**Repo:** wopr-network/wopr-plugin-memory-semantic`.
