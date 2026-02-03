# wopr-plugin-memory-semantic

Semantic memory search with embeddings, auto-recall, and auto-capture for WOPR.

## Features

- **Vector Embeddings**: OpenAI, Gemini, or local (node-llama-cpp)
- **Hybrid Search**: Combines vector similarity with BM25 keyword search
- **Auto-Recall**: Automatically injects relevant memories before agent processing
- **Auto-Capture**: Extracts and stores important information from conversations

## Installation

```bash
wopr plugin install wopr-plugin-memory-semantic
```

## Configuration

In your WOPR config:

```yaml
plugins:
  memory-semantic:
    enabled: true
    config:
      provider: auto  # openai, gemini, local, or auto
      model: text-embedding-3-small

      autoRecall:
        enabled: true
        maxMemories: 5
        minScore: 0.4

      autoCapture:
        enabled: true
        maxPerConversation: 3
```

## How It Works

### Auto-Recall

Before each message is processed by the agent, this plugin:
1. Extracts a search query from the user's message
2. Searches for semantically relevant memories
3. Injects matching memories as context

### Auto-Capture

After each conversation turn, this plugin:
1. Scans messages for important information
2. Detects preferences, decisions, entities, and facts
3. Stores them as embeddings for future recall

### Capture Triggers

The plugin looks for patterns like:
- Explicit: "remember this", "don't forget", "note that"
- Preferences: "I prefer", "I like", "I always"
- Decisions: "we decided", "let's use", "the plan is"
- Personal info: "my name is", "I work at", phone numbers, emails

## API

```typescript
import plugin from 'wopr-plugin-memory-semantic';

// Search memories
const results = await plugin.search("project architecture");

// Manually capture something
await plugin.capture("T prefers tabs over spaces", "preference");

// Get current config
const config = plugin.getConfig();
```

## Embedding Providers

### OpenAI (default)
Requires `OPENAI_API_KEY` environment variable.

### Gemini
Requires `GOOGLE_API_KEY` or `GEMINI_API_KEY`.

### Local
Uses node-llama-cpp with a small embedding model. No API key needed, runs locally.

## Architecture

This plugin separates semantic search from WOPR core:
- **Core**: File-based memory with keyword search (FTS5)
- **This plugin**: Adds vector embeddings and semantic understanding

The plugin hooks into:
- `session:beforeInject` - for auto-recall
- `session:afterInject` - for auto-capture

## License

MIT
