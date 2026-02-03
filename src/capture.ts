/**
 * Auto-capture logic for detecting important information in conversations
 */

import type { CaptureCandidate, MemoryCategory, SemanticMemoryConfig } from "./types.js";

// =============================================================================
// Capture Triggers - patterns that indicate something worth remembering
// =============================================================================

const CAPTURE_TRIGGERS = [
  // Explicit memory requests
  /remember (this|that)|don't forget|note that|keep in mind/i,
  /zapamatuj si|pamatuj/i, // Czech

  // Preferences
  /i (prefer|like|love|hate|want|need|always|never)/i,
  /my favorite|i usually|i tend to/i,
  /preferuji|radši|nechci/i, // Czech

  // Decisions
  /we (decided|agreed|chose|will use|should use)/i,
  /let's (use|go with|stick with)/i,
  /the plan is|going forward/i,
  /rozhodli jsme|budeme používat/i, // Czech

  // Personal info / Entities
  /my (name|email|phone|address|birthday) is/i,
  /i (am|work at|live in)/i,
  /\+\d{10,}/, // Phone numbers
  /[\w.-]+@[\w.-]+\.\w+/, // Emails

  // Facts and corrections
  /actually,? (it's|that's|the)/i,
  /the correct|the right way/i,
  /important:?/i,
];

// =============================================================================
// Skip Patterns - things that look capturable but aren't
// =============================================================================

function shouldSkip(text: string, config: SemanticMemoryConfig): boolean {
  // Length checks
  if (text.length < config.autoCapture.minLength) return true;
  if (text.length > config.autoCapture.maxLength) return true;

  // Skip injected context from memory recall
  if (text.includes("<relevant-memories>")) return true;
  if (text.includes("</relevant-memories>")) return true;

  // Skip system-generated content (XML-like tags)
  if (text.startsWith("<") && text.includes("</")) return true;

  // Skip agent summary responses (markdown formatting)
  if (text.includes("**") && text.includes("\n-") && text.split("\n").length > 5) return true;

  // Skip emoji-heavy responses (likely agent output, not user input)
  const emojiCount = (text.match(/[\u{1F300}-\u{1F9FF}]/gu) || []).length;
  if (emojiCount > 3) return true;

  // Skip code blocks
  if (text.includes("```")) return true;

  // Skip URLs (usually not worth storing as memories)
  if (/https?:\/\/\S+/.test(text) && text.split(/\s+/).length < 10) return true;

  return false;
}

// =============================================================================
// Category Detection
// =============================================================================

function detectCategory(text: string): MemoryCategory {
  const lower = text.toLowerCase();

  // Preferences
  if (/prefer|like|love|hate|want|need|favorite|usually|tend to/i.test(lower)) {
    return "preference";
  }

  // Decisions
  if (/decided|agreed|chose|will use|should use|going forward|the plan/i.test(lower)) {
    return "decision";
  }

  // Entities (names, contacts, etc)
  if (/\+\d{10,}|@[\w.-]+\.\w+|my name is|i am called|i work at|i live in/i.test(lower)) {
    return "entity";
  }

  // Facts
  if (/is|are|has|have|actually|correct|right way/i.test(lower)) {
    return "fact";
  }

  return "other";
}

// =============================================================================
// Importance Scoring
// =============================================================================

function scoreImportance(text: string, category: MemoryCategory): number {
  let score = 0.5; // Base score

  // Explicit memory requests are high importance
  if (/remember|don't forget|note that|keep in mind|important/i.test(text)) {
    score += 0.3;
  }

  // Category adjustments
  if (category === "preference") score += 0.1;
  if (category === "decision") score += 0.15;
  if (category === "entity") score += 0.2; // Contact info is important

  // Length adjustment (longer = more context = more important)
  if (text.length > 100) score += 0.05;
  if (text.length > 200) score += 0.05;

  return Math.min(1, Math.max(0, score));
}

// =============================================================================
// Main Capture Function
// =============================================================================

export function shouldCapture(text: string, config: SemanticMemoryConfig): boolean {
  if (!config.autoCapture.enabled) return false;
  if (shouldSkip(text, config)) return false;
  return CAPTURE_TRIGGERS.some((r) => r.test(text));
}

export function extractCaptureCandidate(text: string): CaptureCandidate {
  const category = detectCategory(text);
  const importance = scoreImportance(text, category);

  return {
    text: text.trim(),
    category,
    importance,
  };
}

/**
 * Extract capturable content from a conversation
 * Returns candidates sorted by importance
 */
export function extractFromConversation(
  messages: Array<{ role: string; content: string }>,
  config: SemanticMemoryConfig
): CaptureCandidate[] {
  const candidates: CaptureCandidate[] = [];

  for (const msg of messages) {
    // Focus on user messages - that's where preferences and facts come from
    // Also check assistant messages for decisions ("we decided", "let's use")
    if (msg.role !== "user" && msg.role !== "assistant") continue;

    const text = msg.content;
    if (!shouldCapture(text, config)) continue;

    candidates.push(extractCaptureCandidate(text));
  }

  // Sort by importance, take top N
  return candidates
    .sort((a, b) => b.importance - a.importance)
    .slice(0, config.autoCapture.maxPerConversation);
}
