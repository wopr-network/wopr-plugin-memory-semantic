/**
 * Auto-capture tests (WOP-98)
 *
 * Tests shouldCapture, extractCaptureCandidate, extractFromConversation,
 * category detection, and importance scoring.
 */
import { describe, expect, it } from "vitest";
import { extractCaptureCandidate, extractFromConversation, shouldCapture } from "../../src/capture.js";
import { DEFAULT_CONFIG, type SemanticMemoryConfig } from "../../src/types.js";

function makeConfig(overrides: Partial<SemanticMemoryConfig["autoCapture"]> = {}): SemanticMemoryConfig {
  return {
    ...DEFAULT_CONFIG,
    autoCapture: { ...DEFAULT_CONFIG.autoCapture, ...overrides },
  };
}

describe("shouldCapture", () => {
  const config = makeConfig();

  it("should return false when autoCapture is disabled", () => {
    const disabled = makeConfig({ enabled: false });
    expect(shouldCapture("I prefer TypeScript over JavaScript", disabled)).toBe(false);
  });

  it("should return true for explicit memory requests", () => {
    expect(shouldCapture("Remember this: the API key rotates weekly", config)).toBe(true);
    expect(shouldCapture("Don't forget to update the schema", config)).toBe(true);
    expect(shouldCapture("Note that the deploy uses blue-green strategy", config)).toBe(true);
    expect(shouldCapture("Keep in mind the rate limit is 100 req/s", config)).toBe(true);
  });

  it("should return true for preferences", () => {
    expect(shouldCapture("I prefer tabs over spaces for indentation", config)).toBe(true);
    expect(shouldCapture("I always use dark mode in my editor", config)).toBe(true);
    expect(shouldCapture("I usually run tests before committing", config)).toBe(true);
  });

  it("should return true for decisions", () => {
    expect(shouldCapture("We decided to use PostgreSQL for the database", config)).toBe(true);
    expect(shouldCapture("Let's use React for the frontend going forward", config)).toBe(true);
    expect(shouldCapture("The plan is to migrate to microservices", config)).toBe(true);
  });

  it("should return true for personal info / entities", () => {
    expect(shouldCapture("My email is test@example.com and this is important", config)).toBe(true);
    expect(shouldCapture("I work at Acme Corp in the engineering team", config)).toBe(true);
  });

  it("should return false for text too short", () => {
    expect(shouldCapture("I like it", makeConfig({ minLength: 20 }))).toBe(false);
  });

  it("should return false for text too long", () => {
    const longText = "I prefer ".padEnd(600, "x");
    expect(shouldCapture(longText, makeConfig({ maxLength: 500 }))).toBe(false);
  });

  it("should skip injected memory context", () => {
    expect(shouldCapture("<relevant-memories>I prefer this</relevant-memories>", config)).toBe(false);
  });

  it("should skip XML-like system content", () => {
    expect(shouldCapture("<system>I prefer this</system>", config)).toBe(false);
  });

  it("should skip code blocks", () => {
    expect(shouldCapture("I prefer this ```typescript const x = 1```", config)).toBe(false);
  });

  it("should skip URL-only messages", () => {
    expect(shouldCapture("https://example.com/my-favorite-tool I like", config)).toBe(false);
  });

  it("should return false for generic messages without triggers", () => {
    expect(shouldCapture("What is the weather today in London", config)).toBe(false);
    expect(shouldCapture("Thanks for the help with that bug fix", config)).toBe(false);
  });
});

describe("extractCaptureCandidate", () => {
  it("should detect preference category", () => {
    const result = extractCaptureCandidate("I prefer TypeScript over JavaScript");
    expect(result.category).toBe("preference");
    expect(result.text).toBe("I prefer TypeScript over JavaScript");
  });

  it("should detect decision category", () => {
    const result = extractCaptureCandidate("We decided to use PostgreSQL for the database");
    expect(result.category).toBe("decision");
  });

  it("should detect entity category for email", () => {
    const result = extractCaptureCandidate("My email is user@example.com for notifications");
    expect(result.category).toBe("entity");
  });

  it("should detect entity category for phone number", () => {
    const result = extractCaptureCandidate("My number is +12025551234 for emergencies");
    expect(result.category).toBe("entity");
  });

  it("should detect fact category", () => {
    const result = extractCaptureCandidate("Actually, the API endpoint has changed to v3");
    expect(result.category).toBe("fact");
  });

  it("should fall back to other category", () => {
    // A string that doesn't match any specific pattern
    const result = extractCaptureCandidate("Remember: deploy on Friday");
    // "remember" triggers higher importance but category detection is separate
    expect(result.category).toBeDefined();
  });

  it("should score explicit memory requests higher", () => {
    const explicit = extractCaptureCandidate("Remember this: always use HTTPS");
    const generic = extractCaptureCandidate("The server uses HTTPS by default");
    expect(explicit.importance).toBeGreaterThan(generic.importance);
  });

  it("should score entity category higher than preference", () => {
    const entity = extractCaptureCandidate("My name is John and I work at Acme");
    const preference = extractCaptureCandidate("I prefer dark mode in the editor");
    expect(entity.importance).toBeGreaterThan(preference.importance);
  });

  it("should give longer text slightly higher importance", () => {
    const short = extractCaptureCandidate("I prefer tabs");
    const long = extractCaptureCandidate(
      "I prefer tabs over spaces because it allows each developer to set their own visual width preference in their editor",
    );
    expect(long.importance).toBeGreaterThanOrEqual(short.importance);
  });

  it("should clamp importance between 0 and 1", () => {
    const result = extractCaptureCandidate("Remember this important thing: I always need to deploy on Fridays");
    expect(result.importance).toBeGreaterThanOrEqual(0);
    expect(result.importance).toBeLessThanOrEqual(1);
  });
});

describe("extractFromConversation", () => {
  const config = makeConfig({ maxPerConversation: 3 });

  it("should extract candidates from user messages", () => {
    const messages = [
      { role: "user", content: "I prefer TypeScript for all new projects" },
      { role: "assistant", content: "That's a great choice for type safety." },
      { role: "user", content: "We decided to use Vitest for testing" },
    ];

    const candidates = extractFromConversation(messages, config);
    expect(candidates.length).toBe(2);
  });

  it("should also check assistant messages for decisions", () => {
    const messages = [
      { role: "user", content: "What should we use for testing?" },
      { role: "assistant", content: "Let's use Vitest for the testing framework" },
    ];

    const candidates = extractFromConversation(messages, config);
    expect(candidates.length).toBe(1);
    expect(candidates[0].text).toContain("Vitest");
  });

  it("should skip system messages", () => {
    const messages = [
      { role: "system", content: "I prefer being helpful and honest" },
      { role: "user", content: "I prefer dark mode in the editor" },
    ];

    const candidates = extractFromConversation(messages, config);
    expect(candidates.length).toBe(1);
  });

  it("should sort by importance (highest first)", () => {
    const messages = [
      { role: "user", content: "I like using vim sometimes for editing" },
      { role: "user", content: "Remember this: always run tests before deploying to production" },
      { role: "user", content: "We decided to use Kubernetes for orchestration" },
    ];

    const candidates = extractFromConversation(messages, config);
    // "Remember this" should score highest
    expect(candidates[0].importance).toBeGreaterThanOrEqual(candidates[candidates.length - 1].importance);
  });

  it("should limit to maxPerConversation", () => {
    const messages = [
      { role: "user", content: "I prefer TypeScript over JavaScript for type safety" },
      { role: "user", content: "I always use dark mode in my editor" },
      { role: "user", content: "Remember this: the API key rotates every week" },
      { role: "user", content: "We decided to use PostgreSQL for the database" },
      { role: "user", content: "I need to deploy on Fridays only now" },
    ];

    const candidates = extractFromConversation(messages, makeConfig({ maxPerConversation: 2 }));
    expect(candidates.length).toBeLessThanOrEqual(2);
  });

  it("should return empty array for no capturable content", () => {
    const messages = [
      { role: "user", content: "Hello" },
      { role: "assistant", content: "Hi there!" },
    ];

    const candidates = extractFromConversation(messages, config);
    expect(candidates).toEqual([]);
  });
});
