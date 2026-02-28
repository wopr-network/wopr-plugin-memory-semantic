import type { PendingEntry } from "./embedding-queue.js";

/**
 * Re-chunk text at multiple granularities for multi-scale vector indexing.
 * Each scale produces independent vectors: small chunks for precision, large for context.
 */
export function multiScaleChunk(
  text: string,
  baseId: string,
  meta: { path: string; startLine: number; endLine: number; source: string; instanceId?: string },
  scales: Array<{ tokens: number; overlap: number }>,
): PendingEntry[] {
  const results: PendingEntry[] = [];

  // Always emit a canonical entry under baseId (smallest scale) so that
  // hasEntry(baseId) works for dedup on restart/bootstrap.
  // Trade-off: when the text fits within the smallest scale, the scale loop below
  // also emits an identical entry under `${baseId}-s${smallest.tokens}`. This
  // intentional duplicate keeps the canonical ID stable for dedup while the
  // per-scale IDs let callers query at a specific granularity.
  const smallest = scales.reduce((a, b) => (a.tokens <= b.tokens ? a : b), scales[0]);
  if (smallest && text.trim().length >= 10) {
    const maxChars = smallest.tokens * 4;
    results.push({
      entry: {
        id: baseId,
        path: meta.path,
        startLine: meta.startLine,
        endLine: meta.endLine,
        source: meta.source,
        snippet: text.slice(0, 500),
        content: text.length <= maxChars ? text : text.slice(0, maxChars),
        instanceId: meta.instanceId,
      },
      text: text.length <= maxChars ? text : text.slice(0, maxChars),
    });
  }

  for (const scale of scales) {
    // Skip invalid scales to prevent infinite loops
    if (!Number.isFinite(scale.tokens) || scale.tokens <= 0) continue;
    const maxChars = scale.tokens * 4;
    let overlapChars = (Number.isFinite(scale.overlap) ? scale.overlap : 0) * 4;
    // Clamp overlap to prevent infinite loop if misconfigured
    if (overlapChars >= maxChars) overlapChars = Math.max(0, maxChars - 4);
    if (text.length <= maxChars) {
      // Text fits in one chunk at this scale
      results.push({
        entry: {
          id: `${baseId}-s${scale.tokens}`,
          path: meta.path,
          startLine: meta.startLine,
          endLine: meta.endLine,
          source: meta.source,
          snippet: text.slice(0, 500),
          content: text,
          instanceId: meta.instanceId,
        },
        text,
      });
    } else {
      // Split into sub-chunks at this scale
      let start = 0;
      let subIdx = 0;
      while (start < text.length) {
        const end = Math.min(start + maxChars, text.length);
        const chunk = text.slice(start, end);
        if (chunk.trim().length >= 10) {
          results.push({
            entry: {
              id: `${baseId}-s${scale.tokens}-${subIdx}`,
              path: meta.path,
              startLine: meta.startLine,
              endLine: meta.endLine,
              source: meta.source,
              snippet: chunk.slice(0, 500),
              content: chunk,
              instanceId: meta.instanceId,
            },
            text: chunk,
          });
        }
        if (end >= text.length) break; // Last chunk reached, stop
        start = end - overlapChars;
        subIdx++;
      }
    }
  }
  return results;
}
