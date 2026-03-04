import { vi } from "vitest";
import type { PluginLogger, StorageApi, WOPREventBus } from "@wopr-network/plugin-types";

export function mockLogger(): PluginLogger {
  return {
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
    debug: vi.fn(),
  } as unknown as PluginLogger;
}

export function mockStorage(): StorageApi {
  return {
    raw: vi.fn().mockResolvedValue([]),
    transaction: vi.fn().mockImplementation(async (fn: () => Promise<void>) => fn()),
  } as unknown as StorageApi;
}

export function mockEvents(): WOPREventBus {
  const handlers = new Map<string, Array<(event: unknown) => unknown>>();
  return {
    on: vi.fn().mockImplementation((event: string, handler: (e: unknown) => unknown) => {
      if (!handlers.has(event)) handlers.set(event, []);
      handlers.get(event)!.push(handler);
      return () => {
        const arr = handlers.get(event);
        if (arr) {
          const idx = arr.indexOf(handler);
          if (idx !== -1) arr.splice(idx, 1);
        }
      };
    }),
    emit: vi.fn().mockResolvedValue(undefined),
    off: vi.fn(),
  } as unknown as WOPREventBus;
}
