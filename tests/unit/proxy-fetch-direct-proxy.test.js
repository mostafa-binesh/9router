import { describe, expect, it, vi } from "vitest";
import { proxyAwareFetch } from "../../open-sse/utils/proxyFetch.js";

// opencode's free tier fingerprints the TLS ClientHello, so a configured proxy
// has to travel inside the adapter's own transport. Routing it through undici's
// ProxyAgent instead answers 429 FreeUsageLimitError (measured 2026-11-10).
describe("proxyAwareFetch with an adapter transport", () => {
  it("hands the resolved proxy to the adapter transport", async () => {
    const directFetch = vi.fn(async () => new Response("ok"));
    const proxyUrl = "http://token:token@relay.internal:10003";

    const response = await proxyAwareFetch(
      "https://opencode.ai/inference/v1/chat/completions",
      { method: "POST", body: "{}", directFetch },
      { connectionProxyEnabled: true, connectionProxyUrl: proxyUrl }
    );

    expect(await response.text()).toBe("ok");
    expect(directFetch).toHaveBeenCalledTimes(1);
    expect(directFetch.mock.calls[0][1].proxyUrl).toBe(proxyUrl);
  });

  it("leaves the adapter transport untouched when no proxy resolves", async () => {
    const directFetch = vi.fn(async () => new Response("ok"));

    await proxyAwareFetch("https://opencode.ai/inference/v1/chat/completions", { directFetch }, null);

    expect(directFetch).toHaveBeenCalledTimes(1);
    expect(directFetch.mock.calls[0][1].proxyUrl).toBeUndefined();
  });
});
