// OpenCode Free's glm-5.3-flash-free is served by the zen inference backend, which
// rejects the z.ai `thinking` object the global *glm-5.3* pattern emits:
//   400 {"error":{"param":"thinking","type":"invalid_request_error",
//        "message":"Upstream request failed: [unknown_parameter] invalid request
//         body: json: unknown field \"thinking\""}}
// Reproduced live (2026-10-18, oc/glm-5.3-flash-free): a request carrying a
// thinking intent answered 400, the identical request without one answered 200.
// The provider-scoped capability override switches this id to the openai format
// (reasoning_effort) — the shape OpenCode Go's glm-5.3-flash already uses on the
// same backend. These tests drive the real applyThinking() so a regression in the
// capability entry shows up as the native object again.
import { describe, expect, it } from "vitest";
import { applyThinking, captureThinking } from "../../open-sse/translator/concerns/thinkingUnified.js";
import { getCapabilitiesForModel } from "../../open-sse/providers/capabilities.js";

// An OpenAI-wire request asking for high effort (what the agent client sends).
const openAIWireBody = () => ({
  model: "glm-5.3-flash-free",
  reasoning_effort: "high",
  messages: [{ role: "user", content: "hi" }],
});

const withIntent = () => captureThinking({ reasoning_effort: "high" });

describe("opencode-free glm-5.3-flash-free thinking shape", () => {
  it("capability override selects openai for the free provider", () => {
    expect(getCapabilitiesForModel("opencode", "glm-5.3-flash-free").thinkingFormat).toBe("openai");
    // The provider alias resolves to the id before capabilities are read, and the
    // override keeps the reasoning flag the catalog reports for this id.
    expect(getCapabilitiesForModel("oc", "glm-5.3-flash-free").thinkingFormat).toBe("zai");
    expect(getCapabilitiesForModel("opencode", "glm-5.3-flash-free").reasoning).toBe(true);
  });

  it("emits reasoning_effort and no native thinking object", () => {
    const body = applyThinking("openai", "glm-5.3-flash-free", openAIWireBody(), "opencode", withIntent());

    expect(body.thinking).toBeUndefined();
    expect(body.reasoning_effort).toBe("high");
  });

  it("keeps the z.ai shape where the backend is z.ai itself", () => {
    // Regression guard: the override must not leak past the free provider.
    const body = applyThinking("openai", "glm-5.3-flash-free", openAIWireBody(), "zai", withIntent());

    expect(body.thinking).toEqual({ type: "enabled" });
  });

  it("leaves the sibling ids on the shapes their backends accept", () => {
    // OpenCode Go carries its own override for the unsuffixed id.
    const go = applyThinking("openai", "glm-5.3-flash", openAIWireBody(), "opencode-go", withIntent());
    expect(go.thinking).toBeUndefined();
    expect(go.reasoning_effort).toBeTruthy();

    // The free tier's deepseek/mimo ids accept the native object live, so the
    // family pattern must keep sending it.
    const deepseek = applyThinking("openai", "deepseek-v4.1-flash-free", openAIWireBody(), "opencode", withIntent());
    expect(deepseek.thinking).toEqual({ type: "enabled" });
  });

  it("still strips thinking entirely for a model that cannot reason", () => {
    const body = applyThinking("openai", "union-alpha", { ...openAIWireBody(), reasoning_effort: "high" }, "opencode", withIntent());

    expect(body.thinking).toBeUndefined();
    expect(body.reasoning_effort).toBeUndefined();
  });
});
