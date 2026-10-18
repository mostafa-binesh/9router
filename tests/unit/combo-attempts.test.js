import { describe, it, expect, beforeEach, vi } from "vitest";

import { handleComboChat, handleFusionChat, reorderByCapabilities, resetComboRotation, canRepeatComboMember } from "../../open-sse/services/combo.js";
import {
  MAX_COMBO_ATTEMPTS,
  MAX_COMBO_TOTAL_ATTEMPTS,
  formatComboEntry,
  mergeComboEntryText,
  parseComboEntry,
  parseComboModels,
  stripComboAttempts,
} from "../../open-sse/services/comboEntry.js";
import { aggregateComboCapabilities } from "../../open-sse/providers/capabilities.js";
import { getStickyComboModel, setStickyComboModel } from "../../open-sse/services/systemoneRouter.js";

const log = { info() {}, warn() {}, debug() {}, error() {} };

function jsonResponse(status, body = {}, headers = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...headers },
  });
}

const okResponse = () => jsonResponse(200, {
  choices: [{ message: { role: "assistant", content: "done" }, finish_reason: "stop" }],
});

// Zero-usable-content 2xx: finish_reason in_progress with an empty message.
const emptyResponse = () => jsonResponse(200, {
  choices: [{ message: { role: "assistant", content: "" }, finish_reason: "in_progress" }],
});

const busy = () => jsonResponse(503, { error: { message: "Service Unavailable" } });

/** Handler that answers from a scripted queue, recording the model of each call. */
function scriptedHandler(script) {
  const calls = [];
  const handler = async (body, model) => {
    calls.push(model);
    const next = script.shift();
    if (!next) throw new Error(`unscripted call for ${model}`);
    return next(model, calls.length);
  };
  return { handler, calls };
}

describe("comboEntry grammar", () => {
  it("plain id -> one attempt", () => {
    expect(parseComboEntry("deepseek/deepseek-v4-flash")).toEqual({ model: "deepseek/deepseek-v4-flash", attempts: 1 });
  });

  it("trailing #N sets the attempt count", () => {
    expect(parseComboEntry("deepseek/deepseek-v4-flash#3")).toEqual({ model: "deepseek/deepseek-v4-flash", attempts: 3 });
  });

  it("clamps #N to the maximum", () => {
    expect(parseComboEntry("a/b#999").attempts).toBe(MAX_COMBO_ATTEMPTS);
  });

  it("keeps a non-positive or non-numeric suffix as part of the model id", () => {
    expect(parseComboEntry("a/b#0")).toEqual({ model: "a/b#0", attempts: 1 });
    expect(parseComboEntry("a/b#x")).toEqual({ model: "a/b#x", attempts: 1 });
    expect(parseComboEntry("a/b#")).toEqual({ model: "a/b#", attempts: 1 });
  });

  it("accepts the descriptor form unchanged", () => {
    expect(parseComboEntry({ model: "a/b", attempts: 4 })).toEqual({ model: "a/b", attempts: 4 });
  });

  it("normalizes a member list and drops empties", () => {
    expect(parseComboModels(["a/b#2", "", "  ", "c/d"])).toEqual([
      { model: "a/b", attempts: 2 },
      { model: "c/d", attempts: 1 },
    ]);
  });

  it("round-trips through formatComboEntry", () => {
    expect(formatComboEntry("a/b", 3)).toBe("a/b#3");
    expect(formatComboEntry("a/b", 1)).toBe("a/b");
  });

  it("strips attempts for the id-only passes", () => {
    expect(stripComboAttempts(["a/b#2", "c/d"])).toEqual(["a/b", "c/d"]);
  });

  it("keeps the attempt count when only the id is edited by hand", () => {
    expect(mergeComboEntryText("a/other", "a/b#3")).toBe("a/other#3");
    expect(mergeComboEntryText("a/other#5", "a/b#3")).toBe("a/other#5");
    expect(mergeComboEntryText("a/other#5", "a/b")).toBe("a/other#5");
  });
});

describe("handleComboChat — attempts per member", () => {
  beforeEach(() => {
    resetComboRotation();
  });

  it("defaults to a single attempt per member (unchanged behavior)", async () => {
    const { handler, calls } = scriptedHandler([busy, okResponse]);
    const result = await handleComboChat({
      body: {}, models: ["p/a", "p/b"], handleSingleModel: handler, log,
    });
    expect(calls).toEqual(["p/a", "p/b"]);
    expect(result.ok).toBe(true);
  });

  it("retries a member up to its #N before giving up on it", async () => {
    const { handler, calls } = scriptedHandler([busy, busy, okResponse]);
    const result = await handleComboChat({
      body: {}, models: ["p/a#3"], handleSingleModel: handler, log,
    });
    expect(calls).toEqual(["p/a", "p/a", "p/a"]);
    expect(result.ok).toBe(true);
  });

  it("moves to the next member once the attempts are spent", async () => {
    const { handler, calls } = scriptedHandler([busy, busy, okResponse]);
    const result = await handleComboChat({
      body: {}, models: ["p/a#2", "p/b"], handleSingleModel: handler, log,
    });
    expect(calls).toEqual(["p/a", "p/a", "p/b"]);
    expect(result.ok).toBe(true);
  });

  it("spends an attempt on a 2xx with zero usable content", async () => {
    const { handler, calls } = scriptedHandler([emptyResponse, okResponse]);
    const result = await handleComboChat({
      body: {}, models: ["p/a#2", "p/b"], handleSingleModel: handler, log,
    });
    expect(calls).toEqual(["p/a", "p/a"]);
    expect(result.ok).toBe(true);
  });

  it("waits out a short model lock instead of skipping the member", async () => {
    const locked = () => jsonResponse(503, { error: { message: "Service Unavailable" } }, { "retry-after": "1" });
    const { handler, calls } = scriptedHandler([locked, okResponse]);
    const startedAt = Date.now();
    const result = await handleComboChat({
      body: {}, models: ["p/a#2", "p/b"], handleSingleModel: handler, log,
    });
    expect(calls).toEqual(["p/a", "p/a"]);
    expect(result.ok).toBe(true);
    expect(Date.now() - startedAt).toBeGreaterThanOrEqual(900);
  });

  // Regression: the upstream's own long `retry-after` (a free tier asking for
  // minutes) used to cancel the member's remaining attempts outright.
  it("keeps the attempts when the reported wait is longer than the window", async () => {
    const limited = () => jsonResponse(429, { error: { message: "Rate limit exceeded" } }, { "retry-after": "600" });
    const { handler, calls } = scriptedHandler([limited, limited, limited]);
    const startedAt = Date.now();
    const result = await handleComboChat({
      body: {}, models: ["oc/deepseek#3"], handleSingleModel: handler, log,
    });
    expect(calls).toEqual(["oc/deepseek", "oc/deepseek", "oc/deepseek"]);
    expect(result.ok).toBe(false);
    expect(result.status).toBe(429);
    // Repeats are paced, not skipped — and not slept out for the 600s asked for.
    expect(Date.now() - startedAt).toBeLessThan(3000);
  });

  it("does not repeat a credential error", async () => {
    const unauthorized = () => jsonResponse(401, { error: { message: "Invalid API key provided" } });
    const { handler, calls } = scriptedHandler([unauthorized, okResponse]);
    const result = await handleComboChat({
      body: {}, models: ["p/a#3", "p/b"], handleSingleModel: handler, log,
    });
    expect(calls).toEqual(["p/a", "p/b"]);
    expect(result.ok).toBe(true);
  });

  // Regression: a rate-limited free member used to be tried exactly once because
  // 429 was treated as a credential failure.
  it("retries a rate-limited member before handing the turn on", async () => {
    const rateLimited = () => jsonResponse(429, { error: { message: "Rate limit exceeded" } });
    const { handler, calls } = scriptedHandler([rateLimited, rateLimited, okResponse]);
    const startedAt = Date.now();
    const result = await handleComboChat({
      body: {}, models: ["oc/deepseek#3", "p/b"], handleSingleModel: handler, log,
    });
    expect(calls).toEqual(["oc/deepseek", "oc/deepseek", "oc/deepseek"]);
    expect(result.ok).toBe(true);
    // Paced off the provider's answer, far quicker than a cooldown wait.
    expect(Date.now() - startedAt).toBeLessThan(1500);
  });

  it("spends every configured attempt on a member that keeps rate limiting", async () => {
    const rateLimited = () => jsonResponse(429, { error: { message: "Rate limit exceeded" } });
    const { handler, calls } = scriptedHandler(Array.from({ length: 10 }, () => rateLimited).concat([okResponse]));
    const result = await handleComboChat({
      body: {}, models: ["oc/deepseek#10", "p/b"], handleSingleModel: handler, log,
    });
    expect(calls).toEqual(Array.from({ length: 10 }, () => "oc/deepseek").concat(["p/b"]));
    expect(result.ok).toBe(true);
  }, 20_000);

  it("retries a member whose call threw", async () => {
    const { handler, calls } = scriptedHandler([
      () => { throw new Error("socket hang up"); },
      okResponse,
    ]);
    const result = await handleComboChat({
      body: {}, models: ["p/a#2", "p/b"], handleSingleModel: handler, log,
    });
    expect(calls).toEqual(["p/a", "p/a"]);
    expect(result.ok).toBe(true);
  });

  it("stops at the total attempt budget", async () => {
    const { handler, calls } = scriptedHandler(Array.from({ length: 60 }, () => busy));
    const result = await handleComboChat({
      body: {}, models: ["p/a#10", "p/b#10", "p/c#10", "p/d#10"], handleSingleModel: handler, log,
    });
    expect(calls).toHaveLength(MAX_COMBO_TOTAL_ATTEMPTS);
    expect(result.ok).toBe(false);
    expect(result.status).toBe(503);
  }, 30_000);

  it("keeps round-robin order and the member's own attempts", async () => {
    const { handler, calls } = scriptedHandler([busy, busy, okResponse]);
    const result = await handleComboChat({
      body: {},
      models: ["p/a#2", "p/b#2"],
      handleSingleModel: handler,
      log,
      comboName: "rr-combo",
      comboStrategy: "round-robin",
    });
    expect(calls).toEqual(["p/a", "p/a", "p/b"]);
    expect(result.ok).toBe(true);
  });

  it("reports the model lock's reset time when every member fails", async () => {
    const { handler } = scriptedHandler([() => jsonResponse(503, { error: { message: "Unavailable" } }, { "retry-after": "3600" })]);
    const result = await handleComboChat({
      body: {}, models: ["p/a"], handleSingleModel: handler, log,
    });
    const reported = Number(result.headers.get("retry-after"));
    expect(reported).toBeGreaterThanOrEqual(3590);
    expect(reported).toBeLessThanOrEqual(3600);
  });
});

describe("canRepeatComboMember", () => {
  const base = { remaining: 2, status: 503, pauseMs: 250, waitedMs: 0 };

  it("repeats a transient failure that leaves attempts", () => {
    expect(canRepeatComboMember(base)).toBe(true);
  });

  it("does not repeat without attempts left", () => {
    expect(canRepeatComboMember({ ...base, remaining: 0 })).toBe(false);
  });

  it("does not repeat a credential or missing-model status", () => {
    for (const status of [401, 402, 403, 404]) {
      expect(canRepeatComboMember({ ...base, status })).toBe(false);
    }
  });

  // Regression: a free tier answers 429 constantly, and the first version of the
  // attempt feature listed 429 as non-retryable, so a `#10` member was tried once
  // and the turn went to the next model.
  it("repeats a rate limit that leaves attempts", () => {
    expect(canRepeatComboMember({ ...base, status: 429 })).toBe(true);
  });

  it("does not skip a long wait, the caller paces it instead", () => {
    // `repeatPauseMs` caps a long reported wait to the pace; the predicate only
    // sees the pause it will actually make.
    expect(canRepeatComboMember({ ...base, pauseMs: 250 })).toBe(true);
    expect(canRepeatComboMember({ ...base, pauseMs: 1_000 })).toBe(true);
  });

  it("does not exceed the per-request wait budget", () => {
    expect(canRepeatComboMember({ ...base, pauseMs: 1_000, waitedMs: 14_000 })).toBe(true);
    expect(canRepeatComboMember({ ...base, pauseMs: 1_000, waitedMs: 14_500 })).toBe(false);
  });

  // Regression: OpenCode's free tier answers a burst with 403 FreeTierError
  // ("the free tier can only be used from within OpenCode"). It is a per-session
  // throttle that clears within seconds — live, a member 403s and then succeeds
  // on a repeat. Reading it as a credential error cut every `#N` member down to a
  // single attempt and sent the turn straight to a non-gated model.
  it("repeats a 403 from OpenCode's free-tier burst gate", () => {
    const message = "OpenCode's free tier can only be used from within OpenCode. Download it at https://opencode.ai and keep building.";
    // The combo parses error.message; the whole body is accepted too.
    expect(canRepeatComboMember({ ...base, status: 403, errorText: message })).toBe(true);
    expect(canRepeatComboMember({ ...base, status: 403, errorText: JSON.stringify({ type: "error", error: { type: "FreeTierError", message } }) })).toBe(true);
  });

  it("still refuses a 403 that is a real credential problem", () => {
    expect(canRepeatComboMember({ ...base, status: 403, errorText: "invalid api key" })).toBe(false);
    expect(canRepeatComboMember({ ...base, status: 403 })).toBe(false);
    // The gate wording only exempts a 403; a 401 keeps ending the turn.
    expect(canRepeatComboMember({ ...base, status: 401, errorText: "OpenCode's free tier can only be used from within OpenCode." })).toBe(false);
  });
});

describe("free-tier gate inside a combo turn", () => {
  const gate = () => jsonResponse(403, {
    type: "error",
    error: { type: "FreeTierError", message: "OpenCode's free tier can only be used from within OpenCode." },
  });

  it("spends the member's remaining attempts before falling through", async () => {
    const { handler, calls } = scriptedHandler([gate, gate, okResponse]);
    const res = await handleComboChat({
      body: { messages: [{ role: "user", content: "hi" }] },
      models: ["opencode/deepseek-v4.1-flash-free#4", "opencode/big-pickle"],
      handleSingleModel: handler,
      log,
    });

    expect(res.status).toBe(200);
    // Three calls to the gated member, not one: the two 403s were repeated, and
    // its success means the turn never reached the fallback member.
    expect(calls).toEqual([
      "opencode/deepseek-v4.1-flash-free",
      "opencode/deepseek-v4.1-flash-free",
      "opencode/deepseek-v4.1-flash-free",
    ]);
  });

  it("hands the turn to the next member when a non-gate 403 persists", async () => {
    const { handler, calls } = scriptedHandler([
      () => jsonResponse(403, { error: { message: "invalid api key" } }),
      okResponse,
    ]);
    const res = await handleComboChat({
      body: { messages: [{ role: "user", content: "hi" }] },
      models: ["opencode/deepseek-v4.1-flash-free#4", "opencode/big-pickle"],
      handleSingleModel: handler,
      log,
    });

    expect(res.status).toBe(200);
    expect(calls).toEqual(["opencode/deepseek-v4.1-flash-free", "opencode/big-pickle"]);
  });
});

describe("auto-switch and capability lookups with attempt suffixes", () => {
  beforeEach(() => {
    resetComboRotation();
  });

  it("floats a vision-capable member to the front without changing its attempts", () => {
    const plan = parseComboModels(["deepseek/deepseek-chat#3", "anthropic/claude-sonnet-4.6"]);
    const out = reorderByCapabilities(plan, new Set(["vision"]), (m) => m.model);
    expect(out[0].model).toBe("anthropic/claude-sonnet-4.6");
    expect(out[0].attempts).toBe(1);
    expect(out[1]).toEqual({ model: "deepseek/deepseek-chat", attempts: 3 });
  });

  it("tries the capable member first when the request carries an image", async () => {
    const { handler, calls } = scriptedHandler([okResponse]);
    await handleComboChat({
      body: { messages: [{ role: "user", content: [{ type: "image_url", image_url: { url: "x" } }] }] },
      models: ["deepseek/deepseek-chat#3", "anthropic/claude-sonnet-4.6"],
      handleSingleModel: handler,
      log,
    });
    expect(calls).toEqual(["anthropic/claude-sonnet-4.6"]);
  });

  it("aggregates combo capabilities from the model ids", () => {
    const caps = aggregateComboCapabilities(["opencode-go/mimo-v2.5#3"]);
    expect(caps.vision).toBe(true);
    expect(caps.contextWindow).toBe(1048576);
  });
});

describe("fusion panels ignore attempt suffixes", () => {
  it("answers directly through the single panel member", async () => {
    const handler = vi.fn(async () => okResponse());
    await handleFusionChat({
      body: {}, models: ["p/a#3"], handleSingleModel: handler, log, comboName: "solo",
    });
    expect(handler).toHaveBeenCalledTimes(1);
    expect(handler.mock.calls[0][1]).toBe("p/a");
  });
});

// Integration coverage for the upstream smart/empty-response behavior combined
// with fork attempts. Keep separate from the pacing cases changed by later picks.
describe("combo attempts with smart routing and empty SSE fallback", () => {
  beforeEach(() => {
    resetComboRotation();
  });

  const sseText = (finishReason = "stop") =>
    `data: ${JSON.stringify({ choices: [{ delta: { role: "assistant" }, finish_reason: finishReason }] })}\n\ndata: [DONE]\n\n`;
  const emptySse = (finishReason = "stop") => new Response(sseText(finishReason), {
    headers: { "Content-Type": "text/event-stream", "X-Upstream": "preserved" },
  });

  it.each(["smart", "strict", "per-turn"])("keeps bare IDs and duplicate members' own attempts under %s ordering", async (smartPolicy) => {
    const models = ["p/a#2", "p/b#2", { model: "p/c", attempts: 2 }, { model: "p/b", attempts: 3 }, "p/a#3"];
    const body = { messages: [{ role: "user", content: "hi" }] };
    const resolveSmartChoice = vi.fn(async () => ({
      choice: "p/b", confidence: 0.99, probabilities: { "p/c": 0.7, "p/a": 0.1 },
    }));
    if (smartPolicy === "strict") setStickyComboModel("smart-attempts", "session", "p/b");
    const { handler, calls } = scriptedHandler([
      ...Array.from({ length: 11 }, () => emptyResponse), okResponse,
    ]);
    const result = await handleComboChat({
      body, models, handleSingleModel: handler, log,
      comboName: "smart-attempts", comboStrategy: "smart", smartPolicy, sessionId: "session",
      resolveSmartChoice, autoSwitch: false,
    });
    if (smartPolicy === "strict") {
      expect(resolveSmartChoice).not.toHaveBeenCalled();
      expect(calls).toEqual(["p/b", "p/b", "p/b", "p/b", "p/b", "p/a", "p/a", "p/c", "p/c", "p/a", "p/a", "p/a"]);
    } else {
      expect(resolveSmartChoice).toHaveBeenCalledTimes(1);
      expect(resolveSmartChoice).toHaveBeenCalledWith(body, ["p/a", "p/b", "p/c", "p/b", "p/a"]);
      expect(calls).toEqual(["p/b", "p/b", "p/b", "p/b", "p/b", "p/c", "p/c", "p/a", "p/a", "p/a", "p/a", "p/a"]);
    }
    expect(result.ok).toBe(true);
  });

  it("keeps duplicate attempts when capability reordering overrides a smart choice", async () => {
    const textModel = "deepseek/deepseek-chat";
    const visionModel = "anthropic/claude-sonnet-4.6";
    const { handler, calls } = scriptedHandler([...Array.from({ length: 8 }, () => emptyResponse), okResponse]);
    const result = await handleComboChat({
      body: { messages: [{ role: "user", content: [{ type: "image_url", image_url: { url: "x" } }] }] },
      models: [`${textModel}#3`, `${visionModel}#2`, { model: visionModel, attempts: 4 }],
      handleSingleModel: handler, log, comboStrategy: "smart",
      resolveSmartChoice: async () => ({ choice: textModel, confidence: 1 }),
    });
    expect(calls).toEqual([...Array(6).fill(visionModel), ...Array(3).fill(textModel)]);
    expect(result.ok).toBe(true);
  });

  it("rotates duplicate member slots without detaching their attempts", async () => {
    const options = { body: {}, models: ["p/a#2", "p/b#3", "p/a#4"], log, comboName: "duplicate-rr", comboStrategy: "round-robin" };
    await handleComboChat({ ...options, handleSingleModel: async () => okResponse() });
    const { handler, calls } = scriptedHandler([...Array.from({ length: 8 }, () => emptyResponse), okResponse]);
    const result = await handleComboChat({ ...options, handleSingleModel: handler });
    expect(calls).toEqual(["p/b", "p/b", "p/b", "p/a", "p/a", "p/a", "p/a", "p/a", "p/a"]);
    expect(result.ok).toBe(true);
  });

  it.each([400, 413, 422])("advances past all remaining attempts on a request-scoped %s", async (status) => {
    const { handler, calls } = scriptedHandler([
      () => jsonResponse(status, { error: { message: "Unsupported parameter: fixture" } }, { "retry-after": "3600" }),
      okResponse,
    ]);
    const result = await handleComboChat({ body: {}, models: ["p/a#3", "p/b#2"], handleSingleModel: handler, log });
    expect(calls).toEqual(["p/a", "p/b"]);
    expect(result.ok).toBe(true);
  });

  it("returns the original request-scoped response after later members fail", async () => {
    const payload = { error: { message: "Unsupported parameter: fixture" } };
    const rejected = jsonResponse(422, payload, { "X-Upstream": "first" });
    const { handler, calls } = scriptedHandler([
      () => rejected,
      () => jsonResponse(400, { error: { message: "Invalid request body" } }),
    ]);
    const result = await handleComboChat({ body: {}, models: ["p/a#3", "p/b#2"], handleSingleModel: handler, log });
    expect(calls).toEqual(["p/a", "p/b"]);
    expect(result).toBe(rejected);
    expect(result.status).toBe(422);
    expect(result.headers.get("X-Upstream")).toBe("first");
    expect(await result.json()).toEqual(payload);
  });

  it("repeats empty SSE within the member before trying the next one", async () => {
    const { handler, calls } = scriptedHandler([() => emptySse(), () => emptySse(), okResponse]);
    const result = await handleComboChat({ body: {}, models: ["p/a#2", "p/b"], handleSingleModel: handler, log });
    expect(calls).toEqual(["p/a", "p/a", "p/b"]);
    expect((await result.json()).choices[0].message.content).toBe("done");
  });

  it("repeats the last member's empty SSE and returns its true final stream intact", async () => {
    const { handler, calls } = scriptedHandler([() => emptySse(), () => emptySse(), () => emptySse()]);
    const result = await handleComboChat({ body: {}, models: ["p/a#3"], handleSingleModel: handler, log });
    expect(calls).toEqual(["p/a", "p/a", "p/a"]);
    expect(result.status).toBe(200);
    expect(result.headers.get("X-Upstream")).toBe("preserved");
    expect(await result.text()).toBe(sseText());
  });

  it("returns the final empty SSE when the total budget ends before the plan does", async () => {
    const { handler, calls } = scriptedHandler(Array.from({ length: MAX_COMBO_TOTAL_ATTEMPTS }, () => () => emptySse()));
    const result = await handleComboChat({
      body: {}, models: ["p/a#10", "p/b#10", "p/c#10", "p/d#10", "p/e"], handleSingleModel: handler, log,
    });
    expect(calls).toHaveLength(MAX_COMBO_TOTAL_ATTEMPTS);
    expect(calls.slice(-2)).toEqual(["p/d", "p/d"]);
    expect(result.status).toBe(200);
    expect(await result.text()).toBe(sseText());
  });

  it.each(["sse", "json"])("never repeats or falls through a content-filtered %s response", async (format) => {
    const blocked = format === "sse" ? emptySse("content_filter") : jsonResponse(200, {
      choices: [{ message: { role: "assistant", content: "" }, finish_reason: "content_filter" }],
    });
    const { handler, calls } = scriptedHandler([() => blocked, okResponse]);
    const result = await handleComboChat({ body: {}, models: ["p/a#3", "p/b"], handleSingleModel: handler, log });
    expect(calls).toEqual(["p/a"]);
    expect(result.status).toBe(200);
    if (format === "sse") expect(await result.text()).toBe(sseText("content_filter"));
    else expect(result).toBe(blocked);
  });

  it.each(["empty-sse", "request-scoped", "exception"])("invalidates a sticky model on %s failure", async (failure) => {
    setStickyComboModel("failed-sticky", "session", "p/a");
    const { handler, calls } = scriptedHandler([
      () => {
        if (failure === "exception") throw new Error("socket hang up");
        if (failure === "request-scoped") return jsonResponse(422, { error: { message: "Unsupported parameter: fixture" } });
        return emptySse();
      },
      okResponse,
    ]);
    const result = await handleComboChat({
      body: {}, models: ["p/a#2", "p/b"], handleSingleModel: handler, log,
      comboName: "failed-sticky", comboStrategy: "smart", smartPolicy: "strict", sessionId: "session",
    });
    expect(calls).toEqual(failure === "request-scoped" ? ["p/a", "p/b"] : ["p/a", "p/a"]);
    expect(getStickyComboModel("failed-sticky", "session", ["p/a", "p/b"])).toBeNull();
    expect(result.ok).toBe(true);
  });
});
