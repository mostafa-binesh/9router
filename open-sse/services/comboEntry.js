/**
 * Combo entry grammar — one combo member: `provider/model`, optionally followed
 * by `#N` to try that member up to N times before the combo moves on.
 *
 *   deepseek/deepseek-v4-flash#3   → 3 attempts, then the next combo member
 *
 * The suffix is how a combo expresses "retry this model before falling back"
 * without storing the same model id three times.
 *
 * Pure module: no env, no DB, no engine imports, so the dashboard can use the
 * exact same parsing the router uses. The retry *policy* (how long a locked
 * model is waited on) lives in `services/combo.js`.
 */

/** Upper bound for one entry's `#N`, so a typo cannot fan out upstream calls. */
export const MAX_COMBO_ATTEMPTS = 10;

/** Upper bound for one request's total model attempts across a whole combo. */
export const MAX_COMBO_TOTAL_ATTEMPTS = 32;

// Only a trailing positive integer is a suffix: `model#0`, `model#x` and `model#`
// stay part of the model id, so a literal `#` in a model name is never eaten.
const ATTEMPT_SUFFIX = /#(\d+)$/;

/**
 * Split a combo member into its model id and attempt count.
 * @param {string|{model: string, attempts?: number}} entry
 * @returns {{ model: string, attempts: number }}
 */
export function parseComboEntry(entry) {
  // Descriptor form (the engine re-uses this on a plan it already expanded).
  if (entry && typeof entry === "object") {
    const model = typeof entry.model === "string" ? entry.model.trim() : "";
    return { model, attempts: clampAttempts(entry.attempts) };
  }

  const raw = typeof entry === "string" ? entry.trim() : "";
  const match = raw.match(ATTEMPT_SUFFIX);
  if (!match) return { model: raw, attempts: 1 };

  const attempts = clampAttempts(Number.parseInt(match[1], 10));
  const model = raw.slice(0, match.index).trim();
  // `#0` is not a repeat count; treat the whole string as the model id.
  if (!model || attempts <= 1) return { model: raw, attempts: 1 };
  return { model, attempts };
}

/**
 * Normalize a combo member list. Accepts a mix of plain ids and descriptors and
 * drops empty entries.
 * @param {Array} models
 * @returns {Array<{model: string, attempts: number}>}
 */
export function parseComboModels(models) {
  if (!Array.isArray(models)) return [];
  return models.map(parseComboEntry).filter((entry) => entry.model);
}

/**
 * Model ids only, in order — drops attempt suffixes and empty entries, for the
 * passes that reason about ids alone (capability lookups, access, adapters).
 */
export function stripComboAttempts(models) {
  return parseComboModels(models).map((entry) => entry.model);
}

/** Rebuild the stored form of a member from its model id and attempt count. */
export function formatComboEntry(model, attempts) {
  const clean = typeof model === "string" ? model.trim() : "";
  const count = clampAttempts(attempts);
  return count > 1 ? `${clean}#${count}` : clean;
}

/**
 * Re-apply the stored form after the member text was edited by hand: an edited
 * id keeps the attempt count it already had, unless the new text states one.
 */
export function mergeComboEntryText(text, previous) {
  const next = parseComboEntry(text);
  const typed = typeof text === "string" && ATTEMPT_SUFFIX.test(text.trim());
  return formatComboEntry(next.model, typed ? next.attempts : parseComboEntry(previous).attempts);
}

function clampAttempts(value) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed) || parsed < 1) return 1;
  return Math.min(parsed, MAX_COMBO_ATTEMPTS);
}
