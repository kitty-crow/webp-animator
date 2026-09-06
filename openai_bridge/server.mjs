import { createServer } from "node:http";
import { OpenAISchema, array, boolean, integer, number, object, shape, string } from "../vendor/openai-schema/dist/openaiSchema.js";

const HOST = process.env.OPENAI_BRIDGE_HOST || "127.0.0.1";
const PORT = Number(process.env.OPENAI_BRIDGE_PORT || 18744);
const BASE = (process.env.OPENAI_API_BASE || "https://api.openai.com/v1").replace(/\/+$/, "");
const API_KEY = process.env.OPENAI_API_KEY || "";

const MODEL_PRICES = {
  "gpt-5.6-luna": { input: 0.20, output: 1.20 },
  "gpt-5.6-terra": { input: 2.00, output: 12.00 },
  "gpt-5.6-sol": { input: 4.00, output: 20.00 },
  "gpt-5.6": { input: 4.00, output: 20.00 },
  "gpt-6-astra": { input: 10.00, output: 50.00 },
};

const IMAGE_TOKEN_PRICES = { input: 8.00, output: 30.00, textInput: 5.00 };
const IMAGE_OUTPUT_ESTIMATE = {
  low: { square: 0.006, portrait: 0.005, landscape: 0.005 },
  medium: { square: 0.053, portrait: 0.041, landscape: 0.041 },
  high: { square: 0.211, portrait: 0.165, landscape: 0.165 },
};

const reasoningEfforts = new Set(["none", "low", "medium", "high", "xhigh", "max"]);
const imageQualities = new Set(["low", "medium", "high"]);

function errorText(error) {
  if (error instanceof Error) return error.message;
  return String(error);
}

function clamp(value, minimum, maximum) {
  const numberValue = Number(value);
  if (!Number.isFinite(numberValue)) return minimum;
  return Math.max(minimum, Math.min(maximum, numberValue));
}

function dataUrl(base64, mime = "image/png") {
  return `data:${mime};base64,${base64}`;
}

function imagePart(base64, mime = "image/png") {
  return { type: "input_image", image_url: dataUrl(base64, mime) };
}

function modelBody(model, effort, maxOutputTokens = 2600) {
  const body = { model, store: false, max_output_tokens: maxOutputTokens };
  const requested = reasoningEfforts.has(effort) ? effort : "medium";
  if (model === "gpt-6-astra" && requested === "none") {
    body.reasoning = { effort: "low" };
  } else {
    body.reasoning = { effort: requested };
  }
  return body;
}

const motionSchema = object({
  region: string(),
  from_state: string(),
  to_state: string(),
  target_state: string(),
  confidence: integer(),
});

const planShape = shape("animation_interpolation_plan", object({
  valid_pair: boolean(),
  sequence_summary: string(),
  transition_summary: string(),
  refined_user_intent: string(),
  target_fraction: number(),
  invariants: array(string()),
  changing_regions: array(motionSchema),
  generation_instruction: string(),
  risks: array(string()),
  additional_frame_worthwhile: boolean(),
  benefit_score: integer(),
  recommended_next_fraction: number(),
  stop_reason: string(),
}));

const violationSchema = object({
  type: string(),
  region: string(),
  description: string(),
  severity: integer(),
});

const auditShape = shape("animation_interpolation_audit", object({
  acceptable: boolean(),
  overall_score: integer(),
  temporal_position_score: integer(),
  source_consistency_score: integer(),
  instruction_match_score: integer(),
  invariant_preservation_score: integer(),
  structural_coherence_score: integer(),
  unnecessary_change_score: integer(),
  observations: array(string()),
  violations: array(violationSchema),
  keep_from_attempt: array(string()),
  change_on_retry: array(string()),
  retry_recommended: boolean(),
  additional_frame_worthwhile: boolean(),
  benefit_score: integer(),
  corrective_instruction: string(),
}));

function labelledContext(context = []) {
  const parts = [];
  for (const item of context) {
    if (!item?.image_b64) continue;
    parts.push({ type: "input_text", text: `Sequence context frame ${item.label ?? item.index ?? "?"}.` });
    parts.push(imagePart(item.image_b64, item.mime || "image/png"));
  }
  return parts;
}

const PLANNER_BASE = `You are the planning component of a generic animation frame interpolation system.
You are given two temporal anchor frames and may also receive other frames from the same animation as context. Do not assume what the frames depict. Infer their contents visually.
Your task is to plan exactly one missing frame at the requested temporal fraction between the two anchors.
Preserve all visual information that should remain invariant. Change only what is necessary to represent the temporal transition. The user's instruction is intent guidance: inspect the images and refine vague or imperfect wording into a precise, visually grounded interpolation plan.
Consider geometry, position, pose/state, occlusion, deformation, lighting, texture, camera/framing, transparency, and any recurring or cyclic motion that is actually visible. Do not invent semantic content unsupported by the supplied frames.
If previous failed attempts and audit feedback are supplied, use them as negative evidence: preserve what was good, explicitly correct what was wrong, and do not repeat the same failure.
If the sequence is marked as a loop, treat the end and beginning as temporally adjacent and reason about smooth loop closure.
Return a conservative plan whose generation_instruction tells an image editor to create only the requested missing frame, preserving as much of the supplied visual state as possible.`;

const AUDITOR_BASE = `You are the visual auditor for a generic animation interpolation system.
Evaluate whether the candidate frame is a sensible temporal state between the two anchor frames at the requested fraction, using the whole supplied animation context when available.
Do not assume the images contain a person or character. Infer the subject and motion from the images.
Judge temporal placement, consistency with both anchors, adherence to the user's intent and interpolation plan, preservation of visual invariants, structural coherence, and whether unrelated regions changed unnecessarily.
Scores are rubric scores from 0 to 100, not probabilities. Be strict. A visually attractive image is not sufficient if it is the wrong temporal state or changes unrelated content.
Set acceptable=true only when the candidate is genuinely usable as an animation frame, not merely plausible in isolation. Any severe structural error, wrong temporal direction, materially changed invariant, or clear instruction mismatch must make the candidate unacceptable.
For a rejected attempt, identify what should be preserved from it and what must change on retry. If the sequence is a loop, also consider whether the candidate improves cyclic continuity.`;

function normalisePlan(value, requestedFraction) {
  value.target_fraction = clamp(requestedFraction, 0, 1);
  value.benefit_score = Math.round(clamp(value.benefit_score, 0, 100));
  value.recommended_next_fraction = clamp(value.recommended_next_fraction, 0, 1);
  if (Array.isArray(value.changing_regions)) {
    value.changing_regions = value.changing_regions.map(item => ({
      ...item,
      confidence: Math.round(clamp(item?.confidence, 0, 100)),
    }));
  }
  return value;
}

function normaliseAudit(value) {
  const scoreFields = [
    "overall_score",
    "temporal_position_score",
    "source_consistency_score",
    "instruction_match_score",
    "invariant_preservation_score",
    "structural_coherence_score",
    "unnecessary_change_score",
    "benefit_score",
  ];
  for (const field of scoreFields) value[field] = Math.round(clamp(value[field], 0, 100));
  if (Array.isArray(value.violations)) {
    value.violations = value.violations.map(item => ({
      ...item,
      severity: Math.round(clamp(item?.severity, 0, 100)),
    }));
  }

  const severeViolation = Array.isArray(value.violations)
    && value.violations.some(item => Number(item?.severity || 0) >= 75);
  const hardScoresPass = value.overall_score >= 80
    && value.temporal_position_score >= 72
    && value.source_consistency_score >= 78
    && value.invariant_preservation_score >= 78
    && value.structural_coherence_score >= 75;

  value.acceptable = Boolean(value.acceptable && hardScoresPass && !severeViolation);
  if (!value.acceptable) value.retry_recommended = true;
  return value;
}

async function runPlan(payload) {
  if (!API_KEY) throw new Error("OPENAI_API_KEY is not configured.");
  const model = String(payload.planner_model || "gpt-5.6-luna");
  const effort = String(payload.planner_effort || "medium");
  const client = new OpenAISchema(API_KEY, planShape, undefined, { conversation: false, base: BASE });

  const requestedFraction = Number(payload.target_fraction ?? 0.5);
  const content = [
    { type: "input_text", text: `${PLANNER_BASE}\n\nRequest metadata:\n${JSON.stringify({
      sequence_mode: payload.sequence_mode || "open",
      gap_type: payload.gap_type || "interior",
      target_fraction: requestedFraction,
      global_fraction: Number(payload.global_fraction ?? requestedFraction),
      user_instruction: String(payload.user_instruction || ""),
      previous_plan: payload.previous_plan || null,
      previous_audit: payload.previous_audit || null,
      user_feedback: String(payload.user_feedback || ""),
    }, null, 2)}` },
    { type: "input_text", text: "Earlier immediate anchor frame A:" },
    imagePart(payload.frame_a_b64, payload.frame_a_mime),
    { type: "input_text", text: "Later immediate anchor frame B:" },
    imagePart(payload.frame_b_b64, payload.frame_b_mime),
    ...labelledContext(payload.context_frames),
  ];

  if (payload.previous_attempt_b64) {
    content.push({ type: "input_text", text: "Previous rejected generation. Use it only as evidence of what worked and what failed:" });
    content.push(imagePart(payload.previous_attempt_b64, payload.previous_attempt_mime));
  }

  const value = await client.send([{ role: "user", content }], {
    body: modelBody(model, effort, Number(payload.max_output_tokens || 2600)),
    retries: 1,
    retryDelayMs: 250,
  });

  return { plan: normalisePlan(value, requestedFraction), usage: client.lastUsage || null, model };
}

async function runAudit(payload) {
  if (!API_KEY) throw new Error("OPENAI_API_KEY is not configured.");
  const model = String(payload.auditor_model || "gpt-5.6-luna");
  const effort = String(payload.auditor_effort || "medium");
  const client = new OpenAISchema(API_KEY, auditShape, undefined, { conversation: false, base: BASE });

  const content = [
    { type: "input_text", text: `${AUDITOR_BASE}\n\nAudit metadata:\n${JSON.stringify({
      sequence_mode: payload.sequence_mode || "open",
      gap_type: payload.gap_type || "interior",
      target_fraction: Number(payload.target_fraction ?? 0.5),
      global_fraction: Number(payload.global_fraction ?? payload.target_fraction ?? 0.5),
      user_instruction: String(payload.user_instruction || ""),
      interpolation_plan: payload.plan || null,
    }, null, 2)}` },
    { type: "input_text", text: "Earlier anchor A:" }, imagePart(payload.frame_a_b64, payload.frame_a_mime),
    { type: "input_text", text: "Candidate missing frame:" }, imagePart(payload.candidate_b64, payload.candidate_mime),
    { type: "input_text", text: "Later anchor B:" }, imagePart(payload.frame_b_b64, payload.frame_b_mime),
    ...labelledContext(payload.context_frames),
  ];

  const value = await client.send([{ role: "user", content }], {
    body: modelBody(model, effort, Number(payload.max_output_tokens || 2200)),
    retries: 1,
    retryDelayMs: 250,
  });

  return { audit: normaliseAudit(value), usage: client.lastUsage || null, model };
}

function b64ToBlob(value, mime = "image/png") {
  const bytes = Uint8Array.from(Buffer.from(value, "base64"));
  return new Blob([bytes], { type: mime });
}

function parseRequestedSize(size) {
  const match = /^(\d+)x(\d+)$/u.exec(String(size || ""));
  if (!match) return null;
  return { width: Number(match[1]), height: Number(match[2]) };
}

function pngDimensions(base64) {
  const buffer = Buffer.from(base64, "base64");
  const signature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  if (buffer.length < 24 || !buffer.subarray(0, 8).equals(signature)) return null;
  return { width: buffer.readUInt32BE(16), height: buffer.readUInt32BE(20) };
}

async function runGenerate(payload) {
  if (!API_KEY) throw new Error("OPENAI_API_KEY is not configured.");
  const model = String(payload.image_model || "gpt-image-2");
  const quality = imageQualities.has(payload.image_quality) ? payload.image_quality : "medium";
  const rawPrompt = String(payload.generation_instruction || "").trim();
  if (!rawPrompt) throw new Error("generation_instruction is required.");

  const refs = Array.isArray(payload.references) ? payload.references : [];
  if (refs.length < 2) throw new Error("At least two reference images are required.");

  const roleInstruction = refs.length >= 3
    ? "Reference image 1 is the earlier temporal anchor. Reference image 2 is the later temporal anchor. Reference image 3 is a previous rejected generation: use it only as negative/corrective evidence, preserve aspects explicitly identified as good, and do not copy its diagnosed failures."
    : "Reference image 1 is the earlier temporal anchor. Reference image 2 is the later temporal anchor. Treat both as hard temporal constraints, not loose style references.";
  const prompt = `${roleInstruction}\n\n${rawPrompt}`;
  const requested = parseRequestedSize(payload.size);
  if (!requested) throw new Error("An exact output size in WIDTHxHEIGHT form is required for animation interpolation.");

  const form = new FormData();
  form.append("model", model);
  form.append("prompt", prompt);
  form.append("quality", quality);
  form.append("output_format", "png");
  form.append("size", `${requested.width}x${requested.height}`);
  if (payload.transparent !== false) form.append("background", "transparent");
  refs.forEach((ref, index) => {
    form.append("image[]", b64ToBlob(ref.image_b64, ref.mime || "image/png"), `reference-${index + 1}.png`);
  });

  const response = await fetch(`${BASE}/images/edits`, {
    method: "POST",
    headers: { authorization: `Bearer ${API_KEY}` },
    body: form,
  });
  const raw = await response.text();
  if (!response.ok) throw new Error(`OpenAI image edit failed (${response.status}): ${raw.slice(0, 1200)}`);
  const value = JSON.parse(raw);
  const item = Array.isArray(value.data) ? value.data[0] : null;
  let imageB64 = item?.b64_json || item?.b64 || null;
  if (!imageB64 && item?.url) {
    const imageResponse = await fetch(item.url);
    if (!imageResponse.ok) throw new Error(`Could not download generated image (${imageResponse.status}).`);
    imageB64 = Buffer.from(await imageResponse.arrayBuffer()).toString("base64");
  }
  if (!imageB64) throw new Error("OpenAI returned no generated image data.");

  const dimensions = pngDimensions(imageB64);
  if (!dimensions) throw new Error("OpenAI image edit did not return a valid PNG image.");
  if (dimensions.width !== requested.width || dimensions.height !== requested.height) {
    throw new Error(
      `OpenAI returned ${dimensions.width}x${dimensions.height}, but animation interpolation requires exact ${requested.width}x${requested.height} output. The frame was rejected rather than stretched.`,
    );
  }

  return {
    image_b64: imageB64,
    mime: "image/png",
    width: dimensions.width,
    height: dimensions.height,
    usage: value.usage || null,
    model,
    quality,
  };
}

function usageCost(model, usage) {
  if (!usage) return 0;
  const price = MODEL_PRICES[model];
  if (!price) return 0;
  const input = Number(usage.inputTokens ?? usage.input_tokens ?? 0);
  const output = Number(usage.outputTokens ?? usage.output_tokens ?? 0);
  return (input * price.input + output * price.output) / 1_000_000;
}

function imageUsageCost(usage) {
  if (!usage || typeof usage !== "object") return 0;
  const input = Number(usage.input_tokens ?? 0);
  const output = Number(usage.output_tokens ?? 0);
  const details = usage.input_tokens_details || {};
  const imageInput = Number(details.image_tokens ?? input);
  const textInput = Number(details.text_tokens ?? 0);
  return (imageInput * IMAGE_TOKEN_PRICES.input + textInput * IMAGE_TOKEN_PRICES.textInput + output * IMAGE_TOKEN_PRICES.output) / 1_000_000;
}

function estimate(payload) {
  const plannerModel = String(payload.planner_model || "gpt-5.6-luna");
  const auditorModel = String(payload.auditor_model || "gpt-5.6-luna");
  const quality = imageQualities.has(payload.image_quality) ? payload.image_quality : "medium";
  const width = Math.max(1, Number(payload.width || 1024));
  const height = Math.max(1, Number(payload.height || 1024));
  const referenceCount = Math.max(2, Number(payload.reference_count || 2));
  const contextCount = Math.max(0, Number(payload.context_count || 0));
  const attempts = Math.max(1, Number(payload.attempts || 1));

  const megapixels = (width * height) / 1_000_000;
  const visionTokensPerImage = 900 + Math.ceil(megapixels * 900);
  const plannerInputTokens = 1100 + (2 + contextCount) * visionTokensPerImage;
  const auditorInputTokens = 1400 + (3 + contextCount) * visionTokensPerImage;
  const plannerOutputTokens = 900;
  const auditorOutputTokens = 700;

  const planner = MODEL_PRICES[plannerModel]
    ? (plannerInputTokens * MODEL_PRICES[plannerModel].input + plannerOutputTokens * MODEL_PRICES[plannerModel].output) / 1_000_000
    : 0;
  const auditor = MODEL_PRICES[auditorModel]
    ? (auditorInputTokens * MODEL_PRICES[auditorModel].input + auditorOutputTokens * MODEL_PRICES[auditorModel].output) / 1_000_000
    : 0;

  const orientation = width === height ? "square" : (height > width ? "portrait" : "landscape");
  const baseOutput = IMAGE_OUTPUT_ESTIMATE[quality][orientation];
  const referenceInputTokens = referenceCount * visionTokensPerImage;
  const imageInput = referenceInputTokens * IMAGE_TOKEN_PRICES.input / 1_000_000;
  const imageText = 700 * IMAGE_TOKEN_PRICES.textInput / 1_000_000;
  const generation = baseOutput + imageInput + imageText;
  const oneAttempt = planner + generation + auditor;

  return {
    currency: "USD",
    approximate: true,
    planner,
    generation,
    auditor,
    one_attempt: oneAttempt,
    attempts,
    maximum: oneAttempt * attempts,
    assumptions: {
      width, height, reference_count: referenceCount, context_count: contextCount,
      planner_input_tokens: plannerInputTokens,
      auditor_input_tokens: auditorInputTokens,
      image_reference_tokens: referenceInputTokens,
    },
  };
}

async function nodeBody(request) {
  const chunks = [];
  for await (const chunk of request) chunks.push(chunk);
  return Buffer.concat(chunks);
}

function sendNode(response, value, status = 200) {
  const body = Buffer.from(JSON.stringify(value));
  response.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": String(body.length),
    "cache-control": "no-store",
  });
  response.end(body);
}

const server = createServer(async (request, response) => {
  const url = new URL(request.url || "/", `http://${HOST}:${PORT}`);
  try {
    if (request.method === "GET" && url.pathname === "/health") {
      sendNode(response, {
        ok: true,
        key_configured: Boolean(API_KEY),
        openai_schema: true,
        bridge: "openai-interrogator",
        version: 2,
      });
      return;
    }
    if (request.method !== "POST") {
      sendNode(response, { error: "Not found." }, 404);
      return;
    }
    const type = request.headers["content-type"] || "";
    if (!String(type).includes("application/json")) throw new Error("Expected application/json.");
    const raw = await nodeBody(request);
    const payload = JSON.parse(raw.toString("utf-8") || "{}");
    let result;
    if (url.pathname === "/plan") result = await runPlan(payload);
    else if (url.pathname === "/generate") result = await runGenerate(payload);
    else if (url.pathname === "/audit") result = await runAudit(payload);
    else if (url.pathname === "/estimate") result = estimate(payload);
    else if (url.pathname === "/cost") {
      result = {
        planner: usageCost(String(payload.planner_model || "gpt-5.6-luna"), payload.planner_usage),
        generation: imageUsageCost(payload.image_usage),
        auditor: usageCost(String(payload.auditor_model || "gpt-5.6-luna"), payload.auditor_usage),
      };
    } else {
      sendNode(response, { error: "Not found." }, 404);
      return;
    }
    sendNode(response, result);
  } catch (error) {
    console.error(error);
    sendNode(response, { error: errorText(error) }, 500);
  }
});

server.listen(PORT, HOST, () => {
  console.log(`OpenAI interpolation bridge listening on http://${HOST}:${PORT}`);
});
