import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { once } from "node:events";

const mockPort = 18746;
const bridgePort = 18747;
const requests = [];

function fakePng(width, height) {
  const buffer = Buffer.alloc(24);
  Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]).copy(buffer, 0);
  buffer.writeUInt32BE(width, 16);
  buffer.writeUInt32BE(height, 20);
  return buffer.toString("base64");
}

const mock = createServer(async (request, response) => {
  const chunks = [];
  for await (const chunk of request) chunks.push(chunk);
  const body = Buffer.concat(chunks).toString("latin1");
  requests.push({ url: request.url, body });

  if (request.url !== "/v1/images/edits") {
    response.writeHead(404, { "content-type": "application/json" });
    response.end(JSON.stringify({ error: { message: "not found" } }));
    return;
  }

  if (requests.length === 1) {
    response.writeHead(400, { "content-type": "application/json", "x-request-id": "req_mock_block" });
    response.end(JSON.stringify({
      error: {
        message: "mock output moderation block",
        type: "image_generation_user_error",
        code: "moderation_blocked",
        moderation_details: { moderation_stage: "output", categories: ["sexual"] },
      },
    }));
    return;
  }

  response.writeHead(200, { "content-type": "application/json" });
  response.end(JSON.stringify({
    data: [{ b64_json: fakePng(1024, 1024) }],
    usage: { input_tokens: 20, output_tokens: 30, input_tokens_details: { image_tokens: 18, text_tokens: 2 } },
  }));
});

mock.listen(mockPort, "127.0.0.1");
await once(mock, "listening");

const child = spawn(process.execPath, ["openai_bridge/server.mjs"], {
  cwd: process.cwd(),
  env: {
    ...process.env,
    OPENAI_API_KEY: "test-key",
    OPENAI_API_BASE: `http://127.0.0.1:${mockPort}/v1`,
    OPENAI_BRIDGE_HOST: "127.0.0.1",
    OPENAI_BRIDGE_PORT: String(bridgePort),
  },
  stdio: ["ignore", "pipe", "pipe"],
});

let childLog = "";
child.stdout.on("data", chunk => { childLog += chunk.toString(); });
child.stderr.on("data", chunk => { childLog += chunk.toString(); });

async function waitForBridge() {
  for (let attempt = 0; attempt < 60; attempt += 1) {
    try {
      const response = await fetch(`http://127.0.0.1:${bridgePort}/health`);
      if (response.ok) return await response.json();
    } catch {}
    await new Promise(resolve => setTimeout(resolve, 50));
  }
  throw new Error(`bridge did not start: ${childLog}`);
}

try {
  const health = await waitForBridge();
  assert.equal(health.ok, true);
  assert.ok(health.version >= 4);

  const dummy = Buffer.from("reference-image-bytes").toString("base64");
  const response = await fetch(`http://127.0.0.1:${bridgePort}/generate`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      image_model: "gpt-image-2",
      image_quality: "medium",
      generation_instruction: "The requested position is 0.500000 of the way between the two anchors. Preserve everything unrelated.",
      size: "1024x1024",
      transparent: true,
      references: [
        { image_b64: dummy, mime: "image/png" },
        { image_b64: dummy, mime: "image/png" },
        { image_b64: dummy, mime: "image/png" },
      ],
    }),
  });

  const result = await response.json();
  assert.equal(response.status, 200, JSON.stringify(result));
  assert.equal(result.moderation_recovery, true);
  assert.equal(result.width, 1024);
  assert.equal(result.height, 1024);
  assert.equal(requests.length, 2);
  assert.equal(requests[0].url, "/v1/images/edits");
  assert.equal(requests[1].url, "/v1/images/edits");
  assert.match(requests[0].body, /reference-3\.png/);
  assert.doesNotMatch(requests[1].body, /reference-3\.png/);
  assert.match(requests[1].body, /conservative temporal in-between frame/);
  assert.match(requests[1].body, /Preserve the same subject\/object identity/);
} finally {
  child.kill();
  mock.close();
  await Promise.allSettled([once(child, "exit"), once(mock, "close")]);
}
