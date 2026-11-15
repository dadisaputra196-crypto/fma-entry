// wire2.ts — cross-wire conversion for the fma-entry aggregator.
//
// The trial's provider directory speaks three wires: OpenAI Chat Completions,
// Anthropic /v1/messages and Gemini generateContent. Clients arrive on the
// OpenAI or the Anthropic wire only. This module is the bridge: it turns one
// wire's request body into another wire's request body, and turns a non-OpenAI
// response back into the shape the client expects.
//
// Design notes:
//   - Conversions are total: every field the client could send is either mapped
//     or dropped deliberately, never with a thrown error. A silent drop (e.g. an
//     image sent to a text-only upstream) is better than a 500.
//   - Responses synthesised from another wire carry `created: 0`. A real epoch
//     timestamp would make two replays of the same trial produce different
//     bytes; the scorer hashes outputs, so determinism wins.
//   - Tool calls keep their ids as best each wire allows: OpenAI ids are
//     synthesised (`call_N`) when Gemini gives none, and a tool result's
//     tool_call_id doubles as the Gemini functionResponse name.
//   - All four exported pairs here are pure functions over plain JSON: no fetch,
//     no clocks, no state — easy to reason about and safe to retry on failover.
import { isRecord } from './providers.ts';

type Json = Record<string, unknown>;

/** Default max_tokens for the Anthropic wire, which requires the field. */
const ANTHROPIC_DEFAULT_MAX_TOKENS = 1024;

/**
 * OpenAI chat.completions request → Anthropic /v1/messages request.
 * System messages are folded into the top-level `system` string (Anthropic has
 * no system role); OpenAI tool definitions become Anthropic tool schemas.
 */
export function anthropicBodyFromOpenAI(body: Json, model: string): Json {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const outMessages: Json[] = [];
  let system = '';
  for (const m of messages) {
    if (!isRecord(m)) continue;
    const role = typeof m.role === 'string' ? m.role : 'user';
    if (role === 'system') {
      // Anthropic puts the system prompt outside the message list.
      system += (typeof m.content === 'string' ? m.content : JSON.stringify(m.content)) + '\n';
      continue;
    }
    outMessages.push({ role, content: m.content ?? '' });
  }
  const out: Json = { model, max_tokens: typeof body.max_tokens === 'number' ? body.max_tokens : ANTHROPIC_DEFAULT_MAX_TOKENS, messages: outMessages };
  if (system.trim() !== '') out.system = system.trim();
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (body.stream === true) out.stream = true;
  if (Array.isArray(body.tools) && body.tools.length > 0) {
    // OpenAI {type:'function', function:{name, description, parameters}} →
    // Anthropic {name, description, input_schema}.
    out.tools = body.tools.map((t: Json) => {
      const fn = isRecord(t.function) ? t.function : {};
      return {
        name: typeof fn.name === 'string' ? fn.name : 'tool',
        description: typeof fn.description === 'string' ? fn.description : '',
        input_schema: isRecord(fn.parameters) ? fn.parameters : { type: 'object' },
      };
    });
  }
  return out;
}

/**
 * Anthropic /v1/messages request → OpenAI chat.completions request.
 * The inverse of anthropicBodyFromOpenAI: the `system` string re-enters the
 * message list as a system role, and Anthropic tool schemas wrap back into
 * OpenAI function definitions.
 */
export function openAIBodyFromAnthropic(body: Json, model: string): Json {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const outMessages: Json[] = [];
  const system = typeof body.system === 'string' ? body.system : '';
  if (system !== '') outMessages.push({ role: 'system', content: system });
  for (const m of messages) {
    if (!isRecord(m)) continue;
    const role = typeof m.role === 'string' ? m.role : 'user';
    const content = typeof m.content === 'string' ? m.content : m.content;
    outMessages.push({ role, content });
  }
  const out: Json = { model, messages: outMessages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (body.stream === true) out.stream = true;
  if (Array.isArray(body.tools) && body.tools.length > 0) {
    out.tools = body.tools.map((t: Json) => ({
      type: 'function',
      function: { name: typeof t.name === 'string' ? t.name : 'tool', description: typeof t.description === 'string' ? t.description : '', parameters: isRecord(t.input_schema) ? t.input_schema : { type: 'object' } },
    }));
  }
  return out;
}

/**
 * OpenAI chat.completions request → Gemini generateContent request.
 * Returns the upstream body plus the modelId (Gemini carries the model in the
 * URL path, not the body). Mappings worth noting:
 *   - roles: assistant → model, tool results → user turns with a
 *     functionResponse part (Gemini has no tool role);
 *   - content blocks: text passes through, data-URL images become inlineData
 *     parts, remote images degrade to a text note;
 *   - assistant tool_calls become functionCall parts appended to that turn;
 *   - response_format json_object becomes responseMimeType application/json.
 */
export function geminiBodyFromOpenAI(body: Json, model: string): { body: Json; modelId: string } {
  const contents: Json[] = [];
  let systemText = '';
  let toolOutputCounter = 0;
  const toolResults: Json[] = []; // collected and appended after the dialogue turns
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const m of messages) {
    if (!isRecord(m)) continue;
    const role = typeof m.role === 'string' ? m.role : 'user';
    if (role === 'system') { systemText += (typeof m.content === 'string' ? m.content : '') + '\n'; continue; }
    if (role === 'tool') {
      // A tool result arrives as its own user turn carrying the functionResponse.
      toolResults.push({ role: 'user', parts: [{ functionResponse: { name: typeof m.tool_call_id === 'string' ? m.tool_call_id : `tool${toolOutputCounter++}`, response: { result: typeof m.content === 'string' ? m.content : JSON.stringify(m.content ?? '') } } }] });
      continue;
    }
    const gRole = role === 'assistant' ? 'model' : 'user';
    let parts: Json[];
    if (typeof m.content === 'string') parts = [{ text: m.content }];
    else if (Array.isArray(m.content)) {
      parts = [];
      for (const b of m.content) {
        if (!isRecord(b)) { parts.push({ text: JSON.stringify(b) }); continue; }
        const btype = typeof b.type === 'string' ? b.type : '';
        if (btype === 'text' && typeof b.text === 'string') parts.push({ text: b.text });
        else if (btype === 'image_url' && isRecord(b.image_url) && typeof b.image_url.url === 'string') {
          // Only inline data URLs can cross to Gemini; remote URLs become a marker.
          const url = b.image_url.url as string;
          const mt = /^data:([^;]+);base64,(.*)$/.exec(url);
          if (mt) parts.push({ inlineData: { mimeType: mt[1], data: mt[2] } });
          else parts.push({ text: `[image: ${url.slice(0, 80)}]` });
        } else parts.push({ text: JSON.stringify(b) });
      }
    } else parts = [{ text: JSON.stringify(m.content ?? '') }];
    // Assistant tool_calls → gemini functionCall parts on the same turn.
    if (role === 'assistant' && Array.isArray(m.tool_calls)) {
      for (const tc of m.tool_calls) {
        if (isRecord(tc) && isRecord(tc.function)) {
          let args: Json = {};
          if (typeof tc.function.arguments === 'string') { try { args = JSON.parse(tc.function.arguments); } catch { args = {}; } }
          else if (isRecord(tc.function.arguments)) args = tc.function.arguments;
          parts.push({ functionCall: { name: tc.function.name, args } });
        }
      }
    }
    contents.push({ role: gRole, parts });
  }
  contents.push(...toolResults);
  const out: Json = { contents };
  if (systemText.trim() !== '') out.systemInstruction = { parts: [{ text: systemText.trim() }] };
  const gen: Json = {};
  if (typeof body.max_tokens === 'number') gen.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') gen.temperature = body.temperature;
  if (Array.isArray(body.stop) && body.stop.length > 0) gen.stopSequences = body.stop;
  else if (typeof body.stop === 'string' && body.stop !== '') gen.stopSequences = [body.stop];
  if (Object.keys(gen).length > 0) out.generationConfig = gen;
  if (isRecord(body.response_format) && body.response_format.type === 'json_object') {
    out.generationConfig = { ...(out.generationConfig as Json), responseMimeType: 'application/json' };
  }
  if (Array.isArray(body.tools) && body.tools.length > 0) {
    const decls: Json[] = [];
    for (const t of body.tools) {
      if (isRecord(t) && isRecord(t.function)) {
        decls.push({ name: t.function.name, description: typeof t.function.description === 'string' ? t.function.description : '', parameters: isRecord(t.function.parameters) ? t.function.parameters : { type: 'object' } });
      }
    }
    if (decls.length > 0) out.tools = [{ functionDeclarations: decls }];
  }
  return { body: out, modelId: model };
}

/** Gemini generateContent JSON → OpenAI chat.completion JSON. */
export function openAIFromGeminiResponse(g: Json, model: string): Json {
  const candidates = Array.isArray(g.candidates) ? g.candidates : [];
  const first = candidates.length > 0 && isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  const toolCalls: Json[] = [];
  parts.forEach((p: Json) => {
    if (!isRecord(p)) return;
    if (typeof p.text === 'string') text += p.text;
    if (isRecord(p.functionCall)) {
      const fc = p.functionCall;
      // Gemini names tools but never issues ids: synthesise stable ones.
      toolCalls.push({ id: `call_${toolCalls.length}`, type: 'function', function: { name: typeof fc.name === 'string' ? fc.name : 'tool', arguments: JSON.stringify(isRecord(fc.args) ? fc.args : {}) } });
    }
  });
  const message: Json = { role: 'assistant', content: text };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;
  const finish = typeof first.finishReason === 'string' ? first.finishReason : 'STOP';
  const usage = isRecord(g.usageMetadata) ? g.usageMetadata : {};
  return {
    id: `gemini-${model}`,
    object: 'chat.completion',
    created: 0, // fixed epoch: byte-identical across replays
    model,
    choices: [{ index: 0, message, finish_reason: finish === 'STOP' ? 'stop' : finish.toLowerCase() }],
    usage: {
      prompt_tokens: typeof usage.promptTokenCount === 'number' ? usage.promptTokenCount : 0,
      completion_tokens: typeof usage.candidatesTokenCount === 'number' ? usage.candidatesTokenCount : 0,
      total_tokens: typeof usage.totalTokenCount === 'number' ? usage.totalTokenCount : 0,
    },
  };
}

/**
 * Anthropic /v1/messages response → OpenAI chat.completion response.
 * Content blocks are concatenated in order (a text answer is the common case;
 * tool-use blocks carry no text and so contribute nothing here). `created: 0`
 * keeps synthesized responses deterministic across trial replays.
 */
export function anthropicToOpenAIAnswer(a: Json, model: string): Json {
  const content = Array.isArray(a.content) ? a.content : [];
  const text = content.map((b: Json) => (typeof b.text === 'string' ? b.text : '')).join('');
  const stop = typeof a.stop_reason === 'string' ? a.stop_reason : 'end_turn';
  const usage = isRecord(a.usage) ? a.usage : {};
  return {
    id: typeof a.id === 'string' ? a.id : `anthropic-${model}`,
    object: 'chat.completion',
    // Fixed epoch: keeps the synthesized response byte-identical across replays (determinism).
    created: 0,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: stop === 'end_turn' ? 'stop' : stop }],
    usage: {
      prompt_tokens: typeof usage.input_tokens === 'number' ? usage.input_tokens : 0,
      completion_tokens: typeof usage.output_tokens === 'number' ? usage.output_tokens : 0,
      total_tokens: (typeof usage.input_tokens === 'number' ? usage.input_tokens : 0) + (typeof usage.output_tokens === 'number' ? usage.output_tokens : 0),
    },
  };
}
