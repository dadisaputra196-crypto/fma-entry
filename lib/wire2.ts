
// wire2.ts: cross-wire conversion helpers for the fma-entry aggregator.
// openAI <-> anthropic <-> gemini request/response conversion (JSON, non-stream first; streaming passthrough when wires match).
import { isRecord } from './providers.ts';

type Json = Record<string, unknown>;

export function anthropicBodyFromOpenAI(body: Json, model: string): Json {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const outMessages: Json[] = [];
  let system = '';
  for (const m of messages) {
    if (!isRecord(m)) continue;
    const role = typeof m.role === 'string' ? m.role : 'user';
    if (role === 'system') {
      system += (typeof m.content === 'string' ? m.content : JSON.stringify(m.content)) + '\n';
      continue;
    }
    outMessages.push({ role, content: m.content ?? '' });
  }
  const out: Json = { model, max_tokens: typeof body.max_tokens === 'number' ? body.max_tokens : 1024, messages: outMessages };
  if (system.trim() !== '') out.system = system.trim();
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (body.stream === true) out.stream = true;
  if (Array.isArray(body.tools) && body.tools.length > 0) {
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

export function geminiBodyFromOpenAI(body: Json, model: string): { body: Json; modelId: string } {
  const contents: Json[] = [];
  let systemText = '';
  let toolOutputCounter = 0;
  const toolResults: Json[] = []; // collect tool results to append after contents
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const m of messages) {
    if (!isRecord(m)) continue;
    const role = typeof m.role === 'string' ? m.role : 'user';
    if (role === 'system') { systemText += (typeof m.content === 'string' ? m.content : '') + '\n'; continue; }
    if (role === 'tool') {
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
          const url = b.image_url.url as string;
          const mt = /^data:([^;]+);base64,(.*)$/.exec(url);
          if (mt) parts.push({ inlineData: { mimeType: mt[1], data: mt[2] } });
          else parts.push({ text: `[image: ${url.slice(0, 80)}]` });
        } else parts.push({ text: JSON.stringify(b) });
      }
    } else parts = [{ text: JSON.stringify(m.content ?? '') }];
    // assistant tool_calls → gemini functionCall parts
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
    created: 0,
    model,
    choices: [{ index: 0, message, finish_reason: finish === 'STOP' ? 'stop' : finish.toLowerCase() }],
    usage: {
      prompt_tokens: typeof usage.promptTokenCount === 'number' ? usage.promptTokenCount : 0,
      completion_tokens: typeof usage.candidatesTokenCount === 'number' ? usage.candidatesTokenCount : 0,
      total_tokens: typeof usage.totalTokenCount === 'number' ? usage.totalTokenCount : 0,
    },
  };
}
/** Anthropic /v1/messages response → OpenAI chat.completion response. */
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
