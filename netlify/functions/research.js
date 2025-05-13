/**
 * Topic-aware "Research Assistant"
 * Supports OpenAI, Gemini, or Groq depending on AI_PROVIDER env var.
 *
 * POST /.netlify/functions/research
 * Body: { messages:[…], topic?:string }
 *
 * ENV:
 *   AI_PROVIDER          openai | gemini | groq   (default groq)
 *   OPENAI_API_KEY       required if provider=openai
 *   GEMINI_API_KEY       required if provider=gemini
 *   GROQ_API_KEY         required if provider=groq
 */

import 'dotenv/config';
import { Groq } from 'groq-sdk';
import { OpenAI } from 'openai';

// ─────────────────────────────────────────────────────────────
const AI_PROVIDER = (process.env.AI_PROVIDER ?? 'groq').toLowerCase();
// ─────────────────────────────────────────────────────────────

/* -----------------------------------------------------------
 * Common helpers
 * --------------------------------------------------------- */
function badRequest(msg) {
  return {
    statusCode: 400,
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ error: msg }),
  };
}

function serverError(msg, details) {
  if (details) console.error(details);
  return {
    statusCode: 500,
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ error: msg }),
  };
}

/* ─────────────────────────────────────────────── */
/*  GROQ branch                                    */
/* ─────────────────────────────────────────────── */
async function groqHandler(messages, topic) {
  const apiKey = process.env.GROQ_API_KEY;
  if (!apiKey) return serverError('GROQ_API_KEY missing');

  const groq = new Groq({ apiKey });

  const systemPrompt = topic
    ? `You are a knowledgeable research assistant focusing on "${topic}". Provide accurate, concise information about misophonia research.`
    : 'You are a knowledgeable research assistant on misophonia research.';

  const gMessages = [
    { role: 'system', content: systemPrompt },
    ...messages,
  ];

  const completion = await groq.chat.completions.create({
    model                 : "qwen-qwq-32b", //'llama-3.3-70b-versatile',
    messages              : gMessages,
    max__tokens : 4096,
    temperature           : 0,
    //stream                : true,
  });

  return {
    statusCode: 200,
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      reply      : completion.choices?.[0]?.message?.content ?? '',
      structured : null,
      provider   : 'groq',
    }),
  };
}

/* ===========================================================
 *  OpenAI branch
 * ========================================================= */
async function openaiHandler(messages, topic) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) return serverError('OPENAI_API_KEY missing');

  const openai = new OpenAI({ apiKey });

  const systemPrompt = topic
    ? `You are a knowledgeable research assistant focusing on "${topic}". Provide accurate, concise information about misophonia research.`
    : 'You are a knowledgeable research assistant on misophonia research.';

  const oaMessages = [
    { role: 'system', content: systemPrompt },
    ...messages,
  ];

  const completion = await openai.chat.completions.create({
    model: 'gpt-4.1-mini-2025-04-14',
    messages: oaMessages,
    max_tokens: 1024,
    temperature: 0,
  });

  return {
    statusCode: 200,
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      reply: completion.choices?.[0]?.message?.content ?? '',
      structured: null,
      provider: 'openai',
    }),
  };
}

/* ===========================================================
 *  Gemini branch
 * ========================================================= */
async function geminiHandler(messages, topic) {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) return serverError('GEMINI_API_KEY missing');

  // Build user prompt
  let userPrompt = '';
  if (topic) userPrompt += `Topic: ${topic}\n`;
  userPrompt += messages
    .map(
      (m) => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content}`
    )
    .join('\n');

  const resp = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro-preview-03-25:generateContent?key=${apiKey}`,
    {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        contents: [{ role: 'user', parts: [{ text: userPrompt }] }],
        generationConfig: { temperature: 0.7, maxOutputTokens: 1024 },
        tools: [
          {
            function_declarations: [
              {
                name: 'structured_output',
                description:
                  'Return information in a structured JSON format for chat display, with sections, bullet points, and highlights.',
              },
            ],
          },
        ],
      }),
    }
  );

  if (!resp.ok) {
    const errText = await resp.text().catch(() => '');
    return serverError('Error from Gemini API', errText);
  }

  const data = await resp.json();

  let structured = null;
  try {
    const part = data.candidates?.[0]?.content?.parts?.[0];
    if (part?.functionCall?.name === 'structured_output') {
      structured = JSON.parse(part.functionCall.args.json || '{}');
    }
  } catch {
    // swallow JSON parse errors – keep structured=null
  }

  const reply =
    data.candidates?.[0]?.content?.parts?.[0]?.text ?? '';

  return {
    statusCode: 200,
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ reply, structured, provider: 'gemini' }),
  };
}

/* ===========================================================
 *  Netlify Function entry-point
 * ========================================================= */
export async function handler(event /* , context */) {
  if (event.httpMethod !== 'POST') {
    return { statusCode: 405, body: 'Method Not Allowed' };
  }

  // ─── parse body ──────────────────────────────────────────
  let body;
  try {
    body = JSON.parse(event.body ?? '{}');
  } catch {
    return badRequest('Invalid JSON');
  }

  const { messages, topic } = body;
  if (!Array.isArray(messages) || messages.length === 0) {
    return badRequest('messages array required');
  }

  try {
    if (AI_PROVIDER === 'gemini') {
      return await geminiHandler(messages, topic);
    }
    if (AI_PROVIDER === 'openai') {
      return await openaiHandler(messages, topic);
    }
    return await groqHandler(messages, topic);
  } catch (err) {
    return serverError(
      `Unexpected ${AI_PROVIDER} handler failure`,
      err
    );
  }
}
