/**
 * Generic OpenAI chat endpoint
 * POST /.netlify/functions/chat
 * Body: { messages: [ { role:"user"|"assistant"|"system", content:"…" }, … ] }
 */
import 'dotenv/config';
import { OpenAI } from 'openai';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

export async function handler(event /* , context */) {
  // ───────── guard HTTP method ──────────
  if (event.httpMethod !== 'POST') {
    return { statusCode: 405, body: 'Method Not Allowed' };
  }

  // ───────── parse body ──────────
  let body;
  try {
    body = JSON.parse(event.body ?? '{}');
  } catch {
    return { statusCode: 400, body: JSON.stringify({ error: 'Invalid JSON' }) };
  }

  const { messages } = body;
  if (!Array.isArray(messages) || messages.length === 0) {
    return {
      statusCode: 400,
      body: JSON.stringify({ error: 'messages array required' }),
    };
  }

  // ───────── OpenAI call ──────────
  try {
    const completion = await openai.chat.completions.create({
      model: 'gpt-4o',
      messages,
      max_tokens: 512,
      temperature: 0.7,
    });

    const reply =
      completion.choices?.[0]?.message?.content ?? '⚠️ no response';

    return {
      statusCode: 200,
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ reply }),
    };
  } catch (err) {
    console.error('OpenAI error →', err);
    return {
      statusCode: 500,
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ error: 'Error from OpenAI API' }),
    };
  }
}
