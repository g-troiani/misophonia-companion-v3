/**
 * Generic OpenAI chat endpoint
 * POST /.netlify/functions/chat
 * Body: { messages: [ { role:"user"|"assistant"|"system", content:"…" }, … ] }
 */
import 'dotenv/config';
import { Groq } from 'groq-sdk';

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

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

  // ───────── Groq call ──────────
  try {
    const completion = await groq.chat.completions.create({
      model: 'llama-3.3-70b-versatile',
      messages,
      max_completion_tokens: 512,
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
    console.error('Groq error →', err);
    return {
      statusCode: 500,
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ error: 'Error from Groq API' }),
    };
  }
}
