// netlify/functions/rag.js
import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import OpenAI from 'openai';

// ─── clients ──────────────────────────────────────────
const sb = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ─── tiny cosine helper ───────────────────────────────
function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na  += a[i] * a[i];
    nb  += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-9);
}

// ─── Netlify Lambda entry-point ───────────────────────
export async function handler(event) {
  if (event.httpMethod !== 'POST') {
    return { statusCode: 405, body: 'Method Not Allowed' };
  }

  let body;
  try { body = JSON.parse(event.body ?? '{}'); }
  catch { return { statusCode: 400, body: 'Bad JSON' }; }

  const query = (body.query || '').trim();
  const limit = body.limit || 8;
  if (!query) {
    return { statusCode: 400, body: 'Query required' };
  }

  try {
    /* 1. Embed the query */
    const emb = await openai.embeddings.create({
      model: 'text-embedding-ada-002',
      input: query,
    });
    const qVec = emb.data[0].embedding;

    /* 2. RPC match (over-fetch 4×, then rerank) */
    const { data: rows } = await sb.rpc('match_research_chunks', {
      query_embedding: qVec,
      match_threshold: 0,
      match_count: limit * 4,
    });

    if (!rows?.length) {
      return {
        statusCode: 200,
        body: JSON.stringify({ answer: 'No relevant documents found.', results: [] })
      };
    }

    const ranked = rows
      .map(r => ({ ...r, similarity: cosine(qVec, r.embedding) }))
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, limit);

    /* 3. Build prompt */
    const ctx = ranked.map((r, i) =>
      `[${i + 1}] ${r.text.replace(/\s+/g, ' ').slice(0, 800)}`
    ).join('\n');

    const prompt = `
Answer strictly from the sources below; cite like [1], [2] …

QUESTION: "${query}"

SOURCES:
${ctx}`.trim();

    /* 4. GPT-4o */
    const chat = await openai.chat.completions.create({
      model: 'gpt-4o',
      messages: [
        { role: 'system', content: 'You are a misophonia research assistant.' },
        { role: 'user', content: prompt }
      ],
      max_tokens: 700,
      temperature: 0.2,
    });

    ranked.forEach(r => delete r.embedding);   // strip big vectors

    return {
      statusCode: 200,
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ answer: chat.choices[0].message.content, results: ranked })
    };
  } catch (err) {
    console.error(err);
    return { statusCode: 500, body: JSON.stringify({ error: String(err) }) };
  }
}
