// File: netlify/functions/rag.js
// -----------------------------------------------------------------------------
//  Netlify Function  ·  Retrieval-Augmented Generation for Misophonia Companion
// -----------------------------------------------------------------------------
//  •  Embeds the query with OpenAI ada-002 (1536-D)
//  •  Vector search in Supabase via pgvector RPC
//  •  Client-side cosine re-rank (-100…+100  %)
//  •  Bibliography-only chunks are discarded
//  •  Prompt budget guard (≤ 24 000 chars)
//  •  Llama-3.3-70B via Groq generates Markdown answer + Bibliography section
//  •  GET  /stats   → return total number of indexed chunks
//  •  POST /search  → return { answer, citations, results }
// -----------------------------------------------------------------------------
//  Environment variables required at *function* runtime:
//
//      SUPABASE_URL                 (e.g. https://xyz.supabase.co)
//      SUPABASE_SERVICE_ROLE_KEY    (or anon key if RLS permits the RPC)
//      OPENAI_API_KEY               (to embed + generate)
//      GROQ_API_KEY                 (to generate responses)
// -----------------------------------------------------------------------------
//  This file is a line-for-line port of rag_web_app_v9.py, rewritten in Node so
//  it can run natively as a Netlify Function (no external Flask server needed).
// -----------------------------------------------------------------------------
import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import OpenAI from 'openai';
import { Groq } from 'groq-sdk';

// ──────────────────────────── configuration ─────────────────────────────── //
const {
  SUPABASE_URL,
  SUPABASE_SERVICE_ROLE_KEY,
  OPENAI_API_KEY,
  GROQ_API_KEY,
} = process.env;

if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY || !OPENAI_API_KEY || !GROQ_API_KEY) {
  throw new Error(
    '❌  Missing env vars: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, OPENAI_API_KEY, GROQ_API_KEY'
  );
}

const sb     = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);
const openai = new OpenAI({ apiKey: OPENAI_API_KEY });
const groq   = new Groq({ apiKey: GROQ_API_KEY });

// ────────────────────────── Enhanced caching with TTL ───────────────────────── //
class LRUCacheWithTTL {
  constructor(maxSize = 100, ttl = 3600000) { // 1 hour TTL by default
    this.maxSize = maxSize;
    this.ttl = ttl;
    this.cache = new Map();
    this.accessOrder = [];
    this.stats = { hits: 0, misses: 0, evictions: 0 };
  }

  get(key) {
    const item = this.cache.get(key);
    if (!item) {
      this.stats.misses++;
      return null;
    }

    // Check if item has expired
    if (Date.now() - item.timestamp > this.ttl) {
      this.cache.delete(key);
      this.removeFromAccessOrder(key);
      this.stats.misses++;
      return null;
    }

    // Move to end (most recently used)
    this.removeFromAccessOrder(key);
    this.accessOrder.push(key);
    this.stats.hits++;
    
    console.log(`Cache hit for key: ${key.substring(0, 50)}...`);
    console.log(`Cache stats - Hits: ${this.stats.hits}, Misses: ${this.stats.misses}, Hit rate: ${(this.stats.hits / (this.stats.hits + this.stats.misses) * 100).toFixed(1)}%`);
    
    return item.value;
  }

  set(key, value) {
    // Remove if already exists
    if (this.cache.has(key)) {
      this.removeFromAccessOrder(key);
    }

    // Evict LRU items if at capacity
    while (this.cache.size >= this.maxSize) {
      const lruKey = this.accessOrder.shift();
      this.cache.delete(lruKey);
      this.stats.evictions++;
    }

    // Add new item
    this.cache.set(key, {
      value: value,
      timestamp: Date.now()
    });
    this.accessOrder.push(key);
    
    console.log(`Cached new embedding for key: ${key.substring(0, 50)}...`);
  }

  removeFromAccessOrder(key) {
    const index = this.accessOrder.indexOf(key);
    if (index > -1) {
      this.accessOrder.splice(index, 1);
    }
  }

  // Clean up expired items
  cleanup() {
    const now = Date.now();
    for (const [key, item] of this.cache.entries()) {
      if (now - item.timestamp > this.ttl) {
        this.cache.delete(key);
        this.removeFromAccessOrder(key);
      }
    }
  }

  getStats() {
    return {
      ...this.stats,
      size: this.cache.size,
      hitRate: this.stats.hits + this.stats.misses > 0 
        ? (this.stats.hits / (this.stats.hits + this.stats.misses) * 100).toFixed(1) + '%'
        : '0%'
    };
  }
}

// Enhanced cache with TTL and LRU eviction
const embCache = new LRUCacheWithTTL(100, 3600000); // 100 items, 1 hour TTL

// Rate limiting for embedding calls
let lastEmbedCall = 0;
const EMBED_RATE_LIMIT_MS = 100; // Minimum 100ms between embed calls

// Periodic cleanup of expired cache entries (every 10 minutes)
setInterval(() => {
  embCache.cleanup();
  console.log('Cache cleanup completed. Current stats:', embCache.getStats());
}, 600000);

// ────────────────────────── embedding & similarity ───────────────────────── //
/** Return ada-002 embedding for *text* (1536-D float array) with enhanced caching */
async function embed(text) {
  // Normalize text for cache key consistency
  const normalizedText = text.trim().toLowerCase().slice(0, 8192);
  const cacheKey = normalizedText;
  
  // Check cache first
  const cached = embCache.get(cacheKey);
  if (cached) {
    return cached;
  }

  // Rate limiting
  const now = Date.now();
  const timeSinceLastCall = now - lastEmbedCall;
  if (timeSinceLastCall < EMBED_RATE_LIMIT_MS) {
    await new Promise(resolve => setTimeout(resolve, EMBED_RATE_LIMIT_MS - timeSinceLastCall));
  }

  try {
    console.log(`Making OpenAI embedding call for text: ${text.substring(0, 100)}...`);
    
    const { data } = await openai.embeddings.create({
      model: 'text-embedding-ada-002',
      input: text.slice(0, 8192), // hard limit: ada-002 max input tokens
    });

    const vec = data[0].embedding;
    lastEmbedCall = Date.now();
    
    // Cache the result
    embCache.set(cacheKey, vec);
    
    console.log('Embedding generated and cached successfully');
    console.log('Current cache stats:', embCache.getStats());
    
    return vec;
  } catch (error) {
    console.error('OpenAI embedding failed:', error);
    
    // If rate limited, wait and retry once
    if (error.status === 429) {
      console.log('Rate limited, waiting 2 seconds and retrying...');
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      const { data } = await openai.embeddings.create({
        model: 'text-embedding-ada-002',
        input: text.slice(0, 8192),
      });
      
      const vec = data[0].embedding;
      lastEmbedCall = Date.now();
      embCache.set(cacheKey, vec);
      return vec;
    }
    
    throw error;
  }
}

/** Plain cosine similarity (no scaling) */
const cosine = (a, b) => {
  let dot = 0,
    na = 0,
    nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na  += a[i] * a[i];
    nb  += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-9); // avoid /0
};

// ─────────────────────── bibliography chunk detector ─────────────────────── //
const DOI_RE  = /\b10\.\d{4,9}\/[-._;()/:A-Z0-9]+\b/gi;
const YEAR_RE = /\b(19|20)\d{2}\b/g;

/** True if chunk looks like a pure references list (skip). */
const looksLikeRefs = (txt) =>
  (txt.match(DOI_RE)?.length || 0) > 12 ||
  (txt.match(YEAR_RE)?.length || 0) > 15;

// ────────────────────────────── vector search ────────────────────────────── //
/**
 * 1. Embed query
 * 2. pgvector RPC (over-fetch 4× so we can re-rank)
 * 3. Drop bibliography noise
 * 4. Pull true embeddings + doc metadata
 * 5. Client-side cosine re-rank (-100…+100 %)
 * 6. Return top *limit* rows
 */
async function semanticSearch(query, limit = 8, threshold = 0) {
  console.log(`Starting semantic search for query: "${query}" (limit: ${limit})`);
  
  const qVec = await embed(query);

  const { data: rows = [] } = await sb.rpc('match_research_chunks', {
    query_embedding: qVec,
    match_threshold: threshold,  // 0 → ANN returns highest-IP matches
    match_count: limit * 4,      // over-fetch (will re-rank)
  });

  console.log(`Retrieved ${rows.length} initial matches from vector search`);

  const filtered = rows.filter(r => !looksLikeRefs(r.text));
  console.log(`After filtering bibliography chunks: ${filtered.length} matches remain`);
  
  if (!filtered.length) return [];

  // Bulk-fetch embeddings + document rows in 2 round-trips
  const chunkIds = filtered.map(r => r.id);
  const docIds   = [...new Set(filtered.map(r => r.document_id))];

  const [
    { data: embRows = [] },
    { data: docs    = [] },
  ] = await Promise.all([
    sb.from('research_chunks')
      .select('id,embedding')
      .in('id', chunkIds)
      .limit(chunkIds.length),
    sb.from('research_documents')
      .select(
        'id,title,authors,year,journal,doi,abstract,keywords,research_topics,source_pdf'
      )
      .in('id', docIds)
      .limit(docIds.length),
  ]);

  const embMap = new Map(
    embRows.map(e => [
      e.id,
      Array.isArray(e.embedding)
        ? e.embedding.map(Number)                       // numeric column
        : e.embedding.slice(1, -1).split(',').map(Number), // text column "[…]"
    ])
  );
  const docMap = new Map(docs.map(d => [d.id, d]));

  // Compute **true cosine** and attach metadata
  filtered.forEach(r => {
    const vec = embMap.get(r.id);
    r.similarity = vec ? Math.round(cosine(qVec, vec) * 1000) / 10 : 0; // –100…+100
    r.doc        = docMap.get(r.document_id) || {};
  });

  const finalResults = filtered
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, limit);

  console.log(`Returning ${finalResults.length} final results after re-ranking`);
  
  return finalResults;
}

// ────────────────────────── prompt-budget helper ─────────────────────────── //
const MAX_PROMPT_CHARS = 19_000; //24_000; // 6 k tokens ≈ GPT-4o safety window

/** Drop low-similarity chunks until sum(text.length) ≤ MAX_PROMPT_CHARS */
function trimChunks(chunks) {
  let acc = 0;
  const out = [];
  for (const c of [...chunks].sort((a, b) => b.similarity - a.similarity)) {
    const len = c.text.length;
    if (acc + len > MAX_PROMPT_CHARS) break;
    out.push(c);
    acc += len;
  }
  
  console.log(`Trimmed chunks: ${out.length}/${chunks.length} chunks fit in ${acc}/${MAX_PROMPT_CHARS} chars`);
  return out;
}

// ────────────────────────── GPT-4 prompt builder ─────────────────────────── //
function buildPrompt(question, chunks) {
  const snippetLines = [];
  const biblioLines  = [];

  chunks.forEach((c, idx) => {
    snippetLines.push(
      `[${idx + 1}] "${c.text.trim()}" (pp. ${c.page_start}-${c.page_end})`
    );

    const d       = c.doc;
    const title   = d.title || 'Untitled';
    const authors = Array.isArray(d.authors) ? d.authors.join(', ') : d.authors;
    const journal = d.journal || 'Unknown journal';
    const year    = d.year || 'n.d.';
    const pages   = `pp. ${c.page_start}-${c.page_end}`;
    const doi     = d.doi ? `[doi:${d.doi}](https://doi.org/${d.doi})` : '';

    biblioLines.push(
      `[${idx + 1}] *${title}* · ${authors || 'Unknown'} · ${journal} (${year}) · ${pages} ${doi}`
    );
  });

  const prompt_parts = [
    'You are Misophonia Companion, a highly knowledgeable and empathetic AI assistant built to support clinicians, researchers, and individuals managing misophonia.',
    'You draw on evidence from peer-reviewed literature, clinical guidelines, and behavioral science.',
    'Your responses are clear, thoughtful, and grounded in the provided context.',
    '====',
    'QUESTION:',
    question,
    '====',
    'CONTEXT:',
    ...snippetLines,
    '====',
    'INSTRUCTIONS:',
    '• Write your answer in **Markdown**.',
    '• Begin with a concise summary (2–3 sentences).',
    '• Then elaborate on key points using well-structured paragraphs.',
    '• Provide relevant insights or suggestions (e.g., clinical, behavioral, emotional, or research-related).',
    '• If helpful, use lists, subheadings, or analogies to enhance understanding.',
    '• Use a professional and empathetic tone.',
    '• Cite sources inline like [1], [2] etc.',
    "• After the answer, include a 'BIBLIOGRAPHY:' section that lists each source exactly as provided below.",
    "• If none of the context answers the question, reply: \"I'm sorry, I don't have sufficient information to answer that.\"",
    '====',
    'BEGIN OUTPUT',
    'ANSWER:',
    '',                        // blank line – GPT populates here
    'BIBLIOGRAPHY:',
    ...biblioLines,
  ];

  return prompt_parts.join('\n');
}

// ───────────────────── citation extractor (for UI) ──────────────────────── //
const extractCitations = txt =>
  [...new Set((txt.match(/\[(\d+)]/g) || []).map(m => m.slice(1, -1)))];

// ──────────────────────────── main Lambda handler ───────────────────────── //
export async function handler(event) {
  try {
    // ── tiny ops endpoint: /stats ─────────────────────────────────────── //
    if (event.httpMethod === 'GET' && event.path.endsWith('/stats')) {
      const { count } = await sb
        .from('research_chunks')
        .select('id', { count: 'exact' })
        .limit(1);

      // Include cache stats in response
      const cacheStats = embCache.getStats();

      return {
        statusCode: 200,
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ 
          total_chunks: count,
          cache_stats: cacheStats
        }),
      };
    }

    // ── POST /search  (main RAG flow) ─────────────────────────────────── //
    if (event.httpMethod !== 'POST') {
      return { statusCode: 405, body: 'Method Not Allowed' };
    }

    const body  = JSON.parse(event.body || '{}');
    const query = (body.query || '').trim();
    const limit = body.limit ? Number(body.limit) : 8;

    if (!query) {
      return { statusCode: 400, body: 'Missing "query"' };
    }

    console.log(`Processing RAG request for query: "${query}"`);

    /* 1. Semantic retrieval */
    const rawMatches = await semanticSearch(query, limit);
    if (!rawMatches.length) {
      console.log('No matches found for query');
      return {
        statusCode: 200,
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          answer:
            "I'm sorry, I don't have sufficient information to answer that.",
          citations: [],
          results: [],
        }),
      };
    }

    /* 2. Prompt construction */
    const chunks = trimChunks(rawMatches);
    const prompt = buildPrompt(query, chunks);

    console.log(`Generated prompt with ${chunks.length} chunks`);

    /* 3. LLM generation via Groq */
    const chat = await groq.chat.completions.create({
      model: "qwen-qwq-32b", //'llama-3.3-70b-versatile',
      messages: [
        { role: 'system',
          content: `You are a supportive misophonia companion.
RULES:
• Do not expose chain-of-thought or meta reasoning.
• Do not emit <think> tags.` },
        { role: 'user', content: prompt }
      ],
      temperature: 0,
      max_tokens: 2800, //4096, =2048/2=1024*3=
      stream: false
      // stream: true,
    });

    let answerText = chat.choices[0].message.content.trim();

    /* ─ Post-filters ─────────────────────────────────────── */
    // 1. Remove <think> … </think> blocks (multiline, greedy)
    answerText = answerText.replace(/<think>[\s\S]*?<\/think>/gi, '');
    // 2. Strip any pre-answer rambling up to the first Markdown heading / paragraph
    answerText = answerText.replace(/^[\s\S]*?\n\s*?(?=#+ |\S)/, '');

    const citations = extractCitations(answerText);

    /* 4. Remove heavy embedding vectors before returning to client */
    rawMatches.forEach(m => delete m.embedding);

    console.log(`RAG request completed successfully. Cache stats:`, embCache.getStats());

    return {
      statusCode: 200,
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ answer: answerText, citations, results: rawMatches }),
    };
  } catch (err) {
    // Any failure (OpenAI, Supabase, JSON parse) → 500
    console.error('RAG λ failed →', err);
    return {
      statusCode: 500,
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ error: String(err) }),
    };
  }
}
