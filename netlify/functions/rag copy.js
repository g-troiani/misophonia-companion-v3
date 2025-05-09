/**
 * Proxy to the Python RAG service (Flask) OR any upstream REST endpoint.
 * POST /.netlify/functions/rag    →  { query, limit? }
 *
 * Required ENV:
 *   RAG_HOST               e.g. https://misophonia-rag.fly.dev  (no trailing /)
 *
 * Optional:
 *   RAG_TIMEOUT_MS         default 30000
 */
import 'dotenv/config';

const RAG_HOST = process.env.RAG_HOST ?? 'http://localhost:8080';
const TIMEOUT  = Number(process.env.RAG_TIMEOUT_MS ?? 30_000);

export async function handler(event /* , context */) {
  if (event.httpMethod !== 'POST') {
    return { statusCode: 405, body: 'Method Not Allowed' };
  }

  try {
    const upstream = await fetch(`${RAG_HOST}/search`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body:     event.body,
      signal:   AbortSignal.timeout(TIMEOUT),
    });

    const text = await upstream.text();     // pass through raw

    return {
      statusCode: upstream.status,
      headers: { 'content-type': 'application/json' },
      body: text,
    };
  } catch (err) {
    console.error('RAG proxy error →', err);
    return {
      statusCode: 500,
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ error: 'Error contacting RAG service' }),
    };
  }
}
