// File: netlify/functions/_utils.js
// Shared constants for Netlify functions

// Which LLM powers the Research Assistant ("openai" | "gemini") – defaults to "openai"
export const AI_PROVIDER = process.env.AI_PROVIDER || 'openai';

// Location of the running RAG service (rag_web_app_v9.py)
// Hard-wire the production URL so the function works even
// when the env var is missing.
export const RAG_HOST =
  process.env.RAG_HOST || 'https://misophonia-rag.fly.dev';   // ← your URL