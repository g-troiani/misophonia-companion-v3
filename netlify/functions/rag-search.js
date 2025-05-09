// server/index.js – Express gateway for Misophonia Companion
// -----------------------------------------------------------------------------
// Responsibilities:
// 1. Acts as a same‑origin API gateway for the React SPA while developing / hosting
//    on Netlify.
// 2. Proxies RAG v8 Flask searches at `/api/rag` so the browser avoids CORS.
// 3. Keeps the existing Chatbot (/api/chat) and Gemini Research Assistant (/api/gemini)
//    endpoints untouched.
// -----------------------------------------------------------------------------
import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { OpenAI } from 'openai';
import fetch from 'node-fetch';

dotenv.config();

// ────────────────────────────────────────────────────────────────────────────
// Configuration
// ────────────────────────────────────────────────────────────────────────────

// Which LLM powers the Research Assistant ("openai" | "gemini") – defaults to "openai"
const AI_PROVIDER = process.env.AI_PROVIDER || 'openai';

// Location of the running RAG service (rag_web_app_v8.py).  In dev you usually run
// `python scripts/rag_web_app_v8.py` which listens on http://localhost:8080.
// In production / Netlify you can expose the Flask app via AWS Lambda, Fly.io, etc.
const RAG_HOST = process.env.RAG_HOST || 'http://localhost:8080';

const app = express();

// Allow the React dev server & Netlify preview to call us
app.use(cors({
  origin: [
    'http://localhost:5173',                // Vite dev server
    'https://misophonia-guide.netlify.app', // Production domain (update as needed)
    /https:\/\/.*--misophonia-guide--.*/   // Netlify deploy‑preview sub‑domains
  ]
}));

app.use(express.json({ limit: '1mb' }));

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ────────────────────────────────────────────────────────────────────────────
// 1. RAG v8 proxy → POST /api/rag  ⇢  Flask / search
// ────────────────────────────────────────────────────────────────────────────
app.post('/api/rag', async (req, res) => {
  try {
    // Forward the exact body (ex. { query: "…", limit: 8 }) to the Flask endpoint
    const ragRes = await fetch(`${RAG_HOST}/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body),
      // 30 s is plenty for embeddings + GPT‑4o in most cases
      timeout: 30_000
    });

    // Honour status codes from Flask (200, 4xx, 5xx)
    const payload = await ragRes.text();
    res
      .status(ragRes.status)
      .type('application/json')  // ensure JSON even for passthrough text
      .send(payload);
  } catch (error) {
    console.error('Error proxying /api/rag →', error);
    res.status(500).json({ error: 'Error calling RAG service.' });
  }
});

// ────────────────────────────────────────────────────────────────────────────
// 2. Gemini / OpenAI hybrid Research Assistant (existing logic)
//    POST /api/gemini
// ────────────────────────────────────────────────────────────────────────────
app.post('/api/gemini', async (req, res) => {
  try {
    const { messages, topic } = req.body;

    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({ error: 'Messages array required.' });
    }

    // ------------------------ OpenAI provider ------------------------------
    if (AI_PROVIDER === 'openai') {
      const systemPrompt = topic
        ? `You are a knowledgeable research assistant focusing on the topic: ${topic}. Provide information about misophonia research.`
        : 'You are a knowledgeable research assistant on misophonia research.';

      const openaiMessages = [
        { role: 'system', content: systemPrompt },
        ...messages
      ];

      const completion = await openai.chat.completions.create({
        model: 'gpt-4o',
        messages: openaiMessages,
        max_tokens: 1024,
        temperature: 0.7
      });

      return res.json({
        reply: completion.choices[0]?.message?.content || '',
        structured: null,
        provider: 'openai'
      });
    }

    // ------------------------ Gemini provider -----------------------------
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      return res.status(500).json({ error: 'GEMINI_API_KEY not set in server/.env.' });
    }

    let userPrompt = '';
    if (topic && typeof topic === 'string') {
      userPrompt += `Topic: ${topic}\n`;
    }
    userPrompt += messages.map(m => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content}`).join('\n');

    const geminiRes = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro-preview-03-25:generateContent?key=${apiKey}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [{ role: 'user', parts: [{ text: userPrompt }] }],
        generationConfig: {
          temperature: 0.7,
          maxOutputTokens: 1024
        },
        tools: [{
          function_declarations: [{
            name: 'structured_output',
            description: 'Return information in a structured JSON format for chat display, with sections, bullet points, and highlights.'
          }]
        }]
      })
    });

    if (!geminiRes.ok) {
      const err = await geminiRes.text();
      return res.status(500).json({ error: 'Error from Gemini API', details: err });
    }

    const geminiData = await geminiRes.json();
    let structured = null;
    if (geminiData.candidates && geminiData.candidates[0]?.content?.parts) {
      const part = geminiData.candidates[0].content.parts[0];
      if (part.functionCall && part.functionCall.name === 'structured_output') {
        try {
          structured = JSON.parse(part.functionCall.args.json || '{}');
        } catch {
          structured = part.functionCall.args;
        }
      }
    }

    return res.json({
      reply: geminiData.candidates?.[0]?.content?.parts?.[0]?.text || '',
      structured,
      provider: 'gemini'
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: `Error from ${AI_PROVIDER === 'openai' ? 'OpenAI' : 'Gemini'} API.` });
  }
});

// ────────────────────────────────────────────────────────────────────────────
// 3. Generic Chatbot (existing logic) – POST /api/chat
// ────────────────────────────────────────────────────────────────────────────
app.post('/api/chat', async (req, res) => {
  try {
    const { messages } = req.body;
    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({ error: 'Messages array required.' });
    }
    const completion = await openai.chat.completions.create({
      model: 'gpt-4o',
      messages,
      max_tokens: 512,
      temperature: 0.7
    });
    const reply = completion.choices[0]?.message?.content || '';
    res.json({ reply });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Error from OpenAI API.' });
  }
});

// ────────────────────────────────────────────────────────────────────────────
// Launch
// ────────────────────────────────────────────────────────────────────────────
const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`AI provider   : ${AI_PROVIDER.toUpperCase()}`);
  console.log(`/api/rag proxy : ${RAG_HOST}/search`);
});
