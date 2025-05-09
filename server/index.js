// File: server/index.js
import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { OpenAI } from 'openai';
import fetch from 'node-fetch';
import httpProxy from 'http-proxy';

dotenv.config();

// Set AI provider here - "openai" or "gemini"
const AI_PROVIDER = process.env.AI_PROVIDER || "openai";

const app = express();

//app.use(cors());  --> This is the default behavior
app.use(cors({
  origin: [
    '[https://misophonia-guide.netlify.app](https://misophonia-guide.netlify.app)',
    'http://localhost:5173'
  ]
}));
app.use(express.json());

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Research Assistant endpoint (handles both Gemini and OpenAI)
app.post('/api/gemini', async (req, res) => {
  try {
    const { messages, topic } = req.body;
    
    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({ error: 'Messages array required.' });
    }
    
    // Using OpenAI
    if (AI_PROVIDER === "openai") {
      // Format messages for OpenAI
      const systemPrompt = topic 
        ? `You are a knowledgeable research assistant focusing on the topic: ${topic}. Provide information about misophonia research.`
        : "You are a knowledgeable research assistant on misophonia research.";
      
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
    
    // Using Gemini
    else {
      const apiKey = process.env.GEMINI_API_KEY;
      if (!apiKey) {
        return res.status(500).json({ error: 'GEMINI_API_KEY not set in server/.env.' });
      }
      
      // Compose prompt for Gemini
      let userPrompt = '';
      if (topic && typeof topic === 'string') {
        userPrompt += `Topic: ${topic}\n`;
      }
      userPrompt += messages.map(m => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content}`).join('\n');
      
      // Gemini API call
      const geminiRes = await fetch('https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro-preview-03-25:generateContent?key=' + apiKey, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: [{ role: 'user', parts: [{ text: userPrompt }] }],
          generationConfig: {
            temperature: 0.7,
            maxOutputTokens: 1024,
          },
          tools: [
            { "function_declarations": [
              { "name": "structured_output", "description": "Return information in a structured JSON format for chat display, with sections, bullet points, and highlights." }
            ]}
          ]
        })
      });
      
      if (!geminiRes.ok) {
        const err = await geminiRes.text();
        return res.status(500).json({ error: 'Error from Gemini API', details: err });
      }
      
      const geminiData = await geminiRes.json();
      // Parse structured output if present
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
    }
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: `Error from ${AI_PROVIDER === "openai" ? "OpenAI" : "Gemini"} API.` });
  }
});

// Keep existing OpenAI chat endpoint
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

// ───────────────────────── RAG proxy ─────────────────────────
const proxy = httpProxy.createProxyServer({ changeOrigin: true });

app.post('/api/rag', (req, res) => {
  proxy.web(req, res, { target: 'http://localhost:8080/search' }, err => {
    console.error(err);
    res.status(502).json({ error: 'RAG service unreachable' });
  });
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`Server running with ${AI_PROVIDER.toUpperCase()} on port ${PORT}`);
});
