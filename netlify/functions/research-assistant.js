// File: netlify/functions/research-assistant.js
import path from 'path';
import fetch from 'node-fetch';
import dotenv from 'dotenv';
import { OpenAI } from 'openai';

dotenv.config({ path: path.resolve(process.cwd(), 'server/.env') });

// Set AI provider here - "openai" or "gemini"
const AI_PROVIDER = process.env.AI_PROVIDER || "openai";

export async function handler(event, context) {
  if (event.httpMethod !== 'POST') {
    return { statusCode: 405, body: 'Method Not Allowed' };
  }
  
  let body;
  try {
    body = JSON.parse(event.body);
  } catch {
    return { statusCode: 400, body: JSON.stringify({ error: 'Invalid JSON' }) };
  }
  
  const { messages, topic } = body;
  
  if (!messages || !Array.isArray(messages)) {
    return { statusCode: 400, body: JSON.stringify({ error: 'Messages array required' }) };
  }
  
  // Using OpenAI
  if (AI_PROVIDER === "openai") {
    const openaiKey = process.env.OPENAI_API_KEY;
    if (!openaiKey) {
      return { statusCode: 500, body: JSON.stringify({ error: 'OPENAI_API_KEY not set' }) };
    }
    
    try {
      const openai = new OpenAI({ apiKey: openaiKey });
      
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
      
      const reply = completion.choices[0]?.message?.content || '';
      return { 
        statusCode: 200, 
        body: JSON.stringify({ 
          reply, 
          structured: null,
          provider: 'openai'
        }) 
      };
    } catch (error) {
      console.error("OpenAI API error:", error);
      return { statusCode: 500, body: JSON.stringify({ error: 'Error from OpenAI API' }) };
    }
  }
  
  // Using Gemini
  else {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      return { statusCode: 500, body: JSON.stringify({ error: 'GEMINI_API_KEY not set' }) };
    }
    
    let userPrompt = '';
    if (topic && typeof topic === 'string') {
      userPrompt += `Topic: ${topic}\n`;
    }
    userPrompt += messages.map(m => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content}`).join('\n');
    const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro-preview-03-25:generateContent?key=${apiKey}`;
    
    try {
      const res = await fetch(apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: [{ role: 'user', parts: [{ text: userPrompt }] }],
          generationConfig: { temperature: 0.7, maxOutputTokens: 1024 },
          tools: [
            { function_declarations: [
                { name: 'structured_output', description: 'Return information in a structured JSON format for chat display, with sections, bullet points, and highlights.' }
              ]
            }
          ]
        })
      });
      
      if (!res.ok) {
        const err = await res.text();
        return { statusCode: 500, body: JSON.stringify({ error: 'Error from Gemini API', details: err }) };
      }
      
      const data = await res.json();
      let structured = null;
      if (data.candidates && data.candidates[0]?.content?.parts) {
        const part = data.candidates[0].content.parts[0];
        if (part.functionCall && part.functionCall.name === 'structured_output') {
          try {
            structured = JSON.parse(part.functionCall.args.json || '{}');
          } catch {
            structured = part.functionCall.args;
          }
        }
      }
      const reply = data.candidates?.[0]?.content?.parts?.[0]?.text || '';
      return { 
        statusCode: 200, 
        body: JSON.stringify({ 
          reply, 
          structured,
          provider: 'gemini'
        }) 
      };
    } catch (error) {
      console.error("Gemini API error:", error);
      return { statusCode: 500, body: JSON.stringify({ error: 'Error from Gemini API' }) };
    }
  }
}
