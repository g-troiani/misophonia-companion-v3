// File: netlify/functions/chat.js
import path from 'path';
import dotenv from 'dotenv';
import { OpenAI } from 'openai';

// Load env vars from Python server .env
dotenv.config({ path: path.resolve(process.cwd(), 'server/.env') });

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
  const { messages } = body;
  if (!messages || !Array.isArray(messages)) {
    return { statusCode: 400, body: JSON.stringify({ error: 'Messages array required' }) };
  }
  // Initialize OpenAI client
  const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  try {
    const completion = await openai.chat.completions.create({
      model: 'gpt-4o',
      messages,
      max_tokens: 512,
      temperature: 0.7
    });
    const reply = completion.choices?.[0]?.message?.content || '';
    return { statusCode: 200, body: JSON.stringify({ reply }) };
  } catch (err) {
    console.error(err);
    return { statusCode: 500, body: JSON.stringify({ error: 'Error from OpenAI API.' }) };
  }
}
