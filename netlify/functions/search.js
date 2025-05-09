// File: netlify/functions/search.js
import { createClient } from '@supabase/supabase-js';
import { Configuration, OpenAIApi } from 'openai';

export async function handler(event, context) {
  if (event.httpMethod !== 'POST') {
    return { statusCode: 405, body: 'Method Not Allowed' };
  }
  
  try {
    const body = JSON.parse(event.body);
    const { query, filters, expandContext, page, pageSize, similarityThreshold } = body;
    
    // Initialize Supabase client
    const supabase = createClient(
      process.env.SUPABASE_URL,
      process.env.SUPABASE_SERVICE_ROLE_KEY
    );
    
    // Initialize OpenAI for embeddings
    const configuration = new Configuration({
      apiKey: process.env.OPENAI_API_KEY,
    });
    const openai = new OpenAIApi(configuration);
    
    // Generate embedding
    const embeddingResponse = await openai.createEmbedding({
      model: "text-embedding-ada-002",
      input: query,
    });
    
    const queryEmbedding = embeddingResponse.data.data[0].embedding;
    
    // Search using the RPC function
    const { data: results, error } = await supabase.rpc(
      'match_documents',
      {
        query_embedding: queryEmbedding,
        match_threshold: similarityThreshold || 0.6,
        match_count: pageSize || 10
      }
    );
    
    if (error) throw error;
    
    return {
      statusCode: 200,
      body: JSON.stringify({
        result: {
          results,
          page: page || 1,
          pageSize: pageSize || 10,
          query
        }
      })
    };
  } catch (error) {
    console.error('Error in search function:', error);
    return {
      statusCode: 500,
      body: JSON.stringify({ error: error.message })
    };
  }
}
