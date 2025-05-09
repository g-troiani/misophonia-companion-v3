// File: scripts/ingest.js
import fs from 'fs';
import path from 'path';
import pdfjsLib from 'pdfjs-dist/legacy/build/pdf.js';
const { getDocument } = pdfjsLib;
import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';

// Load environment variables (make sure FIREBASE_* vars are set in server/.env)
dotenv.config({ path: path.resolve(process.cwd(), 'server/.env') });

// Initialize Supabase
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

// Function to chunk text
function chunkText(text, size = 1000, overlap = 200) {
  const chunks = [];
  for (let start = 0; start < text.length; start += size - overlap) {
    const chunk = text.slice(start, start + size);
    chunks.push(chunk.trim());
  }
  return chunks;
}

async function processPdf(filePath) {
  const buffer = fs.readFileSync(filePath);
  const loadingTask = getDocument({ data: buffer });
  const pdfDoc = await loadingTask.promise;
  const numPages = pdfDoc.numPages;
  let text = '';
  for (let i = 1; i <= numPages; i++) {
    const page = await pdfDoc.getPage(i);
    const content = await page.getTextContent();
    const strings = content.items.map(item => item.str);
    text += strings.join(' ') + '\n';
  }
  const chunks = chunkText(text);
  const basename = path.basename(filePath, '.pdf');
  
  // Prepare chunks for insertion
  const chunksToInsert = chunks.map((chunk, idx) => ({
    file: basename,
    chunk_index: idx,
    text: chunk,
    created_at: new Date()
  }));
  
  // Insert in batches of 500
  const BATCH_SIZE = 500;
  for (let i = 0; i < chunksToInsert.length; i += BATCH_SIZE) {
    const batch = chunksToInsert.slice(i, i + BATCH_SIZE);
    const { error } = await supabase
      .from('research_chunks')
      .insert(batch);
    
    if (error) {
      console.error('Error inserting batch:', error);
      throw error;
    }
  }
  
  console.log(`Indexed ${chunks.length} chunks for ${basename}`);
}

async function main() {
  const dir = path.resolve(process.cwd(), 'documents/research/Global');
  const files = fs.readdirSync(dir).filter(f => f.toLowerCase().endsWith('.pdf'));
  for (const file of files) {
    console.log(`Processing ${file}...`);
    await processPdf(path.join(dir, file));
  }
  console.log('Ingestion complete.');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
