// File: scripts/migrate_to_supabase.js
import fs from 'fs';
import path from 'path';
import admin from 'firebase-admin';
import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config({ path: path.resolve(process.cwd(), 'server/.env') });

// Initialize Firebase Admin SDK (source)
const serviceAccount = {
  projectId: process.env.FIREBASE_PROJECT_ID,
  clientEmail: process.env.FIREBASE_CLIENT_EMAIL,
  privateKey: process.env.FIREBASE_PRIVATE_KEY?.replace(/\\n/g, '\n'),
};
admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
});
const db = admin.firestore();

// Initialize Supabase (destination)
const supabase = createClient(
  process.env.SUPABASE_URL || '',
  process.env.SUPABASE_ANON_KEY || ''
);

// Export Firestore data
async function exportFirestoreData() {
  console.log('Exporting data from Firestore...');
  
  const chunks = await db.collection('research_chunks').get();
  const data = chunks.docs.map(doc => ({
    id: doc.id,
    ...doc.data(),
    // Convert Firestore timestamp to ISO string
    created_at: doc.data().createdAt?.toDate().toISOString()
  }));
  
  fs.writeFileSync('chunks_export.json', JSON.stringify(data));
  console.log(`Exported ${data.length} documents to chunks_export.json`);
  return data;
}

// Import to Supabase
async function importToSupabase(data) {
  console.log('Importing data to Supabase...');
  
  // Insert in batches
  const BATCH_SIZE = 500;
  for (let i = 0; i < data.length; i += BATCH_SIZE) {
    const batch = data.slice(i, i + BATCH_SIZE);
    
    console.log(`Importing batch ${Math.floor(i/BATCH_SIZE) + 1}/${Math.ceil(data.length/BATCH_SIZE)}...`);
    
    // Prepare records for Supabase
    const records = batch.map(item => {
      // Remove Firestore-specific fields and rename fields to match Supabase schema
      const { createdAt, embedding, ...rest } = item;
      return {
        ...rest,
        // Use created_at from previous conversion or current date
        created_at: item.created_at || new Date().toISOString()
      };
    });
    
    const { error } = await supabase
      .from('research_chunks')
      .insert(records);
    
    if (error) {
      console.error('Error importing batch:', error);
    } else {
      console.log(`Successfully imported batch ${Math.floor(i/BATCH_SIZE) + 1}`);
    }
  }
  
  console.log('Import completed!');
}

// Main function
async function main() {
  try {
    // Check if export file exists
    if (fs.existsSync('chunks_export.json')) {
      console.log('Using existing export file: chunks_export.json');
      const data = JSON.parse(fs.readFileSync('chunks_export.json'));
      await importToSupabase(data);
    } else {
      const data = await exportFirestoreData();
      await importToSupabase(data);
    }
  } catch (error) {
    console.error('Migration failed:', error);
  }
}

main().catch(console.error);