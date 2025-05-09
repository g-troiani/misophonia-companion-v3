<!-- File: vector-db-README.md -->
################################################################################
# File: vector-db-README.md
################################################################################
# Misophonia Research Vector Database

This project implements a comprehensive vector database for misophonia research documents using Firebase/Firestore with vector search capabilities. The system processes research PDFs, extracts text with context preservation, and generates embeddings for semantic search.

## Project Structure

```
/
├── firebase.json         # Firebase configuration
├── firestore/            # Firestore rules and indexes
├── functions/            # Cloud Functions for Firebase
│   └── index.js          # Cloud Functions for embedding generation and semantic search
└── scripts/              # Python scripts for document processing
    ├── optimized_batch_process.py       # Batch processing with optimized chunking
    ├── batch_embedding_generator.py     # Parallel embedding generation
    ├── comprehensive_search.py          # Combined semantic and keyword search
    ├── direct_semantic_search.py        # Direct semantic search without Cloud Functions
    ├── check_processed_chunks.py        # Status monitoring for embeddings
    └── test_raw_chunks.py               # Raw chunk search and statistics
```

## Setup Instructions

### Prerequisites

1. Firebase CLI installed: `npm install -g firebase-tools`
2. Firebase project created (misophonia-companion)
3. Python 3.8+ with pip
4. Node.js 18+

### Firebase Setup

1. Log in to Firebase:
   ```
   firebase login
   ```

2. Deploy Firestore rules and indexes:
   ```
   firebase deploy --only firestore
   ```

3. Install dependencies and deploy Cloud Functions:
   ```
   cd functions
   npm install
   firebase deploy --only functions
   ```

### Document Processing

1. Install Python dependencies:
   ```
   cd scripts
   pip install -r requirements.txt
   ```

2. Generate a Firebase service account key:
   - Go to Firebase console > Project settings > Service accounts
   - Click "Generate new private key"
   - Save the JSON file securely

3. Process PDF documents in batches:
   ```
   cd scripts
   python optimized_batch_process.py --batch-size 10
   ```

4. Generate embeddings for chunks:
   ```
   cd scripts
   python batch_embedding_generator.py --batch-size 100 --workers 5
   ```

5. Monitor processing progress:
   ```
   cd scripts
   python check_processed_chunks.py
   ```

## Vector Search Implementation

This project uses Firestore's vector search capabilities to implement semantic search over research documents:

1. Documents are processed and chunked with context preservation
   - Sentence-aware chunking with 2,000 character size and 300 character overlap
   - Hierarchical metadata preservation (document, section, chunk)
   - Context relationships between adjacent chunks

2. Embeddings are generated for each chunk
   - Cloud Functions trigger on new chunks in research_chunks_raw collection
   - Parallel batch processing for efficient embedding generation
   - OpenAI's text-embedding-ada-002 model (1536 dimensions)

3. Embeddings are stored in Firestore with document metadata
   - Complete document context preserved with each embedding
   - Efficient vector search indexes
   - Context expansion capabilities

4. Multiple search approaches are available
   - Cloud Function-based semantic search (semanticSearch)
   - Direct semantic search without Cloud Functions (direct_semantic_search.py)
   - Raw text search for keyword matching (test_raw_chunks.py)
   - Comprehensive search combining semantic and keyword approaches (comprehensive_search.py)

## Usage

### Cloud Function-based Semantic Search

To perform a semantic search from your application:

```javascript
// Initialize Firebase in your client app
const searchResults = await firebase.functions().httpsCallable('semanticSearch')({ 
  query: 'What are the symptoms of misophonia?',
  limit: 5,
  expandContext: true,
  filters: {
    year: [2010, 2023],  // Optional year range filter
    authors: ['Smith']   // Optional author filter
  }
});

// Process and display results
console.log(searchResults.data);
```

### Direct Semantic Search

For direct semantic search without using Cloud Functions:

```bash
cd scripts
python direct_semantic_search.py --query "What are the symptoms of misophonia?" --threshold 0.7
```

### Comprehensive Search

To use the combined semantic and keyword search capabilities:

```bash
cd scripts
python comprehensive_search.py --query "misophonia treatment approaches" --limit 5
```

### Raw Chunk Search

For simple keyword-based search and database statistics:

```bash
cd scripts
python test_raw_chunks.py --stats
python test_raw_chunks.py --topic "misophonia symptoms"
python test_raw_chunks.py --keyword "treatment"
```

## Current Status

- **Documents Processed**: 162 documents from the research collection
- **Unique Authors**: 142 researchers represented
- **Year Range**: 1978-2025 (comprehensive coverage)
- **Raw Chunks**: 56,680 chunks with optimized chunking
- **Processed Chunks**: 37,426 chunks with embeddings (66.0% progress)
- **Unique Documents with Embeddings**: 128 documents (79.0% of collection)
- **Search Capabilities**: Both semantic and keyword search working effectively with high relevance scores (>0.87)
- **Processing Efficiency**: Successfully scaled to 5000 chunks per batch with 100% success rate

## Next Steps

1. ✅ Achieve 70+ unique documents with embeddings (completed: 128 documents)
2. ✅ Achieve 100+ unique documents with embeddings (completed: 128 documents)
3. Continue embedding generation for remaining chunks to reach 100% coverage
4. Implement a web UI for searching the vector database
5. Add feedback mechanisms to improve search quality
6. Implement regular reindexing for new research documents
7. Add authentication and access controls
8. Optimize Cloud Functions for better performance with large result sets

## Resources

- [Firebase Vector Search Documentation](https://firebase.google.com/docs/firestore/vector-search)
- [Cloud Functions Documentation](https://firebase.google.com/docs/functions)
- [Firestore Documentation](https://firebase.google.com/docs/firestore)
- [OpenAI Embeddings Documentation](https://platform.openai.com/docs/guides/embeddings)
- [Implementation Plan](/documents/development/misophonia-vector-db-implementation-plan.md)
- [Processing Guide](/vector-db-processing-README.md)
