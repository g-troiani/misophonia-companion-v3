<!-- 
 * File: documents/development/misophonia-vector-db-implementation-plan.md
  -->
################################################################################
# File: documents/development/misophonia-vector-db-implementation-plan.md
################################################################################
# Misophonia Research Vector Database: Implementation Plan

## Project Overview

This plan outlines the implementation of a comprehensive vector database for Misophonia research documents using Firebase/Firestore. The system will enable semantic search across research papers with context preservation and advanced filtering capabilities.

```
PDFs → Extraction → Chunking → Embedding → Storage → API → UI
```

## Current Status

✅ **Completed:**
- Firebase project setup with proper configuration
- Firestore database created (using the default database)
- Firestore vector search indexes configured
- Cloud Functions deployed:
  - `helloWorld`: Test function
  - `generateEmbeddings`: For processing document chunks
  - `semanticSearch`: For vector similarity search
- Python script for PDF processing with context-aware chunking
- Service account key for Firebase authentication

⏳ **In Progress:**
- Testing PDF processing pipeline
- Implementing actual embedding generation (currently using placeholder)

## Implementation Timeline

### 1. Document Processing Pipeline (Week 1)

- [x] Set up Firebase project structure
- [x] Create basic PDF processing script
- [ ] Complete document extraction implementation:
  ```python
  # Core extraction function using PyPDF2 + unstructured
  def process_pdf(pdf_path):
      # Extract text, preserve structure and metadata
      pdf_doc = unstructured.partition.pdf(pdf_path, strategy="hi_res")
      
      # Extract metadata from filename using regex
      metadata = extract_metadata_from_filename(os.path.basename(pdf_path))
      
      return {
          "content": pdf_doc,
          "metadata": metadata,
          "filepath": pdf_path
      }
  ```
- [ ] Test extraction with sample PDFs
- [ ] Implement OCR for scanned documents
- [ ] Optimize extraction for different PDF formats

### 2. Hierarchical Chunking (Week 1-2)

- [x] Implement basic chunking strategy
- [ ] Enhance context-aware chunking:
  ```python
  def create_chunks_with_context(document):
      chunks = []
      
      # Create document-level metadata
      doc_metadata = {
          "title": document["metadata"]["title"],
          "authors": document["metadata"]["authors"],
          "year": document["metadata"]["year"],
          "filepath": document["filepath"]
      }
      
      # Split by sections first
      sections = split_by_headings(document["content"])
      
      for section_idx, section in enumerate(sections):
          # Create semantic chunks with overlap
          section_chunks = create_overlapping_chunks(
              section["text"], 
              chunk_size=750,
              overlap=150
          )
          
          for chunk_idx, chunk_text in enumerate(section_chunks):
              chunk = {
                  "text": chunk_text,
                  "metadata": {
                      **doc_metadata,
                      "section": section["heading"],
                      "section_idx": section_idx,
                      "chunk_idx": chunk_idx,
                      "total_chunks": len(section_chunks),
                      "context_chunks": [
                          chunk_idx - 1 if chunk_idx > 0 else None,
                          chunk_idx + 1 if chunk_idx < len(section_chunks) - 1 else None
                      ]
                  }
              }
              chunks.append(chunk)
      
      return chunks
  ```
- [ ] Optimize chunk size and overlap for best search results
- [ ] Test chunking with various document types

### 3. Vector Database Setup (Week 2-3)

- [x] Set up Firebase project with Firestore
- [x] Create Firestore security rules
  ```javascript
  // firestore.rules
  rules_version = '2';
  service cloud.firestore {
    match /databases/{database}/documents {
      match /research_chunks/{chunk} {
        allow read: if true;
        allow write: if request.auth.token.admin == true;
      }
    }
  }
  ```
- [x] Configure Firestore Vector Search indexes
- [x] Deploy basic Cloud Functions
- [ ] Implement authentication and access controls
- [ ] Set up monitoring and logging

### 4. Embedding Generation & Storage (Week 2-3)

- [x] Create placeholder embedding generation function
- [ ] Implement actual embedding generation using OpenAI or Vertex AI:
  ```javascript
  // Cloud Function for embedding generation
  exports.generateEmbeddings = functions.firestore
    .document('research_chunks_raw/{chunkId}')
    .onCreate(async (snap, context) => {
      const chunkData = snap.data();
      
      // Generate embedding via OpenAI API or Vertex AI
      const embedding = await generateEmbedding(chunkData.text);
      
      // Store in vectorized collection
      return admin.firestore().collection('research_chunks')
        .doc(context.params.chunkId)
        .set({
          text: chunkData.text,
          embedding: embedding,
          metadata: chunkData.metadata
        });
    });
  ```
- [ ] Set up API key management and security
- [ ] Implement batched embedding generation for efficiency
- [ ] Test embedding quality and performance

### 5. Search & Retrieval API (Week 3-4)

- [x] Create basic vector search function
- [ ] Enhance semantic search implementation:
  ```javascript
  // Cloud Function for semantic search
  exports.semanticSearch = functions.https.onCall(async (data, context) => {
    const { query, limit = 5, expand_context = true } = data;
    
    // Generate query embedding
    const queryEmbedding = await generateEmbedding(query);
    
    // Perform vector search
    const results = await admin.firestore()
      .collection('research_chunks')
      .where('embedding', '!=', null)
      .findNearest('embedding', queryEmbedding, limit)
      .get();
    
    // Process results
    let searchResults = results.docs.map(doc => ({
      id: doc.id,
      text: doc.data().text,
      score: doc.score,
      metadata: doc.data().metadata
    }));
    
    // Expand context if requested
    if (expand_context) {
      searchResults = await expandResultsContext(searchResults);
    }
    
    return { results: searchResults };
  });
  ```
- [ ] Implement context expansion:
  ```javascript
  async function expandResultsContext(results) {
    const expandedResults = [];
    
    for (const result of results) {
      // Get the current chunk's metadata
      const { section_idx, chunk_idx, total_chunks } = result.metadata;
      
      // Determine which context chunks to fetch
      const contextChunkIds = [];
      if (chunk_idx > 0) contextChunkIds.push(`${section_idx}_${chunk_idx-1}`);
      if (chunk_idx < total_chunks-1) contextChunkIds.push(`${section_idx}_${chunk_idx+1}`);
      
      // Fetch context chunks
      const contextChunks = await fetchChunksByIds(contextChunkIds);
      
      // Add to expanded results
      expandedResults.push({
        ...result,
        context: {
          before: contextChunks.filter(c => c.metadata.chunk_idx < chunk_idx),
          after: contextChunks.filter(c => c.metadata.chunk_idx > chunk_idx)
        }
      });
    }
    
    return expandedResults;
  }
  ```
- [ ] Add pre-filtering by metadata (year, authors)
- [ ] Implement citation formatting
- [ ] Optimize search parameters (k, distance threshold)

### 6. Frontend Integration (Week 4-5)

- [ ] Create search interface component:
  ```jsx
  // SearchInterface.jsx
  import React, { useState } from 'react';
  import { functions } from './firebase';

  export default function SearchInterface() {
    const [query, setQuery] = useState('');
    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(false);
    
    const handleSearch = async () => {
      setLoading(true);
      try {
        const semanticSearch = functions.httpsCallable('semanticSearch');
        const response = await semanticSearch({ 
          query, 
          limit: 5,
          expand_context: true
        });
        setResults(response.data.results);
      } catch (error) {
        console.error('Search error:', error);
      } finally {
        setLoading(false);
      }
    };
    
    return (
      <div className="search-container">
        <div className="search-input">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search research papers..."
          />
          <button onClick={handleSearch} disabled={loading}>
            {loading ? 'Searching...' : 'Search'}
          </button>
        </div>
        
        <div className="search-results">
          {results.map(result => (
            <SearchResult key={result.id} result={result} />
          ))}
        </div>
      </div>
    );
  }
  ```
- [ ] Implement search result component:
  ```jsx
  // SearchResult.jsx
  function SearchResult({ result }) {
    const [expanded, setExpanded] = useState(false);
    
    const { 
      text, 
      metadata: { title, authors, year, section },
      context
    } = result;
    
    return (
      <div className="search-result">
        <div className="result-metadata">
          <h3>{title} ({year})</h3>
          <p className="authors">{authors.join(', ')}</p>
          <p className="section">Section: {section}</p>
        </div>
        
        <div className="result-content">
          {/* Show context before if available */}
          {expanded && context?.before?.map(chunk => (
            <p className="context-before">{chunk.text}</p>
          ))}
          
          {/* Main result */}
          <p className="main-result">{text}</p>
          
          {/* Show context after if available */}
          {expanded && context?.after?.map(chunk => (
            <p className="context-after">{chunk.text}</p>
          ))}
        </div>
        
        <div className="result-actions">
          <button onClick={() => setExpanded(!expanded)}>
            {expanded ? 'Show Less' : 'Show Context'}
          </button>
          <a href={`/view-source?doc=${encodeURIComponent(title)}`}>
            View Source Document
          </a>
        </div>
      </div>
    );
  }
  ```
- [ ] Add filtering UI components
- [ ] Implement pagination for results
- [ ] Create loading and error states

### 7. User Experience Features (Week 5-6)

- [ ] Implement advanced search features:
  - [ ] Filters for date range, authors, topics
  - [ ] Saved searches for frequent queries
  - [ ] Recent searches history
  - [ ] "More like this" for related content
- [ ] Add citation capabilities:
  - [ ] Generate citations in various formats (APA, MLA, Chicago)
  - [ ] Copy citation to clipboard
  - [ ] Export citations to reference manager
- [ ] Create document viewer:
  - [ ] In-app PDF viewer with highlights
  - [ ] Jump to cited section from search results
  - [ ] Annotation capabilities

### 8. Testing & Optimization (Week 6-8)

- [ ] Conduct quality testing:
  - [ ] Define 20-30 test queries covering different aspects
  - [ ] Rate search results for relevance
  - [ ] Compare against baseline (e.g., keyword search)
  - [ ] Iteratively improve based on results
- [ ] Perform performance optimization:
  - [ ] Measure and optimize query latency
  - [ ] Implement caching for frequent queries
  - [ ] Monitor and optimize embedding generation costs
  - [ ] Implement pagination for large result sets

### 9. Deployment & Monitoring

- [ ] Execute deployment strategy:
  - [ ] Start with development environment
  - [ ] Create staging environment with subset of documents
  - [ ] Performance and quality testing in staging
  - [ ] Gradual rollout to production
- [ ] Set up monitoring:
  - [ ] Configure Firebase monitoring for functions
  - [ ] Track query volume and performance
  - [ ] Implement user feedback mechanism
  - [ ] Create dashboard for system health

## Technical Architecture

### Components

1. **Document Processing**
   - Python script for PDF extraction and chunking
   - Metadata extraction from filenames
   - Context-aware chunking with relationships

2. **Firebase Infrastructure**
   - Firestore database with vector search capabilities
   - Cloud Functions for serverless processing
   - Authentication and security rules

3. **Embedding Generation**
   - OpenAI or Vertex AI integration
   - Batched processing for efficiency
   - Caching for frequently used embeddings

4. **Search API**
   - Vector similarity search with pre-filtering
   - Context expansion for better results
   - Relevance scoring and ranking

5. **User Interface**
   - Search interface with filtering
   - Result display with context expansion
   - Citation and document viewing capabilities

## Data Flow

1. **Document Ingestion**
   ```
   PDF → Text Extraction → Chunking → Upload to research_chunks_raw
   ```

2. **Embedding Generation**
   ```
   New Chunk in research_chunks_raw → Cloud Function Trigger → Generate Embedding → Store in research_chunks
   ```

3. **Search Process**
   ```
   User Query → Generate Query Embedding → Vector Search → Context Expansion → Return Results
   ```

## Resources & References

- [Firebase Vector Search Documentation](https://firebase.google.com/docs/firestore/vector-search)
- [Cloud Functions Documentation](https://firebase.google.com/docs/functions)
- [Firestore Documentation](https://firebase.google.com/docs/firestore)
- [Vertex AI Embeddings](https://cloud.google.com/vertex-ai/docs/generative-ai/embeddings/get-text-embeddings)
- [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings)

## Next Steps

1. Complete the PDF processing implementation and test with sample documents
2. Implement actual embedding generation using OpenAI or Vertex AI
3. Enhance the search function with context expansion and filtering
4. Begin frontend integration with basic search interface
