<!-- File: README.md -->
################################################################################
# File: README.md
################################################################################

<!-- PROJECT LOGO -->
<p align="center">
  <img src="public/vite.svg" alt="Logo" width="120" height="120">

</p>

<h1 align="center">Misophonia Companion</h1>

<p align="center">
  <b>The modern, AI-powered guide and support tool for those living with misophonia.</b><br>
  <i>Built with React, Vite, Node.js, and OpenAI</i>
  <br><br>
  <a href="https://flourishing-sprite-c819cb.netlify.app/"><img src="https://img.shields.io/badge/Live%20Demo-Online-brightgreen?style=for-the-badge" alt="Live Demo"></a>
  <a href="https://github.com/mannino49/Misophonia-companion-v2"><img src="https://img.shields.io/github/stars/mannino49/Misophonia-companion-v2?style=for-the-badge" alt="GitHub Stars"></a>
</p>

---

## 🚀 Features

- **Conversational AI Chatbot:** Powered by OpenAI, get real-time support and information.
- **Soundscape Player:** Customizable soundscapes to help manage triggers.
- **Modern UI:** Responsive, accessible, and visually appealing interface.
- **Progressive Web App:** Installable and works offline.
- **Secure Backend:** All API keys and secrets are kept on the server, never exposed to the client.

---

## 🖥️ Tech Stack

<div align="center">
  <img src="https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB" />
  <img src="https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=FFD62E" />
  <img src="https://img.shields.io/badge/Node.js-339933?style=for-the-badge&logo=nodedotjs&logoColor=white" />
  <img src="https://img.shields.io/badge/Express-000000?style=for-the-badge&logo=express&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/Netlify-00C7B7?style=for-the-badge&logo=netlify&logoColor=white" />
</div>

---

## 📦 Project Structure

```shell
Misophonia Guide/
├── public/                # Static assets (icons, manifest)
├── src/                   # React frontend source
│   ├── App.jsx            # Main app logic
│   ├── main.jsx           # React entry point
│   └── ...
├── server/                # Node.js/Express backend
│   ├── index.js           # API server entry
│   └── ...
├── netlify.toml           # Netlify deployment config
├── package.json           # Frontend config
└── ...
```

---

## ⚡ Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/g-troiani/misophonia-companion-v3
cd Misophonia-companion-v3
```

### 2. Install dependencies
```bash
npm install
cd server && npm install
```

### 3. Set up environment variables
- Copy `.env.example` to `.env` in the `server/` directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_key_here
```

### 4. Run the backend server
```bash
cd server
npm start
```

### 5. Run the frontend (in a new terminal)
```bash
npm run dev
```

- Frontend: [http://localhost:5173](http://localhost:5173)
- Backend API: [http://localhost:3001](http://localhost:3001)

---

## 🌐 Deployment

- Deployed on Netlify: [Live Demo](https://flourishing-sprite-c819cb.netlify.app/)
- Backend runs as a separate Node.js server (see `server/`)
- All secrets are stored in environment variables and never exposed to the frontend.

---

## 🛡️ Security & Best Practices

- **No secrets or API keys are stored in the frontend.**
- **.env files and private keys are gitignored.**
- **Backend validates API key presence and never exposes it to the client.**

---

## 📚 Research Pipeline Architecture

The Misophonia Companion includes a sophisticated research pipeline that processes academic literature to power the AI chatbot's knowledge base. The pipeline converts PDF research papers into searchable, semantically-organized content.

### 🔄 Pipeline Overview

The optimized modular pipeline (`scripts/pipeline_modular_optimized.py`) processes PDF documents through five sequential stages:

```
PDF → Extract & Clean → LLM Enrich → Chunk → Database → Embeddings
```

### 📊 Pipeline Statistics
- **Total Documents Processed**: 432 research papers
- **Unique Papers**: 229 distinct titles  
- **Total Knowledge Chunks**: 4,769 searchable segments
- **Processing Method**: Token-based chunking with tiktoken validation
- **Embedding Model**: OpenAI text-embedding-ada-002

---

## 🛠️ Pipeline Stages

### Stage 1-2: 📄 **Extract & Clean** 
*File: `scripts/stages/extract_clean.py`*

**Purpose**: Converts PDF research papers into structured, machine-readable JSON format.

**Process**:
- **Primary Engine**: Docling (IBM's advanced PDF parser)
- **Fallback Engines**: Unstructured → PyPDF2 (if Docling fails)
- **OCR Integration**: Tesseract for scanned/image-based PDFs
- **Text Cleaning**: Latin-1 character normalization, whitespace cleanup
- **Structure Preservation**: Maintains headings, sections, page boundaries

**Output**: 
- Structured JSON with sections, elements, and page mappings
- Clean markdown text files for human readability
- Metadata extraction from document headers

**Key Features**:
- Parallel processing with process pools
- Intelligent fallback mechanisms
- Page-level text quality assessment
- Automatic OCR for poor-quality extractions

---

### Stage 3: 🧠 **LLM Enrichment**
*File: `scripts/stages/llm_enrich.py`*

**Purpose**: Uses AI to extract comprehensive metadata from research papers.

**Process**:
- **Model**: GPT-4.1-mini-2025-04-14
- **Context Window**: 3,000 words from document start
- **Concurrent Processing**: Async API calls with rate limiting
- **Metadata Fields**: title, authors, year, journal, DOI, abstract, keywords, research_topics

**Input**: Structured JSON from extraction stage
**Output**: Enhanced JSON with AI-extracted metadata

**Key Features**:
- Smart text truncation to optimize API costs
- Parallel processing of multiple documents
- Robust error handling and fallback strategies
- JSON validation and cleaning

---

### Stage 4: ✂️ **Token-Based Chunking** 
*File: `scripts/stages/chunk_text.py`*

**Purpose**: Divides documents into optimal-sized segments for embedding and retrieval.

**Process**:
- **Method**: Token-based sliding windows (not word-based)
- **Window Size**: 3,000 tokens per chunk
- **Overlap**: 600 tokens (20% overlap for context continuity)
- **Token Counter**: tiktoken (OpenAI's official tokenizer)
- **Validation**: Guarantees no chunk exceeds API limits

**Key Improvements**:
- **Proactive Approach**: Prevents API failures by chunking based on actual tokens
- **Guaranteed Compliance**: Every chunk ≤ 6,000 token safety limit
- **Cost Optimization**: Predictable token usage for embeddings
- **Context Preservation**: Smart overlap maintains semantic continuity

**Output**: JSON files with validated, token-counted chunks

---

### Stage 5: 💾 **Database Upsert**
*File: `scripts/stages/upsert_supabase.py`*

**Purpose**: Stores processed documents and chunks in Supabase for retrieval.

**Process**:
- **Document Storage**: Metadata in `research_documents` table
- **Chunk Storage**: Text segments in `research_chunks` table  
- **Relationships**: Foreign key linking chunks to parent documents
- **Deduplication**: Prevents duplicate document processing
- **Batch Operations**: Efficient bulk inserts

**Database Schema**:
```sql
research_documents: id, title, authors, year, journal, doi, abstract, keywords
research_chunks: id, document_id, text, page_start, page_end, token_count, chunking_strategy
```

---

### Stage 6: 🔗 **Embedding Generation**
*File: `scripts/stages/embed_vectors.py`*

**Purpose**: Creates high-dimensional vector representations for semantic search.

**Process**:
- **Model**: OpenAI text-embedding-ada-002 (1536 dimensions)
- **Dynamic Batching**: Smart token-aware batch sizing
- **Rate Limiting**: Conservative API usage to prevent throttling
- **Token Validation**: Guarantees no batch exceeds 7,692 token limit
- **Quality Control**: Validates embedding generation success

**Key Features**:
- **Intelligent Batching**: Groups chunks by total token count, not just count
- **Safety Margins**: Multiple validation layers prevent API errors
- **Progress Tracking**: Detailed logging of embedding progress
- **Error Recovery**: Robust retry mechanisms for failed batches

**Performance**:
- **Batch Processing**: Multiple chunks per API call for efficiency
- **Token Counting**: Uses tiktoken for accurate token limits
- **Rate Control**: Stays under 800K tokens/minute for stability

---

## 🚀 Running the Pipeline

### Quick Start
```bash
# Process a single PDF
python3 scripts/pipeline_modular_optimized.py "path/to/research-paper.pdf"

# Process all PDFs in a directory
python3 scripts/pipeline_modular_optimized.py "documents/research/Global/*.pdf"
```

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Required environment variables
OPENAI_API_KEY=your_openai_key
SUPABASE_URL=your_supabase_url  
SUPABASE_SERVICE_ROLE_KEY=your_supabase_key
```

### Pipeline Configuration
The pipeline includes intelligent defaults:
- **Rate Limiting**: Conservative API usage (15 docs/batch, 3 concurrent)
- **Hardware Optimization**: Automatic worker scaling based on CPU cores
- **Memory Management**: Streaming for large documents
- **Progress Tracking**: Detailed logging and progress indicators

---

## 🔍 Search & Retrieval

### Vector Search Process
1. **Query Embedding**: User question → OpenAI ada-002 embedding
2. **Vector Search**: Supabase pgvector finds similar chunks
3. **Re-ranking**: Client-side cosine similarity for precision
4. **Context Assembly**: Builds prompt with relevant chunks
5. **AI Generation**: Groq's qwen-qwq-32b generates responses

### Search Features
- **Semantic Understanding**: Finds conceptually related content
- **Citation Tracking**: Every answer includes source references
- **Bibliography Generation**: Automatic scholarly citations
- **Context Limits**: 19,000 character prompt budget for optimal responses

---

## 📈 Performance Metrics

- **Processing Speed**: ~3-4 seconds per document (average)
- **Token Efficiency**: 0% API failures due to token limit errors
- **Storage Efficiency**: 4,769 chunks from 432 documents
- **Search Precision**: Vector similarity + cosine re-ranking
- **Response Quality**: Context-aware answers with source citations

---

## 🤝 Contributing

Contributions are welcome! Please open issues or submit pull requests.

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>Made with ❤️ by Mannino49</b>
</p>
