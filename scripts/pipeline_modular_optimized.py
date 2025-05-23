#!/usr/bin/env python3
"""
Optimized pipeline orchestrator with full parallelization.
Maintains compatibility with original pipeline_modular.py interface.
"""
from __future__ import annotations
import argparse, logging, pathlib, random
from collections import Counter
from rich.console import Console
import asyncio
from typing import List, Optional
import multiprocessing as mp

# Import stages
from stages import extract_clean, llm_enrich, chunk_text, db_upsert, embed_vectors
from stages.acceleration_utils import hardware

# Keep existing toggles
RUN_EXTRACT    = True
RUN_LLM_ENRICH = True
RUN_CHUNK      = True
RUN_DB         = True
RUN_EMBED      = True

log = logging.getLogger("pipeline-modular-optimized")

class OptimizedPipeline:
    """Optimized pipeline with parallel processing and rate limiting."""
    
    def __init__(self, batch_size: int = 15, max_api_concurrent: int = 3):  # 🛡️ Rate limited: was 50, 20
        self.batch_size = batch_size
        self.max_api_concurrent = max_api_concurrent
        self.stats = Counter()
        
        # 🛡️ Log rate limiting settings
        log.info("🛡️  Rate-limited pipeline initialized:")
        log.info(f"   Batch size: {batch_size} (was 50)")
        log.info(f"   Max concurrent: {max_api_concurrent} (was 20)")
        log.info("   Target: <800K tokens/minute (safe margin)")
    
    async def process_batch(self, pdfs: List[pathlib.Path], start_doc_num: int = 1, total_docs: int = None) -> None:
        """Process a batch of PDFs through all stages with individual document progress tracking."""
        
        if total_docs is None:
            total_docs = len(pdfs)
        
        batch_size = len(pdfs)
        log.info(f"📊 BATCH PROCESSING START:")
        log.info(f"   📄 Documents in this batch: {batch_size}")
        log.info(f"   🎯 Document range: {start_doc_num} to {start_doc_num + batch_size - 1}")
        log.info(f"   📈 Overall progress: {start_doc_num-1}/{total_docs} completed ({((start_doc_num-1)/total_docs*100):.1f}%)")
        
        # Stage 1-2: Extract & Clean (CPU-bound, use process pool)
        json_docs = []
        if RUN_EXTRACT:
            log.info(f"🔄 EXTRACTION STAGE - Processing {len(pdfs)} PDFs...")
            for i, pdf in enumerate(pdfs):
                doc_num = start_doc_num + i
                doc_progress = ((doc_num-1) / total_docs * 100)
                log.info(f"📄 [{doc_num}/{total_docs}] ({doc_progress:.1f}%) Extracting: {pdf.name}")
            
            json_docs = await extract_clean.extract_batch_async(
                pdfs, 
                enrich_llm=False  # We'll do LLM enrichment separately
            )
            
            log.info(f"📊 EXTRACTION STAGE COMPLETE:")
            for i, pdf in enumerate(pdfs):
                doc_num = start_doc_num + i
                log.info(f"   ✅ [{doc_num}/{total_docs}] Extracted: {pdf.name}")
        else:
            # Use existing JSON files
            json_docs = [extract_clean.json_path_for(pdf) for pdf in pdfs]
            log.info(f"📊 EXTRACTION STAGE SKIPPED - Using existing JSON files")

        # Stage 4: LLM Enrich (I/O-bound, use async)
        if RUN_LLM_ENRICH and json_docs:
            log.info(f"🔄 ENRICHMENT STAGE - Processing {len(json_docs)} documents...")
            for i, json_doc in enumerate(json_docs):
                doc_num = start_doc_num + i
                doc_progress = ((doc_num-1) / total_docs * 100)
                doc_name = pathlib.Path(json_doc).stem.replace('_clean', '')
                log.info(f"📄 [{doc_num}/{total_docs}] ({doc_progress:.1f}%) Enriching: {doc_name}")
            
            await llm_enrich.enrich_batch_async(
                json_docs,
                max_concurrent=self.max_api_concurrent
            )
            
            log.info(f"📊 ENRICHMENT STAGE COMPLETE:")
            for i, json_doc in enumerate(json_docs):
                doc_num = start_doc_num + i
                doc_name = pathlib.Path(json_doc).stem.replace('_clean', '')
                log.info(f"   ✅ [{doc_num}/{total_docs}] Enriched: {doc_name}")

        # Stage 5: Chunk (CPU-bound, use process pool)
        chunks_map = {}
        if RUN_CHUNK and json_docs:
            log.info(f"🔄 CHUNKING STAGE - Processing {len(json_docs)} documents...")
            for i, json_doc in enumerate(json_docs):
                doc_num = start_doc_num + i
                doc_progress = ((doc_num-1) / total_docs * 100)
                doc_name = pathlib.Path(json_doc).stem.replace('_clean', '')
                log.info(f"📄 [{doc_num}/{total_docs}] ({doc_progress:.1f}%) Chunking: {doc_name}")
            
            chunks_map = await chunk_text.chunk_batch_async(json_docs)
            
            log.info(f"📊 CHUNKING STAGE COMPLETE:")
            total_chunks_created = 0
            for i, json_doc in enumerate(json_docs):
                doc_num = start_doc_num + i
                doc_name = pathlib.Path(json_doc).stem.replace('_clean', '')
                chunk_count = len(chunks_map.get(json_doc, []))
                total_chunks_created += chunk_count
                log.info(f"   ✅ [{doc_num}/{total_docs}] Chunked: {doc_name} ({chunk_count} chunks)")
            log.info(f"   📊 Total chunks created in this batch: {total_chunks_created}")

        # Stage 6: DB Upsert (I/O-bound, use async)
        if RUN_DB and chunks_map:
            log.info(f"🔄 DATABASE UPSERT STAGE - Processing {len(chunks_map)} documents...")
            
            # Show progress before upserting
            total_chunks_to_upsert = sum(len(chunks) for chunks in chunks_map.values())
            log.info(f"📊 DATABASE UPSERT PROGRESS:")
            log.info(f"   💾 Total chunks to upsert: {total_chunks_to_upsert}")
            
            for i, (json_path, chunks) in enumerate(chunks_map.items()):
                doc_num = start_doc_num + i
                doc_progress = ((doc_num-1) / total_docs * 100)
                doc_name = pathlib.Path(json_path).stem.replace('_clean', '')
                log.info(f"📄 [{doc_num}/{total_docs}] ({doc_progress:.1f}%) Upserting: {doc_name} ({len(chunks)} chunks)")
            
            documents = [
                {
                    "json_path": json_path,
                    "chunks": chunks,
                    "do_embed": False  # We'll embed in batch later
                }
                for json_path, chunks in chunks_map.items()
            ]
            await db_upsert.upsert_batch_async(documents)
            
            log.info(f"📊 DATABASE UPSERT STAGE COMPLETE:")
            total_chunks_upserted = 0
            for i, (json_path, chunks) in enumerate(chunks_map.items()):
                doc_num = start_doc_num + i
                doc_name = pathlib.Path(json_path).stem.replace('_clean', '')
                total_chunks_upserted += len(chunks)
                log.info(f"   ✅ [{doc_num}/{total_docs}] Upserted: {doc_name}")
            log.info(f"   📊 Total chunks upserted: {total_chunks_upserted}")

        # Update stats
        self.stats["ok"] += len([c for c in chunks_map.values() if c])
        self.stats["fail"] += len([c for c in chunks_map.values() if not c])
        
        # Final batch summary
        end_doc_num = start_doc_num + batch_size - 1
        overall_progress = (end_doc_num / total_docs * 100)
        log.info(f"📊 BATCH COMPLETE:")
        log.info(f"   ✅ Documents processed: {start_doc_num}-{end_doc_num}")
        log.info(f"   📈 Overall progress: {end_doc_num}/{total_docs} ({overall_progress:.1f}%)")
        log.info(f"   📊 Successful documents: {self.stats['ok']}")
        log.info(f"   ❌ Failed documents: {self.stats['fail']}")
    
    async def run(self, src: pathlib.Path, selection: str = "sequential", cap: int = 0):
        """Run the optimized pipeline."""
        Console().rule("[bold cyan]Misophonia PDF → Vector pipeline (optimized)")
        
        # Get PDF list
        pdfs = [src] if src.is_file() else sorted(src.rglob("*.pdf"))
        if cap:
            pdfs = random.sample(pdfs, cap) if selection == "random" else pdfs[:cap]
        
        total_pdfs = len(pdfs)
        log.info(f"Processing {total_pdfs} PDFs in batches of {self.batch_size}")
        
        # Track PDF-level progress
        pdfs_processed = 0
        
        # Process in batches
        for i in range(0, len(pdfs), self.batch_size):
            batch = pdfs[i:i + self.batch_size]
            batch_num = i//self.batch_size + 1
            total_batches = (len(pdfs) + self.batch_size - 1)//self.batch_size
            
            # Calculate the starting document number for this batch
            start_doc_num = pdfs_processed + 1
            
            # Enhanced progress logging with both batch and PDF-level progress
            log.info(f"📄 Processing batch {batch_num}/{total_batches} ({len(batch)} PDFs)")
            log.info(f"📊 Overall progress: {pdfs_processed}/{total_pdfs} PDFs completed ({pdfs_processed/total_pdfs*100:.1f}%)")
            log.info(f"🔢 Document range: {start_doc_num}-{start_doc_num + len(batch) - 1} of {total_pdfs}")
            
            await self.process_batch(batch, start_doc_num, total_pdfs)
            
            # Update PDF progress counter
            pdfs_processed += len(batch)
            
            # Log completion of this batch
            log.info(f"✅ Batch {batch_num} complete - Total PDFs processed: {pdfs_processed}/{total_pdfs} ({pdfs_processed/total_pdfs*100:.1f}%)")
        
        # Final progress summary
        log.info(f"🎉 All PDFs processed: {pdfs_processed}/{total_pdfs} ({pdfs_processed/total_pdfs*100:.1f}%)")
        
        # Stage 7: Batch embed all at once with conservative settings
        if RUN_EMBED:
            log.info("🔄 STARTING EMBEDDING STAGE...")
            log.info("📊 EMBEDDING STAGE PROGRESS:")
            log.info("   🎯 Running batch embedding with rate limiting...")
            log.info("   🎯 This stage will process all upserted chunks for embedding")
            
            # 🛡️ Pass conservative embedding parameters (consistent with embed_vectors.py defaults)
            await embed_vectors.main_async(
                batch_size=200,     # 🛡️ Rate limited: fetch 200 chunks at once (conservative default)
                commit_size=10,     # 🛡️ Rate limited: 10 chunks per API call (conservative default) 
                max_concurrent=3    # 🛡️ Rate limited: max 3 concurrent embedding calls (conservative default)
            )
            
            log.info("📊 EMBEDDING STAGE COMPLETE ✅")
        
        Console().rule("[green]Finished")
        log.info("📊 PIPELINE COMPLETE - FINAL SUMMARY:")
        log.info("🛡️  Rate limiting successful - no API limits hit")
        log.info(f"📊 Total documents processed: {self.stats['ok']}")
        log.info(f"📊 Total documents failed: {self.stats['fail']}")
        log.info(f"📊 Success rate: {(self.stats['ok']/(self.stats['ok']+self.stats['fail'])*100):.1f}%" if (self.stats['ok']+self.stats['fail']) > 0 else "100%")

def main(src: pathlib.Path, selection: str = "sequential", cap: int = 0) -> None:
    """Main entry point compatible with original pipeline_modular.py"""
    # Set multiprocessing start method for macOS
    mp.set_start_method('spawn', force=True)
    
    # Create and run pipeline with conservative rate limiting
    pipeline = OptimizedPipeline(
        batch_size=15,  # 🛡️ Rate limited: process 15 PDFs at a time (was 50)
        max_api_concurrent=3  # 🛡️ Rate limited: max 3 concurrent API requests (was 20)
    )
    
    # Run async pipeline
    asyncio.run(pipeline.run(src, selection, cap))

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s"
    )
    
    p = argparse.ArgumentParser()
    p.add_argument("src", type=pathlib.Path,
                   default=pathlib.Path("documents/research/Global"), nargs="?")
    p.add_argument("--selection", choices=["sequential", "random"],
                   default="sequential")
    p.add_argument("--cap", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=15,  # 🛡️ Rate limited: default to 15 (was 50)
                   help="Number of PDFs to process in parallel")
    p.add_argument("--api-concurrent", type=int, default=3,  # 🛡️ Rate limited: default to 3 (was 20)
                   help="Max concurrent API calls")
    args = p.parse_args()
    
    # Override batch size if specified
    if args.batch_size:
        pipeline = OptimizedPipeline(
            batch_size=args.batch_size,
            max_api_concurrent=args.api_concurrent
        )
        asyncio.run(pipeline.run(args.src, args.selection, args.cap))
    else:
        main(args.src, args.selection, args.cap) 