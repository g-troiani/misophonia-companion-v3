"""
Hardware acceleration utilities for the pipeline.
Provides Apple Silicon optimization when available, with CPU fallback.
"""
import os
import platform
import multiprocessing as mp
from typing import Optional, Callable, Any, List
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import asyncio
from functools import partial

log = logging.getLogger(__name__)

class HardwareAccelerator:
    """Detects and manages hardware acceleration capabilities."""
    
    def __init__(self):
        self.is_apple_silicon = self._detect_apple_silicon()
        self.cpu_count = mp.cpu_count()
        self.optimal_workers = self._calculate_optimal_workers()
        
        # Set environment for better performance on macOS
        if self.is_apple_silicon:
            os.environ['OPENBLAS_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            os.environ['OMP_NUM_THREADS'] = '1'
        
        log.info(f"Hardware: {'Apple Silicon' if self.is_apple_silicon else platform.processor()}")
        log.info(f"CPU cores: {self.cpu_count}, Optimal workers: {self.optimal_workers}")
    
    def _detect_apple_silicon(self) -> bool:
        """Detect if running on Apple Silicon."""
        if platform.system() != 'Darwin':
            return False
        try:
            import subprocess
            result = subprocess.run(['sysctl', '-n', 'hw.optional.arm64'], 
                                  capture_output=True, text=True)
            return result.stdout.strip() == '1'
        except:
            return False
    
    def _calculate_optimal_workers(self) -> int:
        """Calculate optimal number of workers based on hardware."""
        if self.is_apple_silicon:
            # Apple Silicon has efficiency and performance cores
            # Use 75% of cores to leave room for system
            return max(1, int(self.cpu_count * 0.75))
        else:
            # Traditional CPU - use all but one core
            return max(1, self.cpu_count - 1)
    
    def get_process_pool(self, max_workers: Optional[int] = None) -> ProcessPoolExecutor:
        """Get optimized process pool executor."""
        workers = max_workers or self.optimal_workers
        return ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context('spawn')  # Required for macOS
        )
    
    def get_thread_pool(self, max_workers: Optional[int] = None) -> ThreadPoolExecutor:
        """Get optimized thread pool executor for I/O operations."""
        workers = max_workers or min(32, self.cpu_count * 4)
        return ThreadPoolExecutor(max_workers=workers)

# Global instance
hardware = HardwareAccelerator()

async def run_cpu_bound_concurrent(func: Callable, items: List[Any], 
                                 max_workers: Optional[int] = None,
                                 desc: str = "Processing") -> List[Any]:
    """Run CPU-bound function concurrently on multiple items."""
    loop = asyncio.get_event_loop()
    with hardware.get_process_pool(max_workers) as executor:
        futures = [loop.run_in_executor(executor, func, item) for item in items]
        
        from tqdm.asyncio import tqdm_asyncio
        results = await tqdm_asyncio.gather(*futures, desc=desc)
        return results

async def run_io_bound_concurrent(func: Callable, items: List[Any],
                                max_workers: Optional[int] = None,
                                desc: str = "Processing") -> List[Any]:
    """Run I/O-bound function concurrently on multiple items."""
    loop = asyncio.get_event_loop()
    with hardware.get_thread_pool(max_workers) as executor:
        futures = [loop.run_in_executor(executor, func, item) for item in items]
        
        from tqdm.asyncio import tqdm_asyncio
        results = await tqdm_asyncio.gather(*futures, desc=desc)
        return results 