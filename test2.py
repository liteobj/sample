"""
Multi-process SQS message processor for high CPU utilization.
Bypasses GIL for CPU-intensive processing while maintaining async I/O.
"""

import asyncio
import json
import logging
import threading
import time
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Any, Dict, Optional
from multiprocessing import Pool, Manager
import aiohttp
import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, HTTPException
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Message:
    """Message data structure for SQS messages"""
    id: str
    content: str
    timestamp: float
    receipt_handle: str
    sqs_message_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def cpu_intensive_processing(message_data: Dict[str, Any]) -> Dict[str, Any]:
    """CPU-intensive processing in separate process (bypasses GIL)"""
    import time
    import hashlib
    
    # Simulate CPU-intensive work
    start_time = time.time()
    
    # Heavy computation (this runs in separate process)
    content = message_data['content']
    for i in range(1000000):  # CPU-intensive loop
        hashlib.sha256(content.encode()).hexdigest()
    
    processing_time = time.time() - start_time
    
    return {
        "message_id": message_data['id'],
        "processed_content": content.upper(),
        "hash_result": hashlib.sha256(content.encode()).hexdigest(),
        "cpu_processing_time": processing_time,
        "timestamp": datetime.now().isoformat()
    }

class MultiProcessFileWriter:
    """Thread-safe file writer for multi-process environment"""
    
    def __init__(self, output_file: str):
        self.output_file = Path(output_file)
        self.write_queue = Queue()
        self.running = False
        self.writer_thread = None
        
    def start(self):
        """Start the file writer thread"""
        self.running = True
        self.writer_thread = threading.Thread(target=self._file_writer_worker, daemon=True)
        self.writer_thread.start()
        logger.info("Multi-process file writer thread started")
    
    def stop(self):
        """Stop the file writer thread"""
        self.running = False
        if self.writer_thread:
            self.writer_thread.join(timeout=5)
        logger.info("Multi-process file writer thread stopped")
    
    def write_result(self, result: Dict[str, Any]):
        """Queue a result for writing (thread-safe)"""
        self.write_queue.put(result)
    
    def _file_writer_worker(self):
        """Dedicated thread for file writing"""
        while self.running:
            try:
                result = self.write_queue.get(timeout=1.0)
                
                with open(self.output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result) + '\n')
                
                self.write_queue.task_done()
                
            except Exception as e:
                if self.running:
                    logger.error(f"Error in file writer thread: {e}")

class MultiProcessSQSProcessor:
    """Multi-process SQS processor with CPU-intensive processing"""
    
    def __init__(self, 
                 queue_url: str,
                 region_name: str = "us-east-1",
                 max_workers: int = 50,
                 max_processes: int = 4,
                 output_file: str = "results.jsonl"):
        
        self.queue_url = queue_url
        self.region_name = region_name
        self.max_workers = max_workers
        self.max_processes = max_processes
        self.output_file = Path(output_file)
        self.running = False
        self.stats = {"processed": 0, "failed": 0, "start_time": None}
        
        # SQS client
        self.sqs = boto3.client('sqs', region_name=region_name)
        
        # HTTP session for external API calls
        self.session = None
        
        # Worker semaphore for controlling concurrency
        self.worker_semaphore = asyncio.Semaphore(max_workers)
        
        # Process pool for CPU-intensive work
        self.process_pool = Pool(processes=max_processes)
        
        # Thread-safe file writer
        self.file_writer = MultiProcessFileWriter(output_file)
        
        # Background tasks
        self.background_tasks = []
    
    async def start(self):
        """Start the multi-process SQS processor"""
        self.running = True
        self.stats["start_time"] = time.time()
        
        # Create HTTP session
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
        self.session = aiohttp.ClientSession(connector=connector)
        
        # Start file writer
        self.file_writer.start()
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._sqs_polling_worker()),
            asyncio.create_task(self._monitor_stats()),
        ]
        
        logger.info(f"Started multi-process SQS processor with {self.max_workers} workers and {self.max_processes} processes")
    
    async def stop(self):
        """Stop the multi-process SQS processor"""
        self.running = False
        
        # Wait for all tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Close HTTP session
        if self.session:
            await self.session.close()
        
        # Stop file writer
        self.file_writer.stop()
        
        # Close process pool
        self.process_pool.close()
        self.process_pool.join()
        
        logger.info("Multi-process SQS processor stopped")
    
    async def _sqs_polling_worker(self):
        """Worker that polls SQS for messages"""
        while self.running:
            try:
                # Receive messages from SQS
                response = self.sqs.receive_message(
                    QueueUrl=self.queue_url,
                    MaxNumberOfMessages=10,
                    WaitTimeSeconds=20,
                    VisibilityTimeout=30
                )
                
                messages = response.get('Messages', [])
                
                if messages:
                    logger.info(f"Received {len(messages)} messages from SQS")
                    
                    # Process messages concurrently
                    tasks = []
                    for sqs_message in messages:
                        task = asyncio.create_task(self._process_sqs_message(sqs_message))
                        tasks.append(task)
                    
                    await asyncio.gather(*tasks, return_exceptions=True)
                
            except ClientError as e:
                logger.error(f"SQS polling error: {e}")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Unexpected error in polling worker: {e}")
                await asyncio.sleep(5)
    
    async def _process_sqs_message(self, sqs_message: Dict[str, Any]):
        """Process a single SQS message with multi-process CPU work"""
        try:
            # Extract message data
            message_id = sqs_message['MessageId']
            receipt_handle = sqs_message['ReceiptHandle']
            body = sqs_message['Body']
            
            # Create message object
            message = Message(
                id=message_id,
                content=body,
                timestamp=time.time(),
                receipt_handle=receipt_handle,
                sqs_message_id=message_id
            )
            
            # Process message with concurrency control
            async with self.worker_semaphore:
                # Step 1: Async I/O (API call)
                api_result = await self._process_api_call(message)
                
                # Step 2: CPU-intensive work in separate process
                cpu_result = await asyncio.get_event_loop().run_in_executor(
                    None,  # Use default executor
                    self.process_pool.apply,
                    cpu_intensive_processing,
                    (message.to_dict(),)
                )
                
                # Combine results
                result = {
                    "message_id": message.id,
                    "sqs_message_id": message.sqs_message_id,
                    "receipt_handle": message.receipt_handle,
                    "api_result": api_result,
                    "cpu_result": cpu_result,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Queue result for file writing
            self.file_writer.write_result(result)
            
            # Delete message from SQS if successful
            if "error" not in result:
                self._delete_sqs_message(receipt_handle)
                self.stats["processed"] += 1
            else:
                self.stats["failed"] += 1
            
            logger.info(f"Processed message {message_id} - Success: {'error' not in result}")
            
        except Exception as e:
            logger.error(f"Error processing SQS message {sqs_message.get('MessageId', 'unknown')}: {e}")
            self.stats["failed"] += 1
    
    async def _process_api_call(self, message: Message) -> Dict[str, Any]:
        """Process external API call (async I/O)"""
        start_time = time.time()
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries + 1):
            try:
                # External API call
                async with self.session.post(
                    "https://api.example.com/process",
                    json=message.to_dict(),
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                    else:
                        raise Exception(f"API call failed: {response.status}")
                
                processing_time = time.time() - start_time
                
                return {
                    "api_response": result,
                    "api_processing_time": processing_time,
                    "api_attempts": attempt + 1
                }
                
            except Exception as e:
                if attempt == max_retries:
                    processing_time = time.time() - start_time
                    logger.error(f"API call for message {message.id} failed after {attempt + 1} attempts: {e}")
                    
                    return {
                        "error": str(e),
                        "api_processing_time": processing_time,
                        "api_attempts": attempt + 1
                    }
                
                delay = retry_delay * (2 ** attempt)
                logger.warning(f"Retrying API call for message {message.id} in {delay}s (attempt {attempt + 1})")
                await asyncio.sleep(delay)
    
    def _delete_sqs_message(self, receipt_handle: str):
        """Delete message from SQS queue"""
        try:
            self.sqs.delete_message(
                QueueUrl=self.queue_url,
                ReceiptHandle=receipt_handle
            )
        except ClientError as e:
            logger.error(f"Error deleting SQS message: {e}")
    
    async def _monitor_stats(self):
        """Monitor and log statistics"""
        while self.running:
            await asyncio.sleep(30)
            
            elapsed = time.time() - self.stats["start_time"]
            rate = self.stats["processed"] / elapsed if elapsed > 0 else 0
            
            logger.info(f"Multi-process Stats - Processed: {self.stats['processed']}, "
                       f"Failed: {self.stats['failed']}, "
                       f"Rate: {rate:.2f} msg/s")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        elapsed = time.time() - self.stats["start_time"]
        rate = self.stats["processed"] / elapsed if elapsed > 0 else 0
        
        return {
            "processed": self.stats["processed"],
            "failed": self.stats["failed"],
            "processing_rate": rate,
            "uptime": elapsed,
            "running": self.running,
            "active_workers": self.max_workers if self.running else 0,
            "active_processes": self.max_processes
        }

# FastAPI application
app = FastAPI(title="Multi-Process SQS Message Processor")

# Initialize processor with multi-process configuration
processor = MultiProcessSQSProcessor(
    queue_url=os.getenv("SQS_QUEUE_URL", "https://sqs.us-east-1.amazonaws.com/123456789012/my-queue"),
    region_name=os.getenv("AWS_REGION", "us-east-1"),
    max_workers=int(os.getenv("MAX_WORKERS", "50")),
    max_processes=int(os.getenv("MAX_PROCESSES", "4")),
    output_file=os.getenv("OUTPUT_FILE", "results.jsonl")
)

@app.on_event("startup")
async def startup_event():
    await processor.start()

@app.on_event("shutdown")
async def shutdown_event():
    await processor.stop()

@app.get("/stats")
async def get_stats():
    """Get current processing statistics"""
    return processor.get_stats()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    stats = processor.get_stats()
    
    is_healthy = (
        processor.running and
        stats["active_workers"] > 0
    )
    
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "uptime": stats["uptime"],
        "processed": stats["processed"],
        "failed": stats["failed"],
        "processing_rate": stats["processing_rate"],
        "active_processes": stats["active_processes"]
    }

@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "service": "Multi-Process SQS Message Processor",
        "version": "1.0.0",
        "description": "High-throughput SQS processor with async I/O and multi-process CPU work",
        "sqs_queue": processor.queue_url,
        "endpoints": {
            "GET /stats": "Get processing statistics",
            "GET /health": "Health check"
        }
    }

if __name__ == "__main__":
    # Run the FastAPI application
    uvicorn.run(
        "multi_process_app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
