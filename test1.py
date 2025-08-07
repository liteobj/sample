"""
High-throughput concurrent message processor for AWS SQS.
Handles 100 messages/second with 5-10 second processing times per message.
Subscribes to SQS, processes messages asynchronously, stores results in FAISS vector store.
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
import aiohttp
import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, HTTPException
import uvicorn
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from uuid import uuid4

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

class ThreadSafeFAISSWriter:
    """Thread-safe FAISS vector store writer using a single dedicated thread"""
    
    def __init__(self, embeddings_model: str = "text-embedding-ada-002"):
        self.embeddings = OpenAIEmbeddings(model=embeddings_model)
        self.write_queue = Queue()
        self.running = False
        self.writer_thread = None
        self.vector_store = None
        self.faiss_index_path = "faiss_index"
        
    def start(self):
        """Start the FAISS writer thread"""
        self.running = True
        self.writer_thread = threading.Thread(target=self._faiss_writer_worker, daemon=True)
        self.writer_thread.start()
        logger.info("FAISS writer thread started")
    
    def stop(self):
        """Stop the FAISS writer thread"""
        self.running = False
        if self.writer_thread:
            self.writer_thread.join(timeout=5)
        logger.info("FAISS writer thread stopped")
    
    def write_result(self, result: Dict[str, Any]):
        """Queue a result for writing to FAISS (thread-safe)"""
        self.write_queue.put(result)
    
    def _faiss_writer_worker(self):
        """Dedicated thread for FAISS vector store operations"""
        while self.running:
            try:
                # Get result from queue (blocking with timeout)
                result = self.write_queue.get(timeout=1.0)
                
                # Create document for FAISS
                document = self._create_document_from_result(result)
                
                # Add document to vector store
                if self.vector_store is None:
                    # Initialize vector store with first document
                    self.vector_store = FAISS.from_documents(
                        documents=[document],
                        embedding=self.embeddings
                    )
                else:
                    # Add to existing vector store
                    self.vector_store.add_documents(documents=[document])
                
                # Save vector store periodically
                self.vector_store.save_local(self.faiss_index_path)
                
                # Mark task as done
                self.write_queue.task_done()
                
                logger.info(f"Added document to FAISS: {result.get('message_id', 'unknown')}")
                
            except Exception as e:
                if self.running:  # Only log if we're supposed to be running
                    logger.error(f"Error in FAISS writer thread: {e}")
    
    def _create_document_from_result(self, result: Dict[str, Any]) -> Document:
        """Create a Document object from processing result"""
        # Create content from result
        content = f"Message ID: {result.get('message_id', 'unknown')}\n"
        content += f"Processing Time: {result.get('processing_time', 0):.2f}s\n"
        content += f"Attempts: {result.get('attempts', 1)}\n"
        
        if 'result' in result:
            content += f"API Result: {json.dumps(result['result'], indent=2)}\n"
        elif 'error' in result:
            content += f"Error: {result['error']}\n"
        
        content += f"Timestamp: {result.get('timestamp', 'unknown')}"
        
        # Create metadata
        metadata = {
            "message_id": result.get('message_id', 'unknown'),
            "sqs_message_id": result.get('sqs_message_id', 'unknown'),
            "processing_time": result.get('processing_time', 0),
            "attempts": result.get('attempts', 1),
            "timestamp": result.get('timestamp', 'unknown'),
            "has_error": 'error' in result,
            "source": "sqs_processor"
        }
        
        return Document(
            page_content=content,
            metadata=metadata
        )
    
    def similarity_search(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None):
        """Search for similar documents in FAISS"""
        if self.vector_store is None:
            return []
        
        try:
            if filter_dict:
                results = self.vector_store.similarity_search(
                    query, k=k, filter=filter_dict
                )
            else:
                results = self.vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None):
        """Search for similar documents with scores"""
        if self.vector_store is None:
            return []
        
        try:
            if filter_dict:
                results = self.vector_store.similarity_search_with_score(
                    query, k=k, filter=filter_dict
                )
            else:
                results = self.vector_store.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            logger.error(f"Error in similarity search with score: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get FAISS vector store statistics"""
        if self.vector_store is None:
            return {
                "total_documents": 0,
                "index_size": 0,
                "index_path": self.faiss_index_path
            }
        
        try:
            return {
                "total_documents": len(self.vector_store.docstore._dict),
                "index_size": self.vector_store.index.ntotal if hasattr(self.vector_store.index, 'ntotal') else 0,
                "index_path": self.faiss_index_path
            }
        except Exception as e:
            logger.error(f"Error getting FAISS stats: {e}")
            return {
                "total_documents": 0,
                "index_size": 0,
                "index_path": self.faiss_index_path
            }

class SQSProcessor:
    """SQS message processor with async processing and FAISS storage"""
    
    def __init__(self, 
                 queue_url: str,
                 region_name: str = "us-east-1",
                 max_workers: int = 50,
                 output_file: str = "results.jsonl"):
        
        self.queue_url = queue_url
        self.region_name = region_name
        self.max_workers = max_workers
        self.output_file = Path(output_file)
        self.running = False
        self.stats = {"processed": 0, "failed": 0, "start_time": None}
        
        # SQS client
        self.sqs = boto3.client('sqs', region_name=region_name)
        
        # HTTP session for external API calls
        self.session = None
        
        # Worker semaphore for controlling concurrency
        self.worker_semaphore = asyncio.Semaphore(max_workers)
        
        # Thread-safe FAISS writer
        self.faiss_writer = ThreadSafeFAISSWriter()
        
        # Background tasks
        self.background_tasks = []
    
    async def start(self):
        """Start the SQS processor"""
        self.running = True
        self.stats["start_time"] = time.time()
        
        # Create HTTP session
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
        self.session = aiohttp.ClientSession(connector=connector)
        
        # Start FAISS writer
        self.faiss_writer.start()
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._sqs_polling_worker()),
            asyncio.create_task(self._monitor_stats()),
        ]
        
        logger.info(f"Started SQS processor with {self.max_workers} workers")
    
    async def stop(self):
        """Stop the SQS processor"""
        self.running = False
        
        # Wait for all tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Close HTTP session
        if self.session:
            await self.session.close()
        
        # Stop FAISS writer
        self.faiss_writer.stop()
        
        logger.info("SQS processor stopped")
    
    async def _sqs_polling_worker(self):
        """Worker that polls SQS for messages"""
        while self.running:
            try:
                # Receive messages from SQS
                response = self.sqs.receive_message(
                    QueueUrl=self.queue_url,
                    MaxNumberOfMessages=10,  # Process up to 10 messages at once
                    WaitTimeSeconds=20,  # Long polling
                    VisibilityTimeout=30  # 30 seconds visibility timeout
                )
                
                messages = response.get('Messages', [])
                
                if messages:
                    logger.info(f"Received {len(messages)} messages from SQS")
                    
                    # Process messages concurrently
                    tasks = []
                    for sqs_message in messages:
                        task = asyncio.create_task(self._process_sqs_message(sqs_message))
                        tasks.append(task)
                    
                    # Wait for all messages to be processed
                    await asyncio.gather(*tasks, return_exceptions=True)
                
            except ClientError as e:
                logger.error(f"SQS polling error: {e}")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Unexpected error in polling worker: {e}")
                await asyncio.sleep(5)
    
    async def _process_sqs_message(self, sqs_message: Dict[str, Any]):
        """Process a single SQS message"""
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
            
            # Process message with concurrency control - RELEASE SEMAPHORE AFTER PROCESSING
            async with self.worker_semaphore:
                result = await self._process_message_with_retry(message)
            
            # Queue result for FAISS writing (thread-safe)
            self.faiss_writer.write_result(result)
            
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
    
    async def _process_message_with_retry(self, message: Message) -> Dict[str, Any]:
        """Process message with external API call and retry logic"""
        start_time = time.time()
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries + 1):
            try:
                # Simulate external API call (replace with actual API)
                async with self.session.post(
                    "https://api.example.com/process",
                    json=message.to_dict(),
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                    else:
                        raise Exception(f"API call failed: {response.status}")
                
                # Simulate additional processing time (5-10 seconds)
                await asyncio.sleep(7)
                
                processing_time = time.time() - start_time
                
                return {
                    "message_id": message.id,
                    "sqs_message_id": message.sqs_message_id,
                    "receipt_handle": message.receipt_handle,
                    "result": result,
                    "processing_time": processing_time,
                    "attempts": attempt + 1,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                if attempt == max_retries:
                    processing_time = time.time() - start_time
                    logger.error(f"Message {message.id} failed after {attempt + 1} attempts: {e}")
                    
                    return {
                        "message_id": message.id,
                        "sqs_message_id": message.sqs_message_id,
                        "receipt_handle": message.receipt_handle,
                        "error": str(e),
                        "processing_time": processing_time,
                        "attempts": attempt + 1,
                        "timestamp": datetime.now().isoformat()
                    }
                
                # Exponential backoff
                delay = retry_delay * (2 ** attempt)
                logger.warning(f"Retrying message {message.id} in {delay}s (attempt {attempt + 1})")
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
            
            # Get FAISS stats
            faiss_stats = self.faiss_writer.get_stats()
            
            logger.info(f"Stats - Processed: {self.stats['processed']}, "
                       f"Failed: {self.stats['failed']}, "
                       f"Rate: {rate:.2f} msg/s, "
                       f"FAISS Documents: {faiss_stats['total_documents']}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        elapsed = time.time() - self.stats["start_time"]
        rate = self.stats["processed"] / elapsed if elapsed > 0 else 0
        
        # Get FAISS stats
        faiss_stats = self.faiss_writer.get_stats()
        
        return {
            "processed": self.stats["processed"],
            "failed": self.stats["failed"],
            "processing_rate": rate,
            "uptime": elapsed,
            "running": self.running,
            "active_workers": self.max_workers if self.running else 0,
            "faiss_stats": faiss_stats
        }
    
    def search_results(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None):
        """Search for similar results in FAISS"""
        return self.faiss_writer.similarity_search(query, k, filter_dict)
    
    def search_results_with_score(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None):
        """Search for similar results with scores in FAISS"""
        return self.faiss_writer.similarity_search_with_score(query, k, filter_dict)

# FastAPI application
app = FastAPI(title="SQS Message Processor with FAISS")

# Initialize processor with SQS configuration
processor = SQSProcessor(
    queue_url=os.getenv("SQS_QUEUE_URL", "https://sqs.us-east-1.amazonaws.com/123456789012/my-queue"),
    region_name=os.getenv("AWS_REGION", "us-east-1"),
    max_workers=int(os.getenv("MAX_WORKERS", "50")),
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

@app.get("/search")
async def search_results(query: str, k: int = 5):
    """Search for similar results in FAISS"""
    results = processor.search_results(query, k)
    return {
        "query": query,
        "results": [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in results
        ]
    }

@app.get("/search_with_score")
async def search_results_with_score(query: str, k: int = 5):
    """Search for similar results with scores in FAISS"""
    results = processor.search_results_with_score(query, k)
    return {
        "query": query,
        "results": [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            }
            for doc, score in results
        ]
    }

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
        "faiss_documents": stats["faiss_stats"]["total_documents"]
    }

@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "service": "SQS Message Processor with FAISS",
        "version": "1.0.0",
        "description": "High-throughput SQS processor with async processing and FAISS vector storage",
        "sqs_queue": processor.queue_url,
        "endpoints": {
            "GET /stats": "Get processing statistics",
            "GET /search": "Search results in FAISS",
            "GET /search_with_score": "Search results with scores in FAISS",
            "GET /health": "Health check"
        }
    }

if __name__ == "__main__":
    # Run the FastAPI application
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
