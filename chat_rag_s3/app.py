import argparse
import asyncio
import logging
import os
import signal
import sys
from aiohttp import web
import websockets
import aiobotocore

QUEUE_SIZE = int(os.environ.get('QUEUE_SIZE', 100))
NUM_WORKERS = int(os.environ.get('NUM_WORKERS', 10))

message_queue = asyncio.Queue(maxsize=QUEUE_SIZE)
shutdown_event = asyncio.Event()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

async def process(message):
    # Simulate processing time (replace with real logic later)
    await asyncio.sleep(3)
    logging.info(f"Processed: {message}")

async def websocket_client(uri):
    logging.info(f"WebSocket client connecting to {uri}")
    try:
        async with websockets.connect(uri) as websocket:
            async for message in websocket:
                try:
                    await message_queue.put(message)
                except asyncio.QueueFull:
                    logging.warning("Queue is full! Dropping WebSocket message.")
    except Exception as e:
        logging.exception("WebSocket client error")
        shutdown_event.set()

async def sqs_poller(queue_url, region_name):
    session = aiobotocore.get_session()
    async with session.create_client('sqs', region_name=region_name) as client:
        while not shutdown_event.is_set():
            try:
                response = await client.receive_message(
                    QueueUrl=queue_url,
                    MaxNumberOfMessages=10,
                    WaitTimeSeconds=20
                )
                for msg in response.get('Messages', []):
                    try:
                        await message_queue.put(msg['Body'])
                    except asyncio.QueueFull:
                        logging.warning("Queue is full! Dropping SQS message.")
                    # Delete message after processing
                    await client.delete_message(
                        QueueUrl=queue_url,
                        ReceiptHandle=msg['ReceiptHandle']
                    )
            except Exception as e:
                logging.exception("SQS poller error")
                await asyncio.sleep(5)  # Backoff on error

async def worker():
    while not shutdown_event.is_set():
        try:
            message = await asyncio.wait_for(message_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue
        try:
            await process(message)
        except Exception as e:
            logging.exception("Worker error during processing")
        finally:
            message_queue.task_done()

# Health and readiness endpoints
async def handle_health(request):
    return web.Response(text="ok")

async def handle_ready(request):
    return web.Response(text="ready")

def setup_signal_handlers(loop):
    def _signal_handler():
        logging.info("Shutdown signal received.")
        shutdown_event.set()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

async def start_http_server():
    app = web.Application()
    app.router.add_get('/healthz', handle_health)
    app.router.add_get('/readyz', handle_ready)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8080)
    await site.start()
    logging.info("Health endpoints running on :8080 (/healthz, /readyz)")
    await shutdown_event.wait()
    await runner.cleanup()

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--channel', choices=['websocket', 'sqs'], required=True, help='Channel to use for incoming data')
    parser.add_argument('--ws-uri', default=os.environ.get('WS_URI'), help='WebSocket server URI (required if channel is websocket)')
    parser.add_argument('--sqs-queue-url', help='SQS queue URL (required if channel is sqs)')
    parser.add_argument('--sqs-region', default=os.environ.get('SQS_REGION', 'us-east-1'), help='AWS region for SQS (default: us-east-1)')
    parser.add_argument('--queue-size', type=int, default=QUEUE_SIZE, help='Max message queue size')
    parser.add_argument('--num-workers', type=int, default=NUM_WORKERS, help='Number of worker tasks')
    args = parser.parse_args()

    global QUEUE_SIZE, NUM_WORKERS
    QUEUE_SIZE = args.queue_size
    NUM_WORKERS = args.num_workers

    # Start worker tasks
    workers = [asyncio.create_task(worker()) for _ in range(NUM_WORKERS)]
    # Start health endpoints
    http_server = asyncio.create_task(start_http_server())

    # Start selected channel
    if args.channel == 'websocket':
        if not args.ws_uri:
            raise ValueError('You must provide --ws-uri or set WS_URI when using the websocket channel')
        await websocket_client(args.ws_uri)
    elif args.channel == 'sqs':
        if not args.sqs_queue_url:
            raise ValueError('You must provide --sqs-queue-url when using the sqs channel')
        await sqs_poller(args.sqs_queue_url, args.sqs_region)

    # Wait for shutdown
    await shutdown_event.wait()
    logging.info("Waiting for workers to finish...")
    await message_queue.join()
    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)
    logging.info("Shutdown complete.")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    setup_signal_handlers(loop)
    try:
        loop.run_until_complete(main())
    except Exception:
        logging.exception("Fatal error in main loop")
        sys.exit(1)
