"""
chat_processor.py

A one-file implementation of the "Ingest → Enrich → Store" pipeline:
- Ingests messages from a JSON file or stream.
- Annotates with business terms from a glossary.
- Extracts metadata (tickers, sectors, brokers, sentiment) via OpenAI.
- Generates embeddings via OpenAI.
- Stores messages and metadata in Postgres.
- Stores embeddings in a FAISS index, with shard-per-day strategy.
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
import re

import openai
import psycopg2
import faiss

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment / config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PG_DSN = os.getenv("PG_DSN")  # e.g., "host=localhost dbname=chat user=postgres password=secret"
GLOSSARY_PATH = os.getenv("GLOSSARY_PATH", "glossary.json")
FAISS_DIR = os.getenv("FAISS_DIR", "./faiss_indices")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 50))

openai.api_key = OPENAI_API_KEY

class GlossaryAnnotator:
    def __init__(self, glossary_path):
        with open(glossary_path, 'r') as f:
            self.terms = json.load(f)  # {"IOI": "Indication of Interest", ...}
        # Precompile regex
        self.pattern = re.compile(r'\b(' + '|'.join(map(re.escape, self.terms.keys())) + r')\b', re.IGNORECASE)

    def annotate(self, text):
        hits = []
        for m in self.pattern.finditer(text):
            term = m.group(0).upper()
            defn = self.terms.get(term, "")
            hits.append({"term": term, "definition": defn})
        return hits

class OpenAIExtractor:
    def __init__(self):
        pass

    def extract_meta(self, text):
        prompt = (
            "Extract JSON with keys 'tickers', 'sectors', 'brokers', and 'sentiment' "
            "from the following message. Return only valid JSON.

"
            f"Message: "{text}""
        )
        resp = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150,
            temperature=0
        )
        try:
            return json.loads(resp.choices[0].text.strip())
        except Exception as e:
            logger.error(f"Failed to parse metadata JSON: {e}")
            return {"tickers": [], "sectors": [], "brokers": [], "sentiment": "neutral"}

    def embed(self, text):
        resp = openai.Embedding.create(
            engine="text-embedding-ada-002",
            input=text
        )
        return resp['data'][0]['embedding']

class PostgresStorage:
    def __init__(self, dsn):
        self.conn = psycopg2.connect(dsn)
        self._ensure_tables()

    def _ensure_tables(self):
        with self.conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                chat_id TEXT PRIMARY KEY,
                ts TIMESTAMP,
                user_id TEXT,
                raw_text TEXT,
                enriched JSONB
            );
            CREATE TABLE IF NOT EXISTS message_terms (
                chat_id TEXT REFERENCES messages,
                term TEXT,
                definition TEXT,
                PRIMARY KEY(chat_id, term)
            );
            CREATE TABLE IF NOT EXISTS message_meta (
                chat_id TEXT REFERENCES messages,
                entity_type TEXT,
                value TEXT
            );
            CREATE TABLE IF NOT EXISTS vector_index_map (
                chat_id TEXT PRIMARY KEY,
                shard_date DATE,
                vector_idx INTEGER
            );
            """)
            self.conn.commit()

    def store_message(self, msg):
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO messages(chat_id, ts, user_id, raw_text, enriched) VALUES (%s,%s,%s,%s,%s) "
                "ON CONFLICT (chat_id) DO NOTHING",
                (msg['chat_id'], msg['ts'], msg['user'], msg['text'], json.dumps(msg))
            )
            for term in msg.get('terms', []):
                cur.execute(
                    "INSERT INTO message_terms(chat_id,term,definition) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                    (msg['chat_id'], term['term'], term['definition'])
                )
            meta = msg.get('meta', {})
            for etype in ['tickers', 'sectors', 'brokers', 'sentiment']:
                vals = meta.get(etype, [])
                if isinstance(vals, list):
                    for v in vals:
                        cur.execute(
                            "INSERT INTO message_meta(chat_id,entity_type,value) VALUES (%s,%s,%s)",
                            (msg['chat_id'], etype, v)
                        )
                else:  # sentiment as string
                    cur.execute(
                        "INSERT INTO message_meta(chat_id,entity_type,value) VALUES (%s,%s,%s)",
                        (msg['chat_id'], etype, vals)
                    )
            self.conn.commit()

    def map_vector(self, chat_id, shard_date, idx):
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO vector_index_map(chat_id,shard_date,vector_idx) VALUES (%s,%s,%s) ON CONFLICT(chat_id) DO UPDATE SET shard_date=EXCLUDED.shard_date, vector_idx=EXCLUDED.vector_idx",
                (chat_id, shard_date, idx)
            )
            self.conn.commit()

class FaissIndex:
    def __init__(self, shard_date, dim):
        self.shard_path = os.path.join(FAISS_DIR, f"{shard_date}.index")
        self.dim = dim
        os.makedirs(FAISS_DIR, exist_ok=True)
        if os.path.exists(self.shard_path):
            self.index = faiss.read_index(self.shard_path)
        else:
            self.index = faiss.IndexFlatIP(dim)  # inner-product (requires normalized vectors)

    def add(self, vecs):
        self.index.add(vecs)

    def save(self):
        faiss.write_index(self.index, self.shard_path)

class ChatProcessor:
    def __init__(self, args):
        self.annotator = GlossaryAnnotator(args.glossary)
        self.extractor = OpenAIExtractor()
        self.storage = PostgresStorage(args.dsn)
        self.batch = []
        self.args = args

    def process_batch(self):
        shard_date = datetime.utcnow().date().isoformat()
        # Prepare FAISS
        dim = len(self.extractor.embed("test"))
        faiss_idx = FaissIndex(shard_date, dim)
        for msg in self.batch:
            # Annotate glossary terms
            msg['terms'] = self.annotator.annotate(msg['text'])
            # Extract meta
            msg['meta'] = self.extractor.extract_meta(msg['text'])
            # Build embedding text
            ctx = f"[RAW] {msg['text']}\n[TERMS] {json.dumps(msg['terms'])}\n[META] {json.dumps(msg['meta'])}"
            vec = self.extractor.embed(ctx)
            # Store in Postgres
            self.storage.store_message(msg)
            # Add to FAISS
            faiss_idx.add([[v / (sum(x*x for x in vec)**0.5) for v in vec]])  # normalize
            self.storage.map_vector(msg['chat_id'], shard_date, faiss_idx.index.ntotal - 1)
        # Persist FAISS shard
        faiss_idx.save()
        self.batch = []

    def run(self):
        input_source = open(self.args.input) if self.args.input != "-" else sys.stdin
        for line in input_source:
            if not line.strip():
                continue
            raw = json.loads(line)
            msg = {
                "chat_id": raw["chat_id"],
                "ts": raw["timestamp"],
                "user": raw.get("user", ""),
                "text": raw.get("text", "")
            }
            self.batch.append(msg)
            if len(self.batch) >= self.args.batch_size:
                self.process_batch()
        # Process leftovers
        if self.batch:
            self.process_batch()

def main():
    parser = argparse.ArgumentParser(description="Process Bloomberg chat JSON feed")
    parser.add_argument("--input", type=str, default="-", help="Path to JSON feed file or '-' for stdin")
    parser.add_argument("--glossary", type=str, default=GLOSSARY_PATH, help="Path to glossary JSON")
    parser.add_argument("--dsn", type=str, default=PG_DSN, help="Postgres DSN")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for processing")
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        logger.error("Please set OPENAI_API_KEY environment variable")
        sys.exit(1)
    if not args.dsn:
        logger.error("Please set PG_DSN environment variable or pass --dsn")
        sys.exit(1)

    processor = ChatProcessor(args)
    processor.run()

if __name__ == "__main__":
    main()
