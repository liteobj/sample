"""
special_agent.py

Implements Service 2: Query & Notify API in one file.
- FastAPI for HTTP endpoints (/query, /subscribe, /unsubscribe, /subscriptions).
- Uses Postgres to store messages, metadata, subscriptions, and events.
- Uses FAISS for vector search.
- Uses OpenAI for embeddings and optional summarization.
"""

import os
import json
from datetime import datetime
from typing import List, Optional, Dict, Any

import openai
import psycopg2
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field
import faiss

# Configuration from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PG_DSN = os.getenv("PG_DSN")  # e.g. "host=localhost dbname=chat user=postgres password=secret"
FAISS_DIR = os.getenv("FAISS_DIR", "./faiss_indices")
EMBED_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-3.5-turbo"

# Initialize OpenAI
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")
openai.api_key = OPENAI_API_KEY

# Initialize Postgres connection
conn = psycopg2.connect(PG_DSN)
conn.autocommit = True

# Ensure required tables exist
with conn.cursor() as cur:
    cur.execute("""
    CREATE TABLE IF NOT EXISTS subscriptions (
        sub_id SERIAL PRIMARY KEY,
        user_id TEXT NOT NULL,
        callback TEXT NOT NULL,
        filter JSONB NOT NULL
    );
    """)

# Load FAISS index (single combined index)
# For demonstration, load a single index file "global.index"
index_path = os.path.join(FAISS_DIR, "global.index")
if not os.path.exists(index_path):
    raise RuntimeError(f"FAISS index not found at {index_path}")
faiss_index = faiss.read_index(index_path)

# FastAPI app
app = FastAPI(title="Special Agent API")

# ----- Pydantic models -----

class QueryIn(BaseModel):
    text: str
    start_ts: Optional[datetime] = None
    end_ts: Optional[datetime] = None
    page_size: int = Field(10, ge=1, le=100)
    page: int = Field(0, ge=0)
    summarize: bool = False
    use_vector: bool = True
    filters: Optional[Dict[str, Any]] = None  # e.g. {"tickers": ["AAPL"], "sentiment": ["positive"]}

class Hit(BaseModel):
    chat_id: str
    ts: datetime
    user: str
    text: str
    score: float

class QueryOut(BaseModel):
    query: str
    page: int
    page_size: int
    hits: List[Hit]
    next_page: Optional[int]
    answer: Optional[str] = None

class SubscribeIn(BaseModel):
    user_id: str
    callback: str
    filter: Dict[str, Any]

class SubscribeOut(BaseModel):
    sub_id: int

class Subscription(BaseModel):
    sub_id: int
    user_id: str
    callback: str
    filter: Dict[str, Any]

# ----- Utility functions -----

def embed_text(text: str):
    resp = openai.Embedding.create(engine=EMBED_MODEL, input=text)
    vec = resp['data'][0]['embedding']
    # normalize
    norm = sum(x*x for x in vec) ** 0.5
    return [v / norm for v in vec]

def fetch_messages_by_ids(chat_ids: List[str], start_ts: Optional[datetime], end_ts: Optional[datetime]):
    if not chat_ids:
        return {}
    placeholders = ','.join(['%s'] * len(chat_ids))
    sql = f"SELECT chat_id, ts, user_id, raw_text FROM messages WHERE chat_id IN ({placeholders})"
    params = chat_ids
    if start_ts:
        sql += " AND ts >= %s"
        params.append(start_ts)
    if end_ts:
        sql += " AND ts <= %s"
        params.append(end_ts)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return {row[0]: {"ts": row[1], "user": row[2], "text": row[3]} for row in cur.fetchall()}

def summarize_with_llm(query: str, context: str):
    prompt = [
        {"role": "system", "content": "You are an expert market summarizer."},
        {"role": "user", "content": f"Query: {query}\nContext:\n{context}"}
    ]
    resp = openai.ChatCompletion.create(model=LLM_MODEL, messages=prompt, max_tokens=200)
    return resp.choices[0].message.content.strip()

# ----- Auth stub -----
def authenticate(api_key: str = Depends()):
    # TODO: validate api_key or JWT
    return "user-x"

# ----- API endpoints -----

@app.post("/query", response_model=QueryOut)
def query_api(req: QueryIn, user_id: str = Depends(authenticate)):
    # 1) Determine filters
    filters = req.filters or {}

    # 2) If not using vector search and filters exist, perform metadata-only query
    if not req.use_vector and filters:
        # Example: filter by tickers
        conditions = []
        params = []
        for etype, vals in filters.items():
            if isinstance(vals, list):
                placeholders = ','.join(['%s'] * len(vals))
                conditions.append(f"chat_id IN (SELECT chat_id FROM message_meta WHERE entity_type = %s AND value IN ({placeholders}))")
                params += [etype] + vals
        if req.start_ts:
            conditions.append("ts >= %s"); params.append(req.start_ts)
        if req.end_ts:
            conditions.append("ts <= %s"); params.append(req.end_ts)
        where_clause = " AND ".join(conditions) or "TRUE"
        sql = f"SELECT chat_id, ts, user_id, raw_text FROM messages WHERE {where_clause} ORDER BY ts DESC LIMIT %s OFFSET %s"
        params += [req.page_size, req.page * req.page_size]
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        hits = [Hit(chat_id=r[0], ts=r[1], user=r[2], text=r[3], score=0.0) for r in rows]
        return QueryOut(query=req.text, page=req.page, page_size=req.page_size,
                        hits=hits,
                        next_page=(req.page + 1 if len(rows) == req.page_size else None))

    # 3) Otherwise perform vector search (optionally pre-filtered by metadata)
    # 3a) Embed the query
    q_vec = embed_text(req.text)
    # 3b) Search FAISS for top N
    total_to_fetch = (req.page + 1) * req.page_size
    D, I = faiss_index.search([q_vec], total_to_fetch)
    ids = [str(i) for i in I[0]]
    scores = D[0].tolist()
    # 3c) Slice this page
    start = req.page * req.page_size
    end = start + req.page_size
    page_ids = ids[start:end]
    page_scores = scores[start:end]
    # 4) Fetch from Postgres
    rows = fetch_messages_by_ids(page_ids, req.start_ts, req.end_ts)
    hits = []
    for cid, score in zip(page_ids, page_scores):
        if cid in rows:
            r = rows[cid]
            hits.append(Hit(chat_id=cid, ts=r["ts"], user=r["user"], text=r["text"], score=score))
    # 5) Summarize if requested
    answer = None
    if req.summarize and hits:
        context = "\n\n".join([h.text for h in hits])
        answer = summarize_with_llm(req.text, context)
    next_page = req.page + 1 if len(ids) > end else None
    return QueryOut(query=req.text, page=req.page, page_size=req.page_size,
                    hits=hits, next_page=next_page, answer=answer)

@app.post("/subscribe", response_model=SubscribeOut)
def subscribe(req: SubscribeIn):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO subscriptions(user_id, callback, filter) VALUES (%s, %s, %s) RETURNING sub_id",
            (req.user_id, req.callback, json.dumps(req.filter))
        )
        sub_id = cur.fetchone()[0]
    return SubscribeOut(sub_id=sub_id)

@app.delete("/unsubscribe/{sub_id}")
def unsubscribe(sub_id: int):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM subscriptions WHERE sub_id = %s", (sub_id,))
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Subscription not found")
    return {"status": "unsubscribed"}

@app.get("/subscriptions", response_model=List[Subscription])
def list_subscriptions(user_id: str = Depends(authenticate)):
    with conn.cursor() as cur:
        cur.execute("SELECT sub_id, user_id, callback, filter FROM subscriptions WHERE user_id = %s", (user_id,))
        rows = cur.fetchall()
    return [Subscription(sub_id=r[0], user_id=r[1], callback=r[2], filter=r[3]) for r in rows]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
