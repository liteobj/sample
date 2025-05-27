
"""
special_agent.py

FastAPI application providing:
1. /query endpoint with metadata filtering, FAISS vector search, pagination, and optional summarization.
2. Supports filtering by tickers, sectors, brokers, sentiment, and time range.
"""

import os
import json
from datetime import datetime
from typing import List, Optional, Dict, Any

import openai
import faiss
import psycopg2
from fastapi import FastAPI, Depends, HTTPException, Query as FQuery
from pydantic import BaseModel
from psycopg2.extras import RealDictCursor

# Configuration from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PG_DSN = os.getenv("PG_DSN")  # e.g., "host=localhost dbname=chat user=postgres password=secret"
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./faiss/combined.index")
EMBED_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-4o"

# Initialize clients
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")
openai.api_key = OPENAI_API_KEY

# Connect to Postgres
pg_conn = psycopg2.connect(PG_DSN, cursor_factory=RealDictCursor)

# Load FAISS index
if not os.path.exists(FAISS_INDEX_PATH):
    raise RuntimeError(f"FAISS index not found at {FAISS_INDEX_PATH}")
faiss_index = faiss.read_index(FAISS_INDEX_PATH)

# Build inverse id_map: chat_id -> vector_idx
def load_id_map():
    with pg_conn.cursor() as cur:
        cur.execute("SELECT chat_id, vector_idx FROM vector_index_map")
        return {row['chat_id']: row['vector_idx'] for row in cur.fetchall()}

id_map = load_id_map()
# Build inverse for idx -> chat_id
id_map_inv = {v: k for k, v in id_map.items()}

# FastAPI app
app = FastAPI(title="Chat Query & Notification Service")

# Pydantic models
class QueryIn(BaseModel):
    text: Optional[str] = None
    tickers: Optional[List[str]] = None
    sectors: Optional[List[str]] = None
    brokers: Optional[List[str]] = None
    sentiment: Optional[List[str]] = None
    start_ts: Optional[datetime] = None
    end_ts: Optional[datetime] = None
    page_size: int = 10
    page: int = 0
    summarize: bool = False

class Hit(BaseModel):
    chat_id: str
    ts: datetime
    user: str
    text: str
    score: float

class QueryOut(BaseModel):
    query: Optional[str]
    filters: Dict[str, Any]
    page: int
    page_size: int
    hits: List[Hit]
    next_page: Optional[int]
    answer: Optional[str]

# Authentication stub
def authenticate(api_key: str = FQuery(..., alias="api_key")):
    # Replace with real validation
    if api_key != "secret":
        raise HTTPException(status_code=401, detail="Invalid api_key")
    return "user-id"

# Helper functions
def embed_text(text: str) -> List[float]:
    resp = openai.Embedding.create(engine=EMBED_MODEL, input=text)
    return resp["data"][0]["embedding"]

def summarize_with_llm(query: str, context: str) -> str:
    prompt = (
        f"User query: {query}\n\n"
        "Based on the following chat messages, provide a concise answer:\n\n"
        f"{context}"
    )
    resp = openai.ChatCompletion.create(
        model=LLM_MODEL,
        messages=[{"role":"user", "content": prompt}],
        max_tokens=200
    )
    return resp.choices[0].message.content.strip()

def fetch_messages_by_ids(ids: List[str],
                          start_ts: Optional[datetime],
                          end_ts: Optional[datetime]) -> Dict[str, Dict]:
    if not ids:
        return {}
    placeholders = ",".join(["%s"] * len(ids))
    conditions = [f"chat_id IN ({placeholders})"]
    params: List[Any] = ids.copy()
    if start_ts:
        conditions.append("ts >= %s")
        params.append(start_ts)
    if end_ts:
        conditions.append("ts <= %s")
        params.append(end_ts)
    where_clause = " AND ".join(conditions)
    sql = f"SELECT chat_id, ts, user_id AS user, raw_text AS text FROM messages WHERE {where_clause}"
    with pg_conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return {row["chat_id"]: row for row in rows}

def filter_by_metadata(req: QueryIn) -> List[str]:
    clauses = []
    params: List[Any] = []
    meta_map = {
        "tickers": req.tickers,
        "sectors": req.sectors,
        "brokers": req.brokers,
        "sentiment": req.sentiment
    }
    for etype, values in meta_map.items():
        if values:
            placeholders = ",".join(["%s"] * len(values))
            clauses.append(f"(entity_type = %s AND value IN ({placeholders}))")
            params.append(etype)
            params.extend(values)
    time_clauses = []
    if req.start_ts:
        time_clauses.append("m.ts >= %s")
        params.append(req.start_ts)
    if req.end_ts:
        time_clauses.append("m.ts <= %s")
        params.append(req.end_ts)
    sql = "SELECT DISTINCT mm.chat_id FROM message_meta mm JOIN messages m USING(chat_id)"
    if clauses or time_clauses:
        sql += " WHERE " + " AND ".join(clauses + time_clauses)
    with pg_conn.cursor() as cur:
        cur.execute(sql, params)
        return [row["chat_id"] for row in cur.fetchall()]

@app.post("/query", response_model=QueryOut)
def query_api(req: QueryIn, user_id: str = Depends(authenticate)):
    filters = {
        "tickers": req.tickers,
        "sectors": req.sectors,
        "brokers": req.brokers,
        "sentiment": req.sentiment,
        "start_ts": req.start_ts,
        "end_ts": req.end_ts
    }

    candidate_ids = filter_by_metadata(req) if any([req.tickers, req.sectors, req.brokers, req.sentiment, req.start_ts, req.end_ts]) else None

    ids, scores = [], []
    if req.text:
        vec = embed_text(req.text)
        norm = sum(v*v for v in vec)**0.5
        vec = [v/norm for v in vec]
        search_k = (req.page + 1) * req.page_size
        if candidate_ids:
            # Full search then filter (for illustration)
            dist, idx = faiss_index.search([vec], search_k)
            pairs = []
            for d, i in zip(dist[0], idx[0]):
                cid = id_map_inv.get(i)
                if cid in candidate_ids:
                    pairs.append((cid, d))
            pairs = sorted(pairs, key=lambda x: -x[1])[:search_k]
            if pairs:
                ids, scores = zip(*pairs)
            else:
                ids, scores = [], []
        else:
            dist, idx = faiss_index.search([vec], search_k)
            ids = [id_map_inv.get(i) for i in idx[0]]
            scores = dist[0].tolist()
    else:
        ids = candidate_ids or []
        scores = [0.0] * len(ids)

    start = req.page * req.page_size
    end = start + req.page_size
    page_ids = ids[start:end] if ids else []
    page_scores = scores[start:end] if scores else []

    rows = fetch_messages_by_ids(page_ids, req.start_ts, req.end_ts)
    hits = []
    for cid, score in zip(page_ids, page_scores):
        row = rows.get(cid)
        if row:
            hits.append(Hit(chat_id=cid, ts=row["ts"], user=row["user"], text=row["text"], score=score))

    answer = None
    if req.summarize and hits:
        context = "\n\n".join([h.text for h in hits])
        answer = summarize_with_llm(req.text or "", context)

    next_page = req.page + 1 if len(ids) > end else None

    return QueryOut(
        query=req.text,
        filters=filters,
        page=req.page,
        page_size=req.page_size,
        hits=hits,
        next_page=next_page,
        answer=answer
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("special_agent:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
