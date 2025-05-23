#!/usr/bin/env python3
"""
app.py – Service #2: user-facing API for Q&A + market-event notifications
========================================================================
ENV VARS
  OPENAI_API_KEY    – OpenAI key
  PG_DSN            – "postgresql+asyncpg://user:pass@host/db"
  INDEX_FILE        – default ./chat_index.faiss
  IDS_FILE          – default ./chat_index.ids
"""
import os, datetime as dt, json, pathlib
from typing import List, Optional

import asyncpg, fastapi, faiss, numpy as np, openai
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

# ------------------------------------------------------------------#
# 1.  CONSTANTS & SHORTCUTS
# ------------------------------------------------------------------#
EMBED_MODEL  = "text-embedding-ada-002"
CHAT_MODEL   = "gpt-4o-mini"
TOP_K_FAISS  = 120     # vectors to pull from FAISS for every query
MAX_CONTEXT  = 20      # lines forwarded to GPT for synthesis

INDEX_FILE   = pathlib.Path(os.getenv("INDEX_FILE", "chat_index.faiss"))
IDS_FILE     = pathlib.Path(os.getenv("IDS_FILE",   "chat_index.ids"))

# ------------------------------------------------------------------#
# 2.  LOAD FAISS + ID LIST (kept in RAM)
# ------------------------------------------------------------------#
if not INDEX_FILE.exists():
    raise RuntimeError("chat_index.faiss not found – run Service #1 first")

faiss_index = faiss.read_index(str(INDEX_FILE))
msg_ids: List[int] = json.loads(IDS_FILE.read_text())   # simple list[int]


# ------------------------------------------------------------------#
# 3.  POSTGRES CONNECTION POOL
# ------------------------------------------------------------------#
pool: asyncpg.Pool | None = None
async def pg():  # lazy getter
    global pool
    if pool is None:
        pool = await asyncpg.create_pool(os.getenv("PG_DSN"), min_size=2, max_size=10)
    return pool


# ------------------------------------------------------------------#
# 4.  API MODELS
# ------------------------------------------------------------------#
class AskRequest(BaseModel):
    question: str
    lookback_days: Optional[int] = 7    # default 7-day window
    top_k: Optional[int] = 3            # how many bullets to return

class AskResponse(BaseModel):
    answer: str
    sources: List[str]                  # raw chat lines we exposed to GPT

class Event(BaseModel):
    ts_event: dt.datetime
    entity: str
    event_type: str
    score: float
    sample: str


# ------------------------------------------------------------------#
# 5.  CORE SEARCH → ANSWER PIPELINE
# ------------------------------------------------------------------#
async def embed(text: str) -> np.ndarray:
    vec = await openai.embeddings.async_create(input=text, model=EMBED_MODEL)
    v = np.asarray(vec["data"][0]["embedding"], dtype="float32")
    return v / np.linalg.norm(v)

async def search_chat(query: str, days: int) -> List[asyncpg.Record]:
    """1) embed & FAISS → top IDs; 2) db fetch & time-filter; returns
       chat rows sorted by FAISS score."""
    q_vec = await embed(query)
    D, I = faiss_index.search(q_vec[None, :], TOP_K_FAISS)
    ids = [msg_ids[i] for i in I[0] if i >= 0]
    if not ids:
        return []
    rows = await (await pg()).fetch(
        """
        SELECT c.ts, c.speaker, c.body_raw
        FROM chat_messages c
        WHERE c.id = ANY($1::bigint[])
          AND c.ts >= NOW() - $2::interval
        ORDER BY c.ts DESC
        """,
        ids, f"{days} days"
    )
    # preserve FAISS order
    id2rank = {mid: r for r, mid in enumerate(ids)}
    rows.sort(key=lambda r: id2rank[r["id"]])
    return rows

async def gpt_answer(question: str, bullets: List[str]) -> str:
    ctx = "\n\n".join(f"• {b}" for b in bullets[:MAX_CONTEXT])
    messages = [
        {"role": "system",
         "content": ("You are a sell-side market strategist summarising "
                     "Bloomberg chat intel.  Cite evidence where relevant.")},
        {"role": "system", "content": f"Context:\n{ctx}"},
        {"role": "user",   "content": question}
    ]
    resp = await openai.chat.completions.async_create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=400
    )
    return resp.choices[0].message.content.strip()


# ------------------------------------------------------------------#
# 6.  FASTAPI ENDPOINTS
# ------------------------------------------------------------------#
app = fastapi.FastAPI(title="Bloomberg-Chat Q&A API")

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    rows = await search_chat(req.question, req.lookback_days)
    if not rows:
        return AskResponse(answer="Sorry, no relevant chat found.", sources=[])

    bullets = [f"{r['speaker']} ({r['ts']:%b %d %H:%M}): {r['body_raw']}"
               for r in rows[:req.top_k * 5]]       # a bit more for GPT
    answer  = await gpt_answer(req.question, bullets)
    return AskResponse(answer=answer, sources=bullets[:req.top_k])

@app.get("/events", response_model=List[Event])
async def latest_events(limit: int = 25):
    recs = await (await pg()).fetch(
        """
        SELECT e.ts_event, e.entity, e.event_type, e.score, c.body_raw AS sample
        FROM market_events e
        JOIN chat_messages c ON c.id = e.sample_msg
        ORDER BY e.ts_event DESC
        LIMIT $1
        """, limit
    )
    return [Event(**dict(r)) for r in recs]

# ---------- optional: live event stream via WebSocket ----------------#
class ConnectionManager:
    def __init__(self): self.active: set[WebSocket] = set()
    async def connect(self, ws: WebSocket):
        await ws.accept(); self.active.add(ws)
    def disconnect(self, ws: WebSocket): self.active.discard(ws)
    async def broadcast(self, data: str):
        for ws in list(self.active):
            try:    await ws.send_text(data)
            except WebSocketDisconnect:
                self.disconnect(ws)

manager = ConnectionManager()

@app.websocket("/events/ws")
async def events_ws(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:  # keep alive; push from NOTIFY later
            await asyncio.sleep(60)
    finally:
        manager.disconnect(ws)

# ------------------------------------------------------------------#
# 7.  OPTIONAL: tiny task that LISTEN/NOTIFY publishes new events
# ------------------------------------------------------------------#
import asyncio
async def event_notifier():
    con = await (await pg()).acquire()
    await con.add_listener("new_event", lambda *_: None)  # dummy to activate channel
    try:
        while True:
            await asyncio.sleep(0.5)   # notify polling interval
            while con.notifies:
                n = con.notifies.pop()
                await manager.broadcast(n.payload)
    finally:
        await (await pg()).release(con)

@app.on_event("startup")
async def on_start():
    asyncio.create_task(event_notifier())

@app.on_event("shutdown")
async def on_stop():
    if pool: await pool.close()