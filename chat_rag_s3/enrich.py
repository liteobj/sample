#!/usr/bin/env python3
"""
simple_ingest.py – minimal end-to-end Bloomberg chat ingestor with
OpenAI-enriched metadata + single-file FAISS index.

ENV VARS ---------------------------------------------------------------
  OPENAI_API_KEY    – OpenAI key
  PG_DSN            – postgres+asyncpg DSN
  BIZ_DIR           – folder with *.txt / *.csv chunks of business terms
------------------------------------------------------------------------
"""
import os, asyncio, datetime as dt, json, pickle, pathlib, typing as T

import aiofiles, asyncpg, faiss, numpy as np, openai

# ---------------------------------------------------------------------#
# 1.  CONSTANTS
# ---------------------------------------------------------------------#
EMBED_MODEL  = "text-embedding-ada-002"
CHAT_MODEL   = "gpt-4o-mini"
K_SUPPORT    = 4                   # business-term passages per prompt
INDEX_FILE   = pathlib.Path("chat_index.faiss")
IDS_FILE     = INDEX_FILE.with_suffix(".ids")

# ---------------------------------------------------------------------#
# 2.  BUSINESS-TERM PASSAGES  (plain Python, no FAISS here)
# ---------------------------------------------------------------------#
async def load_passages(folder: pathlib.Path) -> list[str]:
    plist = []
    for p in folder.glob("**/*"):
        if p.suffix.lower() in {".txt", ".csv"}:
            async with aiofiles.open(p) as f:
                plist += [ln.strip() for ln in (await f.read()).splitlines() if ln.strip()]
    return plist

async def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    resp = await openai.embeddings.async_create(input=texts, model=EMBED_MODEL)
    return [d["embedding"] for d in resp["data"]]

async def build_passage_vectors(folder: pathlib.Path):
    passages = await load_passages(folder)
    vecs     = await embed_texts(passages)
    arr      = np.asarray(vecs, dtype="float32")
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    return passages, arr

# ---------------------------------------------------------------------#
# 3.  VERY SIMPLE GLOBAL FAISS INDEX (one file)
# ---------------------------------------------------------------------#
def load_chat_index() -> tuple[faiss.IndexFlatIP, list[int]]:
    if INDEX_FILE.exists():
        idx  = faiss.read_index(str(INDEX_FILE))
        ids  = pickle.loads(IDS_FILE.read_bytes())
    else:
        idx = faiss.IndexFlatIP(1536)
        ids = []
    return idx, ids

def save_chat_index(idx: faiss.IndexFlatIP, ids: list[int]):
    faiss.write_index(idx, str(INDEX_FILE))
    IDS_FILE.write_bytes(pickle.dumps(ids))

def add_embedding(idx: faiss.IndexFlatIP, ids: list[int], vec: list[float], msg_id: int):
    v = np.asarray(vec, dtype="float32")[None]
    v /= np.linalg.norm(v) + 1e-12
    idx.add(v)
    ids.append(msg_id)

# ---------------------------------------------------------------------#
# 4.  OPENAI FUNCTION-CALL FOR ENRICHMENT
# ---------------------------------------------------------------------#
FUNC_SCHEMA = {
    "name": "enrich_chat",
    "parameters": {
        "type": "object",
        "properties": {
            "tickers":   {"type": "array", "items": {"type": "string"}},
            "brokers":   {"type": "array", "items": {"type": "string"}},
            "geos":      {"type": "array", "items": {"type": "string"}},
            "biz_terms": {"type": "array", "items": {"type": "string"}},
            "sector":    {"type": "string"},
            "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]}
        },
        "required": ["biz_terms"]
    }
}

async def enrich_line(body: str,
                      passages: list[str],
                      pvecs: np.ndarray) -> dict:
    # --- pick K_SUPPORT passages via cosine similarity (dot product) ----
    e = await openai.embeddings.async_create(input=body, model=EMBED_MODEL)
    q = np.asarray(e["data"][0]["embedding"], dtype="float32")
    q /= np.linalg.norm(q) + 1e-12
    sims = (pvecs @ q).flatten()
    topk = sims.argsort()[-K_SUPPORT:][::-1]
    support = "\n\n".join(passages[i] for i in topk)

    # --- LLM call -------------------------------------------------------
    messages = [
        {"role": "system",
         "content": ("You are a domain expert. Use the reference text to decode "
                     "market jargon and produce JSON via the function schema.")},
        {"role": "system", "content": support},
        {"role": "user",   "content": body}
    ]
    resp = await openai.chat.completions.async_create(
        model=CHAT_MODEL,
        messages=messages,
        functions=[FUNC_SCHEMA],
        function_call={"name": "enrich_chat"}
    )
    args = resp.choices[0].message.function_call.arguments
    return json.loads(args)

# ---------------------------------------------------------------------#
# 5.  POSTGRES WRITER (single row + meta in one tx)
# ---------------------------------------------------------------------#
class Store:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.pool: asyncpg.Pool | None = None
    async def open(self):  self.pool = await asyncpg.create_pool(self.dsn)
    async def close(self): await self.pool.close()
    async def insert(self, raw: dict, meta: dict) -> int:
        q1 = """INSERT INTO chat_messages(conv_id, ts, speaker, speaker_firm, body_raw)
                VALUES($1,$2,$3,$4,$5) RETURNING id"""
        q2 = """INSERT INTO chat_metadata(
                   msg_id, body_clean, tickers, brokers, geos, biz_terms, sector, sentiment,
                   tsv)
                VALUES($1,$2,$3,$4,$5,$6,$7,$8,
                       to_tsvector('english', $2))"""
        async with self.pool.acquire() as c, c.transaction():
            mid = await c.fetchval(q1, raw["conv_id"], raw["ts"],
                                   raw["speaker"], raw["speaker_firm"],
                                   raw["body_raw"])
            await c.execute(q2, mid, raw["body_raw"],
                            meta.get("tickers"), meta.get("brokers"),
                            meta.get("geos"), meta.get("biz_terms"),
                            meta.get("sector"), meta.get("sentiment"))
        return mid

# ---------------------------------------------------------------------#
# 6.  DUMMY CHAT FEED  (swap with real Bloomberg consumer)
# ---------------------------------------------------------------------#
async def fake_feed(queue: "asyncio.Queue[dict]"):
    demo = [
        {"FirstName": "Joe",   "LastName": "Trader", "CompanyName": "GS",
         "ConversationID": "C1", "ChatContent": "Unusual IOI in AAPL 200c",
         "Timestamp": dt.datetime.utcnow()},
        {"FirstName": "Alice", "LastName": "Desk", "CompanyName": "BAML",
         "ConversationID": "C2", "ChatContent": "Brazil FX vols spiking hard",
         "Timestamp": dt.datetime.utcnow()},
    ]
    i = 0
    while True:
        await queue.put(demo[i % len(demo)]); i += 1
        await asyncio.sleep(0.3)

def to_raw_row(m: dict) -> dict:
    return dict(
        conv_id=m["ConversationID"],
        ts=m["Timestamp"],
        speaker=f"{m['FirstName']} {m['LastName']}",
        speaker_firm=m["CompanyName"],
        body_raw=m["ChatContent"]
    )

# ---------------------------------------------------------------------#
# 7.  MAIN LOOP
# ---------------------------------------------------------------------#
async def main():
    # -- prepare corpus for RAG once ------------------------------------
    biz_dir = pathlib.Path(os.getenv("BIZ_DIR", "./biz_terms"))
    passages, pvecs = await build_passage_vectors(biz_dir)

    # -- load / create chat index ---------------------------------------
    chat_idx, id_list = load_chat_index()

    # -- postgres -------------------------------------------------------
    store = Store(os.getenv("PG_DSN"))
    await store.open()

    # -- queue & tasks --------------------------------------------------
    q: "asyncio.Queue[dict]" = asyncio.Queue(2048)
    feeder  = asyncio.create_task(fake_feed(q))

    async def worker():
        while True:
            msg = await q.get()
            try:
                meta = await enrich_line(msg["ChatContent"], passages, pvecs)
                mid  = await store.insert(to_raw_row(msg), meta)
                # embed & faiss
                e = await openai.embeddings.async_create(input=msg["ChatContent"],
                                                         model=EMBED_MODEL)
                add_embedding(chat_idx, id_list, e["data"][0]["embedding"], mid)
            except Exception as exc:
                print("⚠️  failed:", exc)
            finally:
                q.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(4)]

    # -- graceful shutdown ---------------------------------------------
    async def shutdown():
        feeder.cancel()
        for w in workers: w.cancel()
        await q.join()
        save_chat_index(chat_idx, id_list)
        await store.close()

    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        await shutdown()

if __name__ == "__main__":
    asyncio.run(main())