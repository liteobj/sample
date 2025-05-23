#!/usr/bin/env python3
"""
ingest_service.py  –  End-to-end Bloomberg chat ingestor + LLM-based enricher
---------------------------------------------------------------------------
ENV VARS NEEDED
  OPENAI_API_KEY         – OpenAI credentials
  PG_DSN                 – e.g. "postgresql+asyncpg://user:pass@host/db"
  BIZ_CORPUS_PATH        – Folder with CSV/TXT/PDF chunks for biz terms
  BIZ_INDEX_PATH         – pre-built FAISS index file (will create if absent)
  CHAT_INDEX_DIR         – folder to store daily FAISS shards
  EMBED_MODEL            – default: "text-embedding-3-large"
  CHAT_MODEL             – default: "gpt-4o-mini"
"""
import asyncio, contextlib, datetime as dt, json, os, pathlib, pickle, typing as T
import asyncpg, aiofiles
import faiss
import numpy as np
import openai

# ---------------------------------------------------------------------------#
# 0.  CONFIG
# ---------------------------------------------------------------------------#
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
CHAT_MODEL  = os.getenv("CHAT_MODEL",  "gpt-4o-mini")
K_SUPPORT   = 4                       # RAG top-k
BATCH_EMBED = 64                      # embed chat lines in batches

# ---------------------------------------------------------------------------#
# 1.  BUSINESS-TERM RAG INDEX
# ---------------------------------------------------------------------------#
class BizIndex:
    """FAISS index of business-terms corpus. 2 arrays kept in RAM:
       - index: vector store
       - passages: list[str] aligned with index IDs
    """
    def __init__(self, index_path: pathlib.Path, corpus_dir: pathlib.Path):
        self.index_path, self.corpus_dir = index_path, corpus_dir
        self.passages: list[str] = []
        self.index: faiss.IndexFlatIP | None = None

    async def load(self):
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            async with aiofiles.open(self.index_path.with_suffix(".pkl"), "rb") as f:
                self.passages = pickle.loads(await f.read())
        else:
            await self._build_from_corpus()

    async def _build_from_corpus(self):
        print("💡 Building biz-term FAISS index…")
        texts: list[str] = []
        for p in self.corpus_dir.glob("**/*"):
            if p.suffix.lower() in {".txt", ".csv"}:
                async with aiofiles.open(p) as f:
                    texts += [ln.strip() for ln in (await f.read()).splitlines() if ln.strip()]
            # TODO: pdf/docx parsing if needed
        # chunk if any passage > 500 tokens (simple heuristic)
        passages = []
        for t in texts:
            passages += (t[i:i+500] for i in range(0, len(t), 500))
        self.passages = passages

        # embed in small batches
        vecs = []
        for i in range(0, len(passages), BATCH_EMBED):
            batch = passages[i:i+BATCH_EMBED]
            e = await openai.embeddings.async_create(model=EMBED_MODEL, input=batch)
            vecs += [d["embedding"] for d in e["data"]]

        arr = np.asarray(vecs, dtype="float32")
        index = faiss.IndexFlatIP(arr.shape[1])
        index.add(arr / np.linalg.norm(arr, axis=1, keepdims=True))
        self.index = index
        faiss.write_index(index, str(self.index_path))
        async with aiofiles.open(self.index_path.with_suffix(".pkl"), "wb") as f:
            await f.write(pickle.dumps(passages))
        print(f"✅ Biz index built with {len(passages):,} passages")

    async def top_k(self, text: str, k: int = K_SUPPORT) -> list[str]:
        e = await openai.embeddings.async_create(model=EMBED_MODEL, input=text)
        q = np.asarray(e["data"][0]["embedding"], dtype="float32")
        q /= np.linalg.norm(q) + 1e-12
        D, I = self.index.search(q[None, :], k)
        return [self.passages[i] for i in I[0] if i >= 0]


# ---------------------------------------------------------------------------#
# 2.  OPENAI FUNCTION SCHEMA & ENRICHMENT CALL
# ---------------------------------------------------------------------------#
ENRICH_FUNC = {
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

async def enrich_with_llm(msg: str, biz_index: BizIndex) -> dict:
    support = await biz_index.top_k(msg, K_SUPPORT)
    messages = [
        {"role": "system", "content":
         "You are a domain expert.  Use the following reference passages to interpret "
         "financial jargon and produce JSON via the given function schema."},
        {"role": "system", "content": "\n\n".join(support)},
        {"role": "user", "content": msg}
    ]
    resp = await openai.chat.completions.async_create(
        model=CHAT_MODEL,
        messages=messages,
        functions=[ENRICH_FUNC],
        function_call={"name": "enrich_chat"}
    )
    args_json = resp.choices[0].message.function_call.arguments
    return json.loads(args_json)


# ---------------------------------------------------------------------------#
# 3.  POSTGRES STORE
# ---------------------------------------------------------------------------#
class PgStore:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.pool: asyncpg.Pool | None = None
    async def open(self): self.pool = await asyncpg.create_pool(self.dsn, min_size=4, max_size=16)
    async def close(self): await self.pool.close() if self.pool else None

    async def insert_message(self, raw: dict, meta: dict) -> int:
        q_msg = """
            INSERT INTO chat_messages(conv_id, ts, speaker, speaker_firm, body_raw)
            VALUES($1,$2,$3,$4,$5) RETURNING id
        """
        q_meta = """
            INSERT INTO chat_metadata(msg_id, body_clean, tickers, brokers, geos, biz_terms,
                                      sector, sentiment, tsv)
            VALUES($1,$2,$3,$4,$5,$6,$7,$8, to_tsvector('english', $2))
        """
        async with self.pool.acquire() as conn, conn.transaction():
            msg_id = await conn.fetchval(
                q_msg, raw["conv_id"], raw["ts"], raw["speaker"],
                raw["speaker_firm"], raw["body_raw"]
            )
            await conn.execute(
                q_meta, msg_id, raw["body_raw"], meta.get("tickers"),
                meta.get("brokers"), meta.get("geos"), meta.get("biz_terms"),
                meta.get("sector"), meta.get("sentiment")
            )
        return msg_id


# ---------------------------------------------------------------------------#
# 4.  CHAT-MESSAGE EMBEDDINGS & DATE-SHARDED FAISS INDEX
# ---------------------------------------------------------------------------#
class ChatIndexManager:
    """One FAISS index per UTC-date, persisted under CHAT_INDEX_DIR/yyyy-mm-dd.faiss"""
    def __init__(self, base_dir: pathlib.Path):
        self.base_dir = base_dir
        self._cache: dict[str, tuple[faiss.IndexFlatIP, list[int]]] = {}

    def _idx_path(self, date: dt.date) -> pathlib.Path:
        return self.base_dir / f"{date}.faiss"

    def _load_or_create(self, date: dt.date) -> faiss.IndexFlatIP:
        key = str(date)
        if key in self._cache:  # hot in memory
            return self._cache[key][0]
        p = self._idx_path(date)
        if p.exists():
            idx = faiss.read_index(str(p))
            ids = pickle.loads(p.with_suffix(".ids").read_bytes())
        else:
            idx = faiss.IndexFlatIP(1536)  # embedding dim of text-embedding-3-large
            ids = []                       # chat_msg ids aligned to vectors
        self._cache[key] = (idx, ids)
        return idx

    def add_vector(self, vec: list[float], msg_id: int, ts: dt.datetime):
        date = ts.date()          # shard key
        idx = self._load_or_create(date)
        v = np.asarray(vec, dtype="float32")[None, :]
        v /= np.linalg.norm(v) + 1e-12
        idx.add(v)
        self._cache[str(date)][1].append(msg_id)

    def flush(self):
        for key, (idx, ids) in self._cache.items():
            p = self._idx_path(dt.date.fromisoformat(key))
            faiss.write_index(idx, str(p))
            p.with_suffix(".ids").write_bytes(pickle.dumps(ids))


# ---------------------------------------------------------------------------#
# 5.  CHAT FEED (stub)  – replace with real Bloomberg connector
# ---------------------------------------------------------------------------#
async def chat_feed_source(queue: asyncio.Queue):
    """Streams dummy chat messages every 0.3 s."""
    demo_lines = [
        {"FirstName": "Joe", "LastName": "Trdr", "CompanyName": "GS",
         "ConversationID": "C1", "ChatContent": "Huge IOI in AAPL calls atm",
         "Timestamp": dt.datetime.utcnow()},
        {"FirstName": "Alice", "LastName": "Desk", "CompanyName": "BAML",
         "ConversationID": "C2", "ChatContent": "Brazil FX vols spiking vs USD",
         "Timestamp": dt.datetime.utcnow()},
    ]
    i = 0
    while True:
        await queue.put(demo_lines[i % len(demo_lines)])
        i += 1
        await asyncio.sleep(0.3)

# ---------------------------------------------------------------------------#
# 6.  ENRICHMENT WORKER
# ---------------------------------------------------------------------------#
async def enrichment_worker(queue: asyncio.Queue,
                            biz_index: BizIndex,
                            pg: PgStore,
                            chat_idx: ChatIndexManager):
    while True:
        msg = await queue.get()
        try:
            meta = await enrich_with_llm(msg["ChatContent"], biz_index)
            msg_id = await pg.insert_message(_raw_row(msg), meta)

            # embed & faiss
            e = await openai.embeddings.async_create(model=EMBED_MODEL,
                                                     input=msg["ChatContent"])
            chat_idx.add_vector(e["data"][0]["embedding"], msg_id, msg["Timestamp"])
        except Exception as exc:
            # basic retry / logging
            print("❌ enrichment error:", exc, "– message skipped")
        finally:
            queue.task_done()

def _raw_row(m: dict) -> dict:
    return {
        "conv_id":    m["ConversationID"],
        "ts":         m["Timestamp"],
        "speaker":    f"{m['FirstName']} {m['LastName']}",
        "speaker_firm": m["CompanyName"],
        "body_raw":   m["ChatContent"]
    }

# ---------------------------------------------------------------------------#
# 7.  MAIN APP
# ---------------------------------------------------------------------------#
async def main():
    # ---- paths & stores
    biz_index_path = pathlib.Path(os.getenv("BIZ_INDEX_PATH", "./biz_index.faiss"))
    biz_corpus_dir = pathlib.Path(os.getenv("BIZ_CORPUS_PATH", "./biz_corpus"))
    chat_index_dir = pathlib.Path(os.getenv("CHAT_INDEX_DIR", "./chat_indices"))
    chat_index_dir.mkdir(exist_ok=True, parents=True)

    biz_idx = BizIndex(biz_index_path, biz_corpus_dir); await biz_idx.load()
    pg = PgStore(os.getenv("PG_DSN")); await pg.open()
    chat_idx = ChatIndexManager(chat_index_dir)

    # ---- async tasks
    q: asyncio.Queue = asyncio.Queue(2048)
    feeder  = asyncio.create_task(chat_feed_source(q))
    worker  = asyncio.create_task(enrichment_worker(q, biz_idx, pg, chat_idx))

    async def _shutdown():
        print("⏳ draining queue…")
        await q.join()
        feeder.cancel(); worker.cancel()
        chat_idx.flush()
        await pg.close()

    # run until Ctrl-C
    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        await _shutdown()

if __name__ == "__main__":
    asyncio.run(main())