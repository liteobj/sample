"""
api/app.py  ·  date-aware, ticker-aware query service
----------------------------------------------------
* Extracts ticker + date range from the user question with GPT function-calling
* Fuzzy-maps misspelt tickers (“NVDIA” → “NVDA”)
* Dynamically loads only the FAISS shards that overlap the requested dates
* Optional label/score filters preserved
"""
import os, json, pathlib, asyncio, datetime, time, aiofiles.os
from dateutil import parser as dtp
from rapidfuzz import process as fuzz
import numpy as np, faiss, openai, sqlalchemy as sa, boto3
from fastapi import FastAPI
from pydantic import BaseModel

# ── ENV & GLOBALS ──────────────────────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")
DB_URL    = os.getenv("DATABASE_URL")
BUCKET    = os.getenv("S3_BUCKET")
CACHE_DIR = "/cache"                    # emptyDir volume
VALID_TICKERS = {"AAPL","NVDA","TSLA","MSFT"}  # extend as needed

engine = sa.create_engine(DB_URL, future=True, pool_pre_ping=True)
s3     = boto3.client("s3")
app    = FastAPI()

# ── Pydantic payload ───────────────────────────────────────────────────────
class Query(BaseModel):
    question: str
    labels:   list[str] | None = None
    min_score: float = 0.30
    top_k:    int = 20

# ── GPT filter-extraction schema ───────────────────────────────────────────
extract_schema = {
  "name": "extract_filters",
  "description": "Pick out explicit or implicit ticker and date range from the question.",
  "parameters": {
    "type": "object",
    "properties": {
      "ticker":    {"type":"string","description":"Stock ticker if present"},
      "date_from": {"type":"string","description":"ISO date"},
      "date_to":   {"type":"string","description":"ISO date"}
    },
    "required": []
  }
}

async def extract_filters(question: str) -> dict:
    comp = await openai.chat.completions.async_create(
        model="gpt-4o-mini",
        tools=[{"type":"function","function":extract_schema}],
        messages=[{"role":"user","content":question}],
        max_tokens=40
    )
    try:
        call = comp.choices[0].message.tool_calls[0]
        return json.loads(call["arguments"])
    except Exception:
        return {}

def fuzzy_ticker(t: str|None) -> str|None:
    if not t: return None
    best, score = fuzz.extractOne(t.upper(), VALID_TICKERS, score_cutoff=80)
    return best if best else None

# ── FAISS shard helpers ────────────────────────────────────────────────────
s3_client = boto3.client("s3")
def s3_key_for_day(d: datetime.date) -> str:
    return "latest/today.faiss" if d == datetime.date.today() else f"latest/{d}.faiss"

def local_path_for_day(d: datetime.date) -> str:
    return f"{CACHE_DIR}/today.faiss" if d == datetime.date.today() else f"{CACHE_DIR}/{d}.faiss"

async def ensure_shard(d: datetime.date):
    key  = s3_key_for_day(d)
    path = local_path_for_day(d)
    remote = s3_client.head_object(Bucket=BUCKET, Key=key)
    r_mtime = remote["LastModified"].timestamp()
    if not pathlib.Path(path).exists() or r_mtime > pathlib.Path(path).stat().st_mtime:
        tmp = path + ".tmp"
        s3_client.download_file(BUCKET, key, tmp)
        await aiofiles.os.rename(tmp, path)

async def build_index(days: list[datetime.date]) -> faiss.Index:
    if len(days) == 1:
        await ensure_shard(days[0])
        return faiss.read_index(local_path_for_day(days[0]),
                                faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
    combo = faiss.IndexShards(1536, False, False)
    for d in days:
        await ensure_shard(d)
        idx = faiss.read_index(local_path_for_day(d),
                               faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
        combo.add_shard(idx)
    return combo

# ── Embedding & SQL helpers ────────────────────────────────────────────────
def embed(txt: str) -> np.ndarray:
    r = openai.embeddings.create(model="text-embedding-ada-002", input=txt)
    return np.array(r.data[0].embedding, dtype="float32")[None,:]

def fetch(ids, labels, ms):
    wh, p = ["id = ANY(:ids)"], {"ids": ids}
    if labels:
        wh.append("""
          EXISTS (SELECT 1 FROM jsonb_array_elements(tags->'labels') l
                  WHERE l->>'label' = ANY(:lbl)
                    AND (l->>'score')::float >= :ms)
        """)
        p["lbl"], p["ms"] = labels, ms
    sql = f"SELECT id,snippet,trader,company,tags FROM chat_messages WHERE {' AND '.join(wh)}"
    with engine.begin() as c:
        return c.execute(sa.text(sql), p).fetchall()

# ── Endpoint ───────────────────────────────────────────────────────────────
@app.post("/query")
async def query(q: Query):
    # 1· extract filters
    f = await extract_filters(q.question)
    ticker = fuzzy_ticker(f.get("ticker"))
    try:
        d_from = dtp.parse(f.get("date_from")).date() if f.get("date_from") else datetime.date.today()
        d_to   = dtp.parse(f.get("date_to")).date()   if f.get("date_to")   else datetime.date.today()
    except Exception:
        d_from = d_to = datetime.date.today()
    if d_from > d_to: d_from, d_to = d_to, d_from
    days = [d_from + datetime.timedelta(days=i) for i in range((d_to-d_from).days+1)]

    # 2· build index for those days
    index = await build_index(days)

    # 3· semantic search
    q_vec = embed(q.question)
    D, I  = index.search(q_vec, q.top_k)
    ids   = [int(x) for x in I[0] if x != -1]
    if not ids:
        return {"answer":"No chats found."}

    # 4· hydrate rows (+ optional label filter)
    rows = fetch(ids, q.labels, q.min_score)
    id2d = dict(zip(ids, D[0]))
    rows.sort(key=lambda r: id2d[r.id])
    text_ctx = "\n".join(r.snippet for r in rows)

    # 5· Compose final answer
    system = "You are a trading-chat analyst. Count quotes if user asks 'how many'."
    chat = [
      {"role":"system","content":system},
      {"role":"user","content":f"Question: {q.question}\nChats:\n{text_ctx}"}
    ]
    ans = openai.chat.completions.create(model="gpt-4o-mini",
                                         messages=chat,
                                         max_tokens=300)
    return {"answer": ans.choices[0].message.content,
            "hits": [dict(r) for r in rows]}
