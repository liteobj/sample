"""
       consumer.py   ·   GPT multilabel tagger
       ────────────────────────────────────────
  * embeds each chat line (ada-002)
  * candidate-filters labels, calls GPT-4o-mini
  * writes row to Postgres
  * adds vector to local FAISS index
  * every SNAPSHOT_INTERVAL rows:
        – saves   /tmp/today.faiss
        – uploads s3://$BUCKET/latest/today.faiss
"""
import os, json, asyncio, pathlib, textwrap, time, datetime
import numpy as np, sqlalchemy as sa, faiss, openai, boto3
from sqlalchemy import text as sql

# ── env ──────────────────────────────────────────────────────────────
DB_URL        = os.getenv("DATABASE_URL")
FAISS_PATH    = "/tmp/today.faiss"
S3_BUCKET     = os.getenv("S3_BUCKET")          # e.g. chat-faiss
LABEL_FILE    = os.getenv("LABEL_FILE", "/app/labels.txt")
TOP_K         = int(os.getenv("LABEL_TOP_K", 12))
THRESHOLD     = float(os.getenv("LABEL_SCORE_THRESHOLD", 0.30))
SNAPSHOT_EVERY= int(os.getenv("SNAPSHOT_INTERVAL", 200))  # rows
openai.api_key= os.getenv("OPENAI_API_KEY")

s3     = boto3.client("s3")
engine = sa.create_engine(DB_URL, future=True, pool_pre_ping=True)

# ── schema ───────────────────────────────────────────────────────────
with engine.begin() as c:
    c.execute(sql("""
      CREATE TABLE IF NOT EXISTS chat_messages (
        id               BIGSERIAL PRIMARY KEY,
        ts               TIMESTAMPTZ NOT NULL,
        trader           TEXT,
        company          TEXT,
        conversation_id  TEXT,
        tags             JSONB,
        snippet          TEXT,
        UNIQUE(conversation_id, snippet)
      );
    """))

# ── labels & embeddings ──────────────────────────────────────────────
def load_labels(p):
    out = []
    with open(p) as f:
        for ln in f:
            if ":" in ln:
                k, v = ln.split(":", 1)
                out.append((k.strip(), v.strip()))
    return out

LABELS      = load_labels(LABEL_FILE)
LABEL_NAMES = [l for l, _ in LABELS]

def embed(txts):
    r = openai.embeddings.create(model="text-embedding-ada-002", input=txts)
    return np.array([d.embedding for d in r.data], dtype="float32")

LABEL_VECS  = embed([d for _, d in LABELS])
LABEL_VECS /= np.linalg.norm(LABEL_VECS, axis=1, keepdims=True)

# ── FAISS local index ────────────────────────────────────────────────
INDEX = faiss.index_factory(1536, "IDMap2,HNSW32")
row_counter = 0

# ── helpers ──────────────────────────────────────────────────────────
def daily_key():
    d = datetime.date.today().isoformat()
    return f"daily/{d}/today.faiss"

def upload_snapshot():
    tmp = FAISS_PATH + ".tmp"
    faiss.write_index(INDEX, tmp)
    os.replace(tmp, FAISS_PATH)
    s3.upload_file(FAISS_PATH, S3_BUCKET, daily_key())
    s3.copy_object(Bucket=S3_BUCKET,
                   CopySource=f"{S3_BUCKET}/{daily_key()}",
                   Key="latest/today.faiss")
    print("▲ uploaded latest/today.faiss")

def prompt(cands, chat):
    bullet = "\n".join(f"- **{l}**: {d}" for l, d in cands)
    return [
        {"role":"system","content":textwrap.dedent(f"""
            Return JSON {{ "labels":[{{"label":"X","score":0.9}}] }}.
            Include only labels ≥5 % confidence.
            ---
            Labels:
            {bullet}
        """).strip()},
        {"role":"user","content":chat}
    ]

async def classify(text, vec):
    sims = (LABEL_VECS @ (vec/np.linalg.norm(vec)).T).squeeze()
    idx  = sims.argsort()[-TOP_K:][::-1]
    cands= [(LABEL_NAMES[i], LABELS[i][1]) for i in idx]
    comp = await openai.chat.completions.async_create(
        model="gpt-4o-mini",
        response_format={"type":"json_object"},
        max_tokens=100,
        messages=prompt(cands, text)
    )
    try:
        out = json.loads(comp.choices[0].message.content)["labels"]
        return [l for l in out if l.get("score",0) >= THRESHOLD]
    except Exception:
        return []

async def process(batch, vecs):
    global row_counter
    with engine.begin() as c:
        for rec, vec in zip(batch, vecs):
            labels = await classify(rec["text"], vec)
            res = c.execute(sql("""
                INSERT INTO chat_messages
                (ts,trader,company,conversation_id,tags,snippet)
                VALUES (to_timestamp(:ts),:tr,:co,:cid,:tg,:snip)
                ON CONFLICT DO NOTHING
                RETURNING id"""),
                dict(ts = rec["timestamp"],
                     tr = rec["trader"],
                     co = rec["company"],
                     cid= rec["conversation_id"],
                     tg = json.dumps({"labels":labels}),
                     snip = rec["text"][:120]))
            row_id = res.scalar_one_or_none()
            if row_id:
                INDEX.add_with_ids(vec.reshape(1,-1),
                                   np.array([row_id],dtype="int64"))
                row_counter += 1
    if row_counter and row_counter % SNAPSHOT_EVERY == 0:
        upload_snapshot()

# --------------------------------------------------------------------
# If running via file_runner this process() is imported & reused.
# --------------------------------------------------------------------
