# placeholderimport sys, json, asyncio, time
from consumer import embed, process, INDEX

rows = json.load(open(sys.argv[1]))
def norm(r):
    ts = r.get("Timestamp")
    return dict(
      timestamp = time.mktime(time.strptime(ts[:19],"%Y-%m-%dT%H:%M:%S")) if ts
                  else time.time(),
      trader    = f"{r['FirstName']} {r['LastName']}",
      company   = r["CompanyName"],
      text      = r["ChatContent"],
      conversation_id = r["ConversationID"]
    )
rows = [norm(r) for r in rows]

async def main():
    B = 64
    for i in range(0,len(rows),B):
        chunk = rows[i:i+B]
        vecs  = embed([c["text"] for c in chunk])
        await process(chunk, vecs)
asyncio.run(main())
