import os, json, asyncio, threading
from typing import Dict, List, Set
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse
from contextlib import asynccontextmanager

JSONL_PATH = "face_embeddings.jsonl"
clients: Set[WebSocket] = set()
db_lock = threading.Lock()
face_db: Dict[str, List[float]] = {}

def load_face_db_json(path: str = JSONL_PATH) -> Dict[str, List[float]]:
    db: Dict[str, List[float]] = {}
    if not os.path.exists(path):
        return db
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            name = obj.get("name")
            emb = obj.get("embedding")
            if name and isinstance(emb, list):
                db[name] = emb
    return db

async def broadcast(msg: dict):
    dead = []
    payload = json.dumps(msg)
    for c in list(clients):
        try:
            await c.send_text(payload)
        except Exception:
            dead.append(c)
    for c in dead:
        clients.discard(c)

async def tail_and_broadcast():
    last_size = os.path.getsize(JSONL_PATH) if os.path.exists(JSONL_PATH) else 0
    while True:
        try:
            size = os.path.getsize(JSONL_PATH) if os.path.exists(JSONL_PATH) else 0
            if size > last_size:
                with open(JSONL_PATH, "r", encoding="utf-8") as f:
                    f.seek(last_size)
                    new_chunk = f.read()
                last_size = size
                for line in new_chunk.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    name = obj.get("name")
                    emb = obj.get("embedding")
                    if name and isinstance(emb, list):
                        with db_lock:
                            face_db[name] = emb
                        await broadcast({"type": "add", "record": {"name": name, "embedding": emb}})
        except Exception:
            pass
        await asyncio.sleep(0.4)

# ---- NEW lifespan handler ----
@asynccontextmanager
async def lifespan(app: FastAPI):
    global face_db
    with db_lock:
        face_db = load_face_db_json()
    asyncio.create_task(tail_and_broadcast())
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/db")
def get_db():
    with db_lock:
        return JSONResponse(face_db)

@app.get("/jsonl")
def get_jsonl():
    if not os.path.exists(JSONL_PATH):
        return PlainTextResponse("", media_type="text/plain")
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        return PlainTextResponse(f.read(), media_type="text/plain")

@app.websocket("/updates")
async def ws_updates(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    with db_lock:
        await ws.send_text(json.dumps({"type": "full", "db": face_db}))
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        clients.discard(ws)


# ---- Auto-start server when run directly ----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("host_server:app", host="0.0.0.0", port=8000, reload=False, workers=1)
