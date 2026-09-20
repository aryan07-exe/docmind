from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import io
from embedder import get_embeddings
from loader import extract_text, chunk_text
from graph import app_graph
from db import init_db, SessionLocal
from auth.models import Chat, Message
from auth.routes import router as auth_router
from vector_store import add_chunks, list_filenames, delete_document as delete_chunks

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)


@app.on_event("startup")
def create_tables_if_not_exist():
    init_db()


@app.post("/upload")
async def upload(user_id: str, file: UploadFile = File(...)):
    db = SessionLocal()
    try:
        data = await file.read()
        file_stream = io.BytesIO(data)

        text = extract_text(file_stream, file.filename)
        chunks = chunk_text(text)
        embeddings = get_embeddings(chunks)
        stored = add_chunks(db, user_id, file.filename, chunks, embeddings)

        return {"status": "ok", "chunks": stored}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


class NewChatRequest(BaseModel):
    user_id: str


class AskInChat(BaseModel):
    user_id: str
    chat_id: str
    question: str


@app.post("/chat/new")
def new_chat(req: NewChatRequest):
    db = SessionLocal()
    try:
        chat = Chat(user_id=req.user_id)
        db.add(chat)
        db.commit()
        db.refresh(chat)
        return {
            "chat_id": chat.chat_id,
            "title": chat.title,
            "created_at": chat.created_at,
        }
    finally:
        db.close()


@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "DocMind API is running"}


@app.get("/chat/list")
def list_chats(user_id: str):
    db = SessionLocal()
    try:
        chats = (
            db.query(Chat)
            .filter(Chat.user_id == user_id)
            .order_by(Chat.created_at.desc())
            .all()
        )
        return [
            {
                "chat_id": c.chat_id,
                "title": c.title,
                "created_at": c.created_at,
            }
            for c in chats
        ]
    finally:
        db.close()


@app.get("/chat/{chat_id}")
def get_chat_messages(chat_id: str):
    db = SessionLocal()
    try:
        msgs = (
            db.query(Message)
            .filter(Message.chat_id == chat_id)
            .order_by(Message.created_at)
            .all()
        )
        return [
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.created_at,
            }
            for m in msgs
        ]
    finally:
        db.close()


@app.post("/chat/ask")
def ask_question(req: AskInChat):
    db = SessionLocal()
    try:
        user_msg = Message(
            chat_id=req.chat_id,
            user_id=req.user_id,
            role="user",
            content=req.question,
        )
        db.add(user_msg)
        db.commit()

        result = app_graph.invoke({
            "user_id": req.user_id,
            "question": req.question,
        })
        answer = result["answer"]

        ai_msg = Message(
            chat_id=req.chat_id,
            user_id=req.user_id,
            role="assistant",
            content=answer,
        )
        db.add(ai_msg)
        db.commit()

        return {
            "answer": answer,
            "context": result["context"],
        }
    finally:
        db.close()


@app.get("/documents")
async def list_documents(user_id: str):
    db = SessionLocal()
    try:
        return {"documents": list_filenames(db, user_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.delete("/documents/delete")
async def delete_document(user_id: str, filename: str):
    db = SessionLocal()
    try:
        deleted = delete_chunks(db, user_id, filename)
        if not deleted:
            return {"status": "not_found", "filename": filename}
        return {
            "status": "deleted",
            "filename": filename,
            "deleted_chunks": deleted,
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.delete("/chat/delete")
def delete_chat(user_id: str, chat_id: str):
    db = SessionLocal()
    try:
        chat = db.query(Chat).filter(
            Chat.chat_id == chat_id,
            Chat.user_id == user_id,
        ).first()
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        db.delete(chat)
        db.commit()
        return {"status": "deleted", "chat_id": chat_id}
    finally:
        db.close()


class RenameChatRequest(BaseModel):
    user_id: str
    chat_id: str
    new_title: str


@app.put("/chat/rename")
def rename_chat(req: RenameChatRequest):
    db = SessionLocal()
    try:
        chat = db.query(Chat).filter(
            Chat.chat_id == req.chat_id,
            Chat.user_id == req.user_id,
        ).first()
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        chat.title = req.new_title
        db.commit()
        db.refresh(chat)
        return {
            "status": "renamed",
            "chat_id": chat.chat_id,
            "new_title": chat.title,
        }
    finally:
        db.close()


FRONTEND_DIR = Path(__file__).resolve().parent / "frontend_build"

if FRONTEND_DIR.is_dir():
    static_dir = FRONTEND_DIR / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=static_dir), name="frontend-static")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        file_path = FRONTEND_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIR / "index.html")
