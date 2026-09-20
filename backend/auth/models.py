from sqlalchemy.dialects.postgresql import UUID
import uuid
from db import Base, get_embedding_dim
from sqlalchemy import Column, String, DateTime, Text, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import relationship
from datetime import datetime
from pgvector.sqlalchemy import Vector


class User(Base):
    __tablename__ = "user_details"

    user_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)


def gen_uuid():
    return str(uuid.uuid4())


class Chat(Base):
    __tablename__ = "chats"

    chat_id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, nullable=False, index=True)
    title = Column(String, default="New Chat")
    created_at = Column(DateTime, default=datetime.utcnow)

    messages = relationship("Message", back_populates="chat", cascade="all, delete")


class Message(Base):
    __tablename__ = "messages"

    message_id = Column(String, primary_key=True, default=gen_uuid)
    chat_id = Column(String, ForeignKey("chats.chat_id"))
    user_id = Column(String, nullable=False)
    role = Column(String)  # 'user' or 'assistant'
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    chat = relationship("Chat", back_populates="messages")


class DocumentChunk(Base):
    """Document text and its embedding in the same pgvector table."""

    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint("user_id", "filename", "chunk_index", name="uq_user_file_chunk"),
    )

    chunk_id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, nullable=False, index=True)
    filename = Column(String, nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(get_embedding_dim()), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
