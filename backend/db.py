from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()


def get_database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if url:
        return url

    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD", "")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db_name = os.getenv("POSTGRES_DB")

    if not user or not db_name:
        raise RuntimeError(
            "Set DATABASE_URL or POSTGRES_USER, POSTGRES_PASSWORD, "
            "POSTGRES_HOST, POSTGRES_PORT, and POSTGRES_DB."
        )

    return f"postgresql://{user}:{password}@{host}:{port}/{db_name}"


def get_embedding_dim() -> int:
    return int(os.getenv("EMBEDDING_DIM", "2048"))


DATABASE_URL = get_database_url()

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Enable pgvector and create tables on first run."""
    from auth import models  # noqa: F401 — register ORM models

    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    Base.metadata.create_all(bind=engine)
