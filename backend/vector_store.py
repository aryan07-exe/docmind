from sqlalchemy.orm import Session
from auth.models import DocumentChunk


def add_chunks(
    db: Session,
    user_id: str,
    filename: str,
    chunks: list[str],
    embeddings: list[list[float]],
):
    db.query(DocumentChunk).filter(
        DocumentChunk.user_id == user_id,
        DocumentChunk.filename == filename,
    ).delete(synchronize_session=False)

    rows = [
        DocumentChunk(
            user_id=user_id,
            filename=filename,
            chunk_index=i,
            content=chunk,
            embedding=emb,
        )
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]
    db.add_all(rows)
    db.commit()
    return len(rows)


def similarity_search(db: Session, user_id: str, query_embedding: list[float], k: int = 5) -> list[str]:
    rows = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.user_id == user_id)
        .order_by(DocumentChunk.embedding.cosine_distance(query_embedding))
        .limit(k)
        .all()
    )
    return [row.content for row in rows]


def list_filenames(db: Session, user_id: str) -> list[str]:
    rows = (
        db.query(DocumentChunk.filename)
        .filter(DocumentChunk.user_id == user_id)
        .distinct()
        .all()
    )
    return [row[0] for row in rows]


def delete_document(db: Session, user_id: str, filename: str) -> int:
    deleted = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.user_id == user_id, DocumentChunk.filename == filename)
        .delete(synchronize_session=False)
    )
    db.commit()
    return deleted
