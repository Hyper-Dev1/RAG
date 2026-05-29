from sqlmodel import SQLModel, Session, create_engine
from sqlalchemy import text
from app.core.config import DB_URL
from app.models import *

engine = create_engine(DB_URL, echo=False, pool_pre_ping=True)

def init_db():
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    SQLModel.metadata.create_all(engine)

    with engine.connect() as conn:
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS paragraph_embedding_hnsw "
            "ON paragraph USING hnsw (embedding vector_cosine_ops)"
        ))
        
        # Add Full-Text Search column and index
        conn.execute(text(
            "ALTER TABLE paragraph ADD COLUMN IF NOT EXISTS fts_vector tsvector "
            "GENERATED ALWAYS AS (to_tsvector('english', content)) STORED"
        ))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS paragraph_fts_idx ON paragraph USING GIN (fts_vector)"
        ))
        conn.commit()

def get_session():
    with Session(engine) as session:
        yield session
