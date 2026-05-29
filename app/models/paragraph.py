from sqlmodel import SQLModel, Field, Session, create_engine, select, Relationship
from sqlalchemy import Column, Text, event, text
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from typing import Optional

class Paragraph(SQLModel, table=True):
    __tablename__ = "paragraph"
    id: Optional[int] = Field(default=None, primary_key=True)
    section_id: int = Field(foreign_key="section.id")
    content: str = Field(sa_column=Column(Text))
    embedding: list = Field(default=None, sa_column=Column(Vector(384)))
    # Rich metadata stored as JSONB for fast key-based queries in PG
    meta: dict = Field(default=None, sa_column=Column("metadata", JSONB))
