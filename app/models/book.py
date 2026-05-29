from sqlmodel import SQLModel, Field, Session, create_engine, select, Relationship
from sqlalchemy import Column, Text, event, text
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from typing import Optional


class Book(SQLModel, table=True):
    __tablename__ = "book"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    publication_id: int = Field(foreign_key="publication.id")
