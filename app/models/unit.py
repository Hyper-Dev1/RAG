from sqlmodel import SQLModel, Field, Session, create_engine, select, Relationship
from sqlalchemy import Column, Text, event, text
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from typing import Optional

class Unit(SQLModel, table=True):
    __tablename__ = "unit"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    book_id: int = Field(foreign_key="book.id")
