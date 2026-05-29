from sqlmodel import SQLModel, Field, Session, create_engine, select, Relationship
from sqlalchemy import Column, Text, event, text
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from typing import Optional


class Section(SQLModel, table=True):
    __tablename__ = "section"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str  # e.g. "1.1", "1.2"
    lesson_id: int = Field(foreign_key="lesson.id")
