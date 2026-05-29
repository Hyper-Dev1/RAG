from sqlmodel import SQLModel, Field, Session, create_engine, select, Relationship
from sqlalchemy import Column, Text, event, text
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from typing import Optional

class Publication(SQLModel, table=True):
    __tablename__ = "publication"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    class_id: int = Field(foreign_key="class.id")