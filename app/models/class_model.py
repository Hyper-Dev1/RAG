from sqlmodel import SQLModel, Field, Session, create_engine, select, Relationship
from sqlalchemy import Column, Text, event, text
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from typing import Optional


class Class(SQLModel, table=True):
    __tablename__ = "class"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
