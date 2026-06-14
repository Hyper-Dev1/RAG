from pydantic import BaseModel
from typing import Optional

VALID_DIAGRAM_TYPES = [
    "flowchart", "sequenceDiagram", "classDiagram",
    "stateDiagram", "erDiagram", "gantt", "pie",
    "quadrantChart", "mindmap", "timeline", "gitgraph",
]

class DiagramRequest(BaseModel):
    query: str
    diagram_type: Optional[str] = None
    top_k: int = 5
    min_score: Optional[float] = None
    use_reranker: Optional[bool] = None
    use_mmr: Optional[bool] = None

class DiagramResponse(BaseModel):
    query: str
    diagram_type: str
    title: str
    mermaid_code: str
    source_paragraphs: int
    message: Optional[str] = None
