import json
import re
import logging
from typing import Optional
from app.services.flashcard_service import retrieve_paragraphs
from app.utils.llm import ollama_chat
from app.schemas.diagram import DiagramResponse, VALID_DIAGRAM_TYPES

logger = logging.getLogger(__name__)

min_relevance_score = 0.45
def generate_diagram(
    query: str,
    diagram_type: Optional[str] = None,
    top_k: int = 5,
    min_score: float = 0.0,
    use_reranker: bool = True,
    use_mmr: bool = True,
) -> DiagramResponse:
    logger.info(f"Generating diagram for query: {query}, type: {diagram_type or 'auto'}")

    paragraphs = retrieve_paragraphs(
        query, top_k=top_k, min_score=min_score,
        use_reranker=use_reranker, use_mmr=use_mmr,
    )
    top_score = max(p["score"] for p in paragraphs) if paragraphs else 0
    
    if not paragraphs or top_score < min_relevance_score:
        logger.warning("No paragraphs retrieved for diagram generation")
        return DiagramResponse(
            query=query,
            diagram_type=diagram_type or "flowchart",
            title="No Content Available",
            mermaid_code="",
            source_paragraphs=0,
            message="This topic is not covered in your current textbook (Class 8). Try a topic from your syllabus."
        )

    combined_text = "\n\n".join([p["content"] for p in paragraphs])
    combined_text = combined_text[:6000]

    type_instruction = ""
    if diagram_type:
        type_instruction = f"Generate a {diagram_type} diagram."
    else:
        type_instruction = "Choose the most appropriate diagram type based on the content (e.g., flowchart, sequenceDiagram, classDiagram, stateDiagram, erDiagram, gantt, pie, quadrantChart, mindmap, timeline)."
    
    system_prompt = f"""You are a Mermaid.js diagram generator for school-level educational content.

    {type_instruction}

    Return ONLY valid JSON (no markdown, no explanation):
    {{
    "diagram_type": "...",
    "title": "...",
    "mermaid_code": "..."
    }}

    Diagram rules:
    - Scope: diagram ONLY the concept in the query — "{query}". Do not introduce related systems, alternatives, historical context, or comparisons unless the query explicitly asks for them
    - Nodes: include every node needed to fully represent the concept, but zero nodes beyond that scope
    - Labels: short and specific (3-5 words max per node/edge)
    - Depth: go as deep as the content requires, but only within the queried concept
    - Source: use ONLY information present in the provided content. Do not add outside knowledge
    - Syntax: valid Mermaid.js, no markdown fences, no comments
    - diagram_type must be one of: {', '.join(VALID_DIAGRAM_TYPES)}
    - title must reflect exactly what was asked, not a broader topic"""

    user_prompt = f"""Query: "{query}"

    Educational content:
    {combined_text}

    Task: Generate a Mermaid diagram scoped strictly to the query. 
    If the content covers more than the query asks, extract only the relevant portions.
    Do not represent anything not asked for in the query, even if the content mentions it."""
    
    try:
        response = ollama_chat(system_prompt, user_prompt, temperature=0)
        
        # Step 1: strip all markdown fences
        response = re.sub(r"```(?:json)?", "", response)
        response = re.sub(r"```", "", response)
        response = response.strip()
        
        # Step 2: extract JSON object if model added text before/after
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if not json_match:
            logger.error(f"No JSON object found in response: {response[:500]}")
            raise ValueError("LLM did not return a JSON object")
        
        data = json.loads(json_match.group())
        
        d_type = data.get("diagram_type", diagram_type or "flowchart")
        title = data.get("title", "Diagram")
        mermaid_code = data.get("mermaid_code", "")
        
        # Step 3: validate mermaid_code isn't empty
        if not mermaid_code:
            logger.error("LLM returned empty mermaid_code")
            raise ValueError("LLM returned empty mermaid_code")

        # Step 4: sanitize mermaid_code in case model wrapped it in fences anyway
        mermaid_code = re.sub(r"```(?:mermaid)?", "", mermaid_code)
        mermaid_code = re.sub(r"```", "", mermaid_code).strip()

        logger.info(f"Generated {d_type} diagram: {title}")
        return DiagramResponse(
            query=query,
            diagram_type=d_type,
            title=title,
            mermaid_code=mermaid_code,
            source_paragraphs=len(paragraphs),
        )

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        logger.error(f"LLM response was: {response[:500]}")
        raise ValueError(f"LLM returned invalid JSON: {e}")
    except Exception as e:
        logger.error(f"Error generating diagram: {e}")
        raise

  