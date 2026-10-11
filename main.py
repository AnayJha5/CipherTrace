import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from neo4j import GraphDatabase
from groq import Groq

load_dotenv()

# ============================================================
# CIPHERTRACE — INVESTIGATIVE INTELLIGENCE BACKEND
# ============================================================

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

neo4j = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
)

groq = Groq(api_key=GROQ_API_KEY)

app = FastAPI(
    title="CipherTrace Investigative Intelligence Platform",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HELPERS
# ============================================================

def json_safe(value: Any) -> Any:
    """Convert Neo4j values into JSON-safe Python values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    # Neo4j Node
    if hasattr(value, "labels") and hasattr(value, "items"):
        return {
            "element_id": getattr(value, "element_id", None),
            "labels": list(value.labels),
            "properties": json_safe(dict(value.items())),
        }

    # Neo4j Relationship
    if hasattr(value, "type") and hasattr(value, "items"):
        return {
            "element_id": getattr(value, "element_id", None),
            "type": value.type,
            "properties": json_safe(dict(value.items())),
        }

    # Neo4j Path
    if hasattr(value, "nodes") and hasattr(value, "relationships"):
        return {
            "nodes": json_safe(list(value.nodes)),
            "relationships": json_safe(list(value.relationships)),
        }

    return str(value)


def run_query(query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    with neo4j.session() as session:
        records = session.run(query, params or {})
        return [json_safe(record.data()) for record in records]


def label_from_node(node: Any) -> str:
    labels = list(node.labels) if hasattr(node, "labels") else []
    return labels[0] if labels else "Entity"


def node_to_graph(node: Any) -> Dict[str, Any]:
    props = dict(node.items()) if hasattr(node, "items") else {}
    element_id = getattr(node, "element_id", str(props.get("id", "")))
    return {
        "id": element_id,
        "label": label_from_node(node),
        "name": (
            props.get("name")
            or props.get("person_name")
            or props.get("entity_name")
            or props.get("id")
            or element_id
        ),
        "properties": json_safe(props),
    }


def rel_to_graph(rel: Any) -> Dict[str, Any]:
    return {
        "id": getattr(rel, "element_id", ""),
        "source": getattr(rel, "start_node", None).element_id if getattr(rel, "start_node", None) else "",
        "target": getattr(rel, "end_node", None).element_id if getattr(rel, "end_node", None) else "",
        "type": getattr(rel, "type", "RELATED_TO"),
        "properties": json_safe(dict(rel.items())) if hasattr(rel, "items") else {},
    }


def ai_call(system: str, user: str) -> str:
    try:
        response = groq.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or "No analysis returned."
    except Exception as e:
        return f"AI unavailable right now: {e}"


# ============================================================
# MODELS
# ============================================================

class InvestigationRequest(BaseModel):
    query: str = Field(min_length=1)


class AIRequest(BaseModel):
    question: str = Field(min_length=1)


class EntityRequest(BaseModel):
    element_id: str = Field(min_length=1)


class PathRequest(BaseModel):
    source_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)


# ============================================================
# CORE / HEALTH
# ============================================================

@app.get("/")
def home():
    return {
        "product": "CipherTrace",
        "message": "Investigative Intelligence Platform",
        "version": "2.0.0",
    }


@app.get("/health")
def health():
    neo4j_status = "CONNECTED"
    try:
        neo4j.verify_connectivity()
    except Exception:
        neo4j_status = "OFFLINE"

    return {
        "backend": "OK",
        "neo4j": neo4j_status,
        "ai": "GROQ CONFIGURED" if GROQ_API_KEY else "GROQ KEY MISSING",
    }


# ============================================================
# COMMAND DASHBOARD
# ============================================================

@app.get("/dashboard")
def dashboard():
    queries = {
        "entities": """
            MATCH (n)
            RETURN count(n) AS count
        """,
        "relationships": """
            MATCH ()-[r]->()
            RETURN count(r) AS count
        """,
        "labels": """
            MATCH (n)
            UNWIND labels(n) AS label
            RETURN label, count(*) AS count
            ORDER BY count DESC
            LIMIT 12
        """,
        "relationship_types": """
            MATCH ()-[r]->()
            RETURN type(r) AS type, count(*) AS count
            ORDER BY count DESC
            LIMIT 12
        """,
    }

    try:
        entities = run_query(queries["entities"])[0]["count"]
        relationships = run_query(queries["relationships"])[0]["count"]
        labels = run_query(queries["labels"])
        relationship_types = run_query(queries["relationship_types"])

        return {
            "entities": entities,
            "relationships": relationships,
            # These are analytical indicators for the synthetic/demo dataset.
            "anomalies": 0,
            "active_cases": 1,
            "labels": labels,
            "relationship_types": relationship_types,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# GRAPH
# ============================================================

@app.get("/graph")
def graph(limit: int = 180):
    limit = max(10, min(limit, 500))

    query = f"""
        MATCH (a)-[r]->(b)
        RETURN a, r, b
        LIMIT {limit}
    """

    try:
        with neo4j.session() as session:
            records = session.run(query)

            nodes = {}
            edges = []

            for record in records:
                a = record["a"]
                b = record["b"]
                r = record["r"]

                na = node_to_graph(a)
                nb = node_to_graph(b)

                nodes[na["id"]] = na
                nodes[nb["id"]] = nb

                edges.append(rel_to_graph(r))

            return {
                "nodes": list(nodes.values()),
                "edges": edges,
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# ENTITY INTELLIGENCE
# ============================================================

@app.post("/entity")
def entity(request: EntityRequest):
    query = """
        MATCH (n)
        WHERE elementId(n) = $element_id
        OPTIONAL MATCH (n)-[r]-(connected)
        RETURN
            n,
            count(r) AS connection_count,
            collect(DISTINCT labels(connected))[..20] AS connected_labels
    """

    try:
        rows = run_query(query, {"element_id": request.element_id})
        if not rows:
            raise HTTPException(status_code=404, detail="Entity not found.")

        return rows[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# CONNECTION / PATH ANALYSIS
# ============================================================

@app.post("/find-connection")
def find_connection(request: PathRequest):
    query = """
        MATCH (a), (b)
        WHERE elementId(a) = $source_id AND elementId(b) = $target_id
        MATCH p = shortestPath((a)-[*..8]-(b))
        RETURN p
        LIMIT 1
    """

    try:
        with neo4j.session() as session:
            record = session.run(
                query,
                {
                    "source_id": request.source_id,
                    "target_id": request.target_id,
                },
            ).single()

        if not record:
            return {
                "found": False,
                "message": "No path found within 8 relationships.",
            }

        path = record["p"]

        return {
            "found": True,
            "path": {
                "nodes": [node_to_graph(n) for n in path.nodes],
                "edges": [rel_to_graph(r) for r in path.relationships],
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# ANOMALY CENTER
# ============================================================

@app.get("/anomalies")
def anomalies(limit: int = 20):
    """
    Lightweight explainable anomaly heuristics.
    These are NOT criminality judgments. They are graph-analytics
    indicators intended for the synthetic demonstration dataset.
    """

    limit = max(5, min(limit, 100))

    query = f"""
        MATCH (n)
        OPTIONAL MATCH (n)-[r]-()
        WITH n, count(r) AS degree
        WHERE degree >= 3
        RETURN
            elementId(n) AS element_id,
            labels(n) AS labels,
            properties(n) AS properties,
            degree,
            CASE
                WHEN degree >= 12 THEN "HIGH"
                WHEN degree >= 7 THEN "MEDIUM"
                ELSE "LOW"
            END AS severity
        ORDER BY degree DESC
        LIMIT {limit}
    """

    try:
        return {
            "basis": "Graph connectivity heuristic for synthetic/demo data",
            "items": run_query(query),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# TIMELINE
# ============================================================

@app.get("/timeline")
def timeline(limit: int = 30):
    """
    Attempts to find common date/time properties without assuming
    a specific dataset schema.
    """
    query = f"""
        MATCH (n)
        WITH n, properties(n) AS p
        UNWIND keys(p) AS k
        WITH n, k, p[k] AS value
        WHERE
            toLower(k) CONTAINS "date"
            OR toLower(k) CONTAINS "time"
            OR toLower(k) CONTAINS "timestamp"
        RETURN
            elementId(n) AS element_id,
            labels(n) AS labels,
            k AS date_field,
            toString(value) AS event_time,
            p AS properties
        LIMIT {limit}
    """

    try:
        return {"items": run_query(query)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# INVESTIGATION ENGINE
# ============================================================

@app.post("/investigate")
def investigate(request: InvestigationRequest):
    try:
        with neo4j.session() as session:
            records = session.run(request.query)
            results = [json_safe(record.data()) for record in records]

        database_context = str(results[:30])

        system = """
You are CipherTrace, an explainable investigative intelligence assistant.

Rules:
1. Use ONLY evidence supplied by the Neo4j results.
2. Never invent people, events, relationships, motives or facts.
3. Clearly distinguish observed graph evidence from analytical interpretation.
4. Treat anomaly/risk signals as analytical indicators, not proof of criminality.
5. Give investigators concise, structured findings.
6. When possible, identify entities, relationships, patterns and limitations.
"""

        prompt = f"""
INVESTIGATION QUERY:
{request.query}

NEO4J EVIDENCE:
{database_context}

Return:
1. INVESTIGATIVE FINDING
2. KEY ENTITIES
3. RELATIONSHIPS / PATTERNS
4. ANOMALY OR RISK SIGNALS
5. EVIDENCE BASIS
6. LIMITATIONS
7. NEXT INVESTIGATION STEP

Do not claim certainty where the data does not support it.
"""

        analysis = ai_call(system, prompt)

        return {
            "query": request.query,
            "result_count": len(results),
            "results": results,
            "analysis": analysis,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# AI INVESTIGATOR COPILOT
# ============================================================

@app.post("/ask-ai")
def ask_ai(request: AIRequest):
    """
    General copilot endpoint. The frontend can use this for
    investigator commands and case-level explanations.
    """

    system = """
You are CipherTrace Copilot inside an investigative intelligence
dashboard.

You help investigators understand graph evidence.

Never fabricate evidence. If no database evidence was provided,
say that the answer requires a graph/database query.

Use concise structured sections:
- Finding
- Evidence needed / evidence available
- Interpretation
- Suggested next investigation

Never present an analytical score as proof that a person is
criminal. This system is designed around synthetic/demo data
and human-in-the-loop investigation.
"""

    return {
        "question": request.question,
        "analysis": ai_call(system, request.question),
    }


# ============================================================
# CASE BRIEF
# ============================================================

@app.post("/case-brief")
def case_brief(request: InvestigationRequest):
    """
    Runs the supplied Cypher query and asks Groq for a structured
    investigation brief based strictly on returned evidence.
    """
    try:
        with neo4j.session() as session:
            records = session.run(request.query)
            results = [json_safe(record.data()) for record in records]

        prompt = f"""
Create a professional CipherTrace investigation brief from this
Neo4j evidence.

DATA:
{str(results[:50])}

Use these headings:
EXECUTIVE SUMMARY
KEY ENTITIES
KEY RELATIONSHIPS
ANALYTICAL SIGNALS
TIMELINE / EVENTS
EVIDENCE
LIMITATIONS
NEXT INVESTIGATION PATHS

Do not invent information.
"""

        analysis = ai_call(
            "You generate evidence-grounded investigation briefs.",
            prompt,
        )

        return {
            "result_count": len(results),
            "analysis": analysis,
            "evidence": results[:50],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# CLEAN SHUTDOWN
# ============================================================

@app.on_event("shutdown")
def shutdown():
    try:
        neo4j.close()
    except Exception:
        pass
