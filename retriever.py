import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

KB_PATH = Path(__file__).resolve().parent / "data" / "knowledge_base.json"
TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9_+-]+")
STOPWORDS = {
    "и", "или", "в", "во", "на", "по", "для", "с", "со", "к", "из", "под", "при",
    "the", "a", "an", "and", "or", "of", "to", "for", "with", "in", "on", "at",
    "есть", "нет", "что", "как", "это", "его", "ее", "их", "is", "are"
}


def _normalize_text(text: str) -> str:
    text = text.lower().replace("ё", "е")
    replacements = {
        "мышечную массу": "hypertrophy",
        "набрать массу": "hypertrophy",
        "массу": "hypertrophy",
        "силу": "strength",
        "силовые": "strength",
        "выносливость": "endurance",
        "общая подготовка": "general",
        "общее здоровье": "general",
        "жим лежа": "bench press",
        "жим лёжа": "bench press",
        "присед": "squat",
        "становая тяга": "deadlift",
        "тяга": "deadlift",
        "гантели": "dumbbells",
        "штанга": "barbell",
        "турник": "pull-up bar",
        "скамья": "bench",
        "силовая рама": "rack",
        "кольне": "knee",
        "колене": "knee",
        "колено": "knee",
        "пояснице": "lower back",
        "поясница": "lower back",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def tokenize(text: str) -> list[str]:
    normalized = _normalize_text(text)
    tokens = [token for token in TOKEN_RE.findall(normalized) if token not in STOPWORDS and len(token) > 1]
    return tokens


@lru_cache(maxsize=1)
def load_knowledge_base() -> list[dict[str, Any]]:
    if not KB_PATH.exists():
        return []
    with KB_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("knowledge_base.json должен содержать массив объектов")
    return data


def _document_text(doc: dict[str, Any]) -> str:
    parts = [
        str(doc.get("title", "")),
        str(doc.get("content", "")),
        " ".join(map(str, doc.get("tags", []))),
        str(doc.get("category", "")),
    ]
    return "\n".join(parts)


def _score_document(query_tokens: set[str], query_text: str, doc: dict[str, Any]) -> float:
    content = _document_text(doc)
    doc_tokens = set(tokenize(content))
    if not doc_tokens:
        return 0.0

    overlap = query_tokens & doc_tokens
    score = float(len(overlap))

    title_tokens = set(tokenize(str(doc.get("title", ""))))
    tag_tokens = set(tokenize(" ".join(map(str, doc.get("tags", [])))))
    category_tokens = set(tokenize(str(doc.get("category", ""))))

    score += 1.5 * len(query_tokens & title_tokens)
    score += 1.2 * len(query_tokens & tag_tokens)
    score += 0.8 * len(query_tokens & category_tokens)

    lowered_query = _normalize_text(query_text)
    lowered_content = _normalize_text(content)
    if lowered_query and lowered_query in lowered_content:
        score += 3.0

    return score


def build_retrieval_query(request_data: dict[str, Any]) -> str:
    profile = request_data.get("user_profile", {}) or {}
    history = request_data.get("session_history", []) or []
    current = request_data.get("current_session") or {}

    parts: list[str] = []

    goal = profile.get("goal")
    if goal:
        parts.append(f"goal {goal}")

    level = profile.get("experience_level")
    if level:
        parts.append(f"experience {level}")

    equipment = profile.get("equipment", []) or []
    if equipment:
        parts.append("equipment " + " ".join(map(str, equipment)))

    restrictions = profile.get("restrictions", []) or []
    if restrictions:
        parts.append("restrictions " + " ".join(map(str, restrictions)))

    session_candidates = []
    if current:
        session_candidates.append(current)
    if history:
        session_candidates.append(history[-1])

    for session in session_candidates:
        exercises = session.get("exercises", []) or []
        exercise_names = [str(ex.get("name", "")) for ex in exercises if ex.get("name")]
        if exercise_names:
            parts.append("exercises " + " ".join(exercise_names))

        notes = session.get("notes")
        if notes:
            parts.append(str(notes))

        sleep = session.get("sleep_hours")
        if isinstance(sleep, (int, float)) and sleep < 6:
            parts.append("poor sleep reduce volume reduce intensity")

        fatigue = session.get("fatigue_level")
        if isinstance(fatigue, int) and fatigue > 7:
            parts.append("high fatigue overload reduce intensity")

        for ex in exercises:
            planned = ex.get("sets_planned")
            completed = ex.get("sets_completed")
            rpe = ex.get("rpe")
            if isinstance(planned, int) and isinstance(completed, int) and completed < planned:
                parts.append("underperformance reduce volume")
            if isinstance(rpe, (int, float)) and rpe >= 9:
                parts.append("high rpe overload deload reduce intensity")
            if isinstance(rpe, (int, float)) and rpe <= 7 and planned == completed:
                parts.append("progression increase load")

    return " ".join(parts).strip()


def retrieve_documents(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    if not query:
        return []

    docs = load_knowledge_base()
    if not docs:
        return []

    query_tokens = set(tokenize(query))
    if not query_tokens:
        return []

    scored_docs = []
    for doc in docs:
        score = _score_document(query_tokens, query, doc)
        if score > 0:
            enriched = dict(doc)
            enriched["score"] = round(score, 3)
            scored_docs.append(enriched)

    scored_docs.sort(key=lambda item: item["score"], reverse=True)
    return scored_docs[:top_k]


def retrieve_for_request(request_data: dict[str, Any], top_k: int = 3) -> list[dict[str, Any]]:
    query = build_retrieval_query(request_data)
    return retrieve_documents(query, top_k=top_k)


def format_retrieved_knowledge(docs: list[dict[str, Any]]) -> str:
    if not docs:
        return ""

    chunks = []
    for idx, doc in enumerate(docs, start=1):
        title = doc.get("title", f"Knowledge {idx}")
        category = doc.get("category", "general")
        content = doc.get("content", "")
        tags = ", ".join(map(str, doc.get("tags", [])))
        chunks.append(
            f"[{idx}] {title}\n"
            f"category: {category}\n"
            f"tags: {tags}\n"
            f"content: {content}"
        )
    return "\n\n".join(chunks)
