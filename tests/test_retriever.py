import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from retriever import build_retrieval_query, format_retrieved_knowledge, retrieve_for_request


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[OK] {message}")


def build_knee_case() -> dict:
    return {
        "user_profile": {
            "goal": "hypertrophy",
            "experience_level": "beginner",
            "equipment": ["dumbbells", "bench", "bands"],
            "restrictions": ["knee pain", "no deep squats"],
        },
        "session_history": [],
        "current_session": None,
    }


def build_fatigue_case() -> dict:
    return {
        "user_profile": {
            "goal": "strength",
            "experience_level": "intermediate",
            "equipment": ["barbell", "rack", "bench"],
            "restrictions": [],
        },
        "current_session": {
            "date": "2026-03-21",
            "exercises": [
                {
                    "name": "Bench Press",
                    "sets_planned": 5,
                    "sets_completed": 3,
                    "reps": "5/5/4",
                    "weight_kg": 82.5,
                    "rpe": 9,
                }
            ],
            "sleep_hours": 5.5,
            "fatigue_level": 8,
            "notes": "Poor sleep, bar felt very heavy",
        },
    }


def check_query_builder() -> None:
    query = build_retrieval_query(build_fatigue_case())
    assert_true("strength" in query.lower(), "В retrieval query попадает goal")
    assert_true("bench press" in query.lower(), "В retrieval query попадает название упражнения")
    assert_true("reduce intensity" in query.lower(), "В retrieval query попадает сигнал про снижение нагрузки")


def check_restriction_retrieval() -> None:
    docs = retrieve_for_request(build_knee_case(), top_k=3)
    joined = " ".join((doc.get("title", "") + " " + doc.get("content", "")).lower() for doc in docs)
    assert_true(len(docs) > 0, "Ретривер возвращает документы для restriction-сценария")
    assert_true("knee" in joined or "squat" in joined, "Среди retrieved документов есть знания про knee pain / squat")


def check_fatigue_retrieval() -> None:
    docs = retrieve_for_request(build_fatigue_case(), top_k=3)
    formatted = format_retrieved_knowledge(docs).lower()
    assert_true(len(docs) > 0, "Ретривер возвращает документы для fatigue-сценария")
    assert_true("reduce intensity" in formatted or "reduce volume" in formatted, "Retrieved knowledge содержит рекомендации по снижению нагрузки")


def main() -> None:
    check_query_builder()
    check_restriction_retrieval()
    check_fatigue_retrieval()
    print("Все проверки retriever.py пройдены успешно.")


if __name__ == "__main__":
    main()
