from datetime import date
from typing import Any

from coach.models import CoachRequest


def build_coach_request_from_locust(content: str, extra_body: dict[str, Any]) -> CoachRequest:
    scenario = extra_body.get("scenario", "initial_plan")
    temperature = float(extra_body.get("temperature", 0.3))

    if scenario == "adaptation":
        return CoachRequest(
            user_profile={
                "goal": "general",
                "experience_level": "beginner",
                "equipment": ["гантели", "турник", "коврик"],
                "restrictions": [],
            },
            session_history=[],
            current_session={
                "date": str(date.today()),
                "exercises": [
                    {
                        "name": "Приседания",
                        "sets_planned": 3,
                        "sets_completed": 3,
                        "reps": "10/10/8",
                        "weight_kg": None,
                        "rpe": 8,
                    }
                ],
                "sleep_hours": 6.0,
                "fatigue_level": 7,
                "notes": content,
            },
            temperature=temperature,
        )

    return CoachRequest(
        user_profile={
            "goal": "general",
            "experience_level": "beginner",
            "equipment": ["гантели", "турник", "коврик"],
            "restrictions": [],
        },
        session_history=[],
        current_session=None,
        temperature=temperature,
    )
