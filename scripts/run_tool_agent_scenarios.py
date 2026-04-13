import asyncio
import json
from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parent.parent))
from llm import get_coach_response_with_trace, get_sgr_response_with_trace


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def build_cases() -> list[tuple[str, dict]]:
    return [
        (
            "initial_plan",
            {
                "user_profile": {
                    "goal": "hypertrophy",
                    "experience_level": "beginner",
                    "equipment": ["гантели", "турник"],
                    "restrictions": [],
                }
            },
        ),
        (
            "progress",
            {
                "user_profile": {
                    "goal": "strength",
                    "experience_level": "intermediate",
                    "equipment": ["штанга", "гантели", "скамья"],
                    "restrictions": [],
                },
                "current_session": {
                    "date": "2026-03-31",
                    "exercises": [
                        {
                            "name": "Жим лёжа",
                            "sets_completed": 3,
                            "sets_planned": 3,
                            "reps": "8,8,8",
                            "weight_kg": 70,
                            "rpe": 7,
                        },
                        {
                            "name": "Тяга в наклоне",
                            "sets_completed": 3,
                            "sets_planned": 3,
                            "reps": "10,10,10",
                            "weight_kg": 50,
                            "rpe": 6,
                        },
                    ],
                    "sleep_hours": 8,
                    "fatigue_level": 3,
                    "notes": "тренировка прошла уверенно",
                },
            },
        ),
        (
            "medical_risk",
            {
                "user_profile": {
                    "goal": "general",
                    "experience_level": "beginner",
                    "equipment": ["гантели"],
                    "restrictions": ["травма поясницы"],
                },
                "current_session": {
                    "date": "2026-03-31",
                    "exercises": [
                        {
                            "name": "Румынская тяга",
                            "sets_completed": 1,
                            "sets_planned": 3,
                            "reps": "10",
                            "weight_kg": 30,
                            "rpe": 8,
                        }
                    ],
                    "sleep_hours": 6,
                    "fatigue_level": 7,
                    "notes": "острая боль в пояснице во время движения",
                },
            },
        ),
        (
            "confirmation",
            {
                "user_profile": {
                    "goal": "strength",
                    "experience_level": "intermediate",
                    "equipment": ["штанга", "гантели", "скамья"],
                    "restrictions": ["дискомфорт в колене"],
                },
                "current_session": {
                    "date": "2026-03-31",
                    "exercises": [
                        {
                            "name": "Присед",
                            "sets_completed": 3,
                            "sets_planned": 3,
                            "reps": "6,6,6",
                            "weight_kg": 90,
                            "rpe": 7,
                        }
                    ],
                    "sleep_hours": 8,
                    "fatigue_level": 4,
                    "notes": "есть небольшой дискомфорт в колене, но без резкой боли",
                },
            },
        ),
    ]


async def main() -> None:
    report_lines = ["# Tool Calling Scenario Report", ""]

    for case_name, payload in build_cases():
        sgr_response, sgr_trace = await get_sgr_response_with_trace(payload)
        coach_response, coach_trace = await get_coach_response_with_trace(payload)

        case_dir = RESULTS_DIR / f"tool_trace_{case_name}"
        case_dir.mkdir(exist_ok=True)

        (case_dir / "request.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (case_dir / "sgr_response.json").write_text(
            json.dumps(sgr_response.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (case_dir / "coach_response.json").write_text(
            json.dumps(coach_response.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (case_dir / "sgr_trace.json").write_text(
            json.dumps(sgr_trace.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (case_dir / "coach_trace.json").write_text(
            json.dumps(coach_trace.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        report_lines.append(f"## {case_name}")
        report_lines.append("")
        report_lines.append(f"- mode: {sgr_response.mode.value}")
        report_lines.append(f"- refused: {sgr_response.final_recommendation.refused}")
        report_lines.append(f"- tools called: {', '.join(item.tool_name for item in sgr_trace.tool_calls)}")
        report_lines.append(f"- decision: {sgr_response.final_recommendation.decision}")
        report_lines.append("")

    (RESULTS_DIR / "tool_calling_report.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )


if __name__ == "__main__":
    asyncio.run(main())
