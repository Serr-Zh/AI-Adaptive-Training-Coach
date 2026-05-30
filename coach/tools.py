import json
from dataclasses import dataclass
from typing import Any, Callable, Type

from pydantic import BaseModel

from coach.models import (
    AgentExecutionTrace,
    AssessMedicalRiskInput,
    AssessMedicalRiskOutput,
    AssessRestrictionsInput,
    AssessRestrictionsOutput,
    AssessTrainingLoadInput,
    AssessTrainingLoadOutput,
    BuildTrainingContextInput,
    BuildTrainingContextOutput,
    RequestConfirmationInput,
    RequestConfirmationOutput,
    RetrieveTrainingKnowledgeInput,
    RetrieveTrainingKnowledgeOutput,
    RetrievedKnowledgeItem,
    ToolCallRecord,
    ToolRequestSnapshot,
)
from coach.retriever import build_retrieval_query, format_retrieved_knowledge, retrieve_for_request


ACUTE_MEDICAL_KEYWORDS = {
    "острая боль",
    "резкая боль",
    "сильная боль",
    "травма",
    "injury",
    "sharp pain",
    "acute pain",
    "прострел",
    "заклинило",
    "не могу двигаться",
    "мешает движению",
    "ухудшение состояния",
    "онемение",
}

AMBIGUOUS_DISCOMFORT_KEYWORDS = {
    "дискомфорт",
    "побаливает",
    "ноет",
    "тянет",
    "неприятные ощущения",
    "soreness",
    "mild pain",
    "discomfort",
}

RESTRICTION_PATTERNS = {
    "knee": {
        "keywords": ["колен", "knee"],
        "cautions": [
            "Ограничить глубокие приседания под высокой нагрузкой",
            "С осторожностью использовать выпады и плиометрику",
        ],
    },
    "lower_back": {
        "keywords": ["поясниц", "спин", "lower back", "back pain"],
        "cautions": [
            "Избегать тяжёлых тяг и осевой нагрузки при симптомах",
            "Снизить нагрузку на тазовый шарнир и контролировать технику",
        ],
    },
    "shoulder": {
        "keywords": ["плеч", "shoulder"],
        "cautions": [
            "С осторожностью использовать жимы над головой",
            "Избегать провоцирующей амплитуды в жимовых движениях",
        ],
    },
    "neck": {
        "keywords": ["ше", "neck"],
        "cautions": [
            "Избегать выраженной осевой нагрузки на шею",
            "Снизить нагрузку в упражнениях с жёсткой фиксацией шеи",
        ],
    },
}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: Type[BaseModel]
    output_model: Type[BaseModel]
    handler: Callable[[BaseModel], BaseModel]


def _request_to_snapshot(data: dict[str, Any] | ToolRequestSnapshot) -> ToolRequestSnapshot:
    if isinstance(data, ToolRequestSnapshot):
        return data
    return ToolRequestSnapshot.model_validate(data)


def _exercise_names(snapshot: ToolRequestSnapshot) -> list[str]:
    session = snapshot.current_session or (snapshot.session_history[-1] if snapshot.session_history else None)
    if not session:
        return []
    return [exercise.name for exercise in session.exercises]


def _notes_text(snapshot: ToolRequestSnapshot) -> str:
    parts: list[str] = []
    if snapshot.user_profile.restrictions:
        parts.extend(snapshot.user_profile.restrictions)
    if snapshot.current_session and snapshot.current_session.notes:
        parts.append(snapshot.current_session.notes)
    if snapshot.session_history:
        last = snapshot.session_history[-1]
        if last.notes:
            parts.append(last.notes)
    return " | ".join(parts).lower()


def _contains_any(text: str, keywords: set[str]) -> list[str]:
    lowered = text.lower()
    return [kw for kw in keywords if kw in lowered]


def _summarize_latest_session(snapshot: ToolRequestSnapshot) -> str | None:
    session = snapshot.current_session or (snapshot.session_history[-1] if snapshot.session_history else None)
    if session is None:
        return None

    exercise_fragments: list[str] = []
    for exercise in session.exercises[:4]:
        exercise_fragments.append(
            f"{exercise.name}: {exercise.sets_completed}/{exercise.sets_planned} подходов, reps={exercise.reps}, rpe={exercise.rpe}"
        )

    extras: list[str] = []
    if session.sleep_hours is not None:
        extras.append(f"сон {session.sleep_hours} ч")
    if session.fatigue_level is not None:
        extras.append(f"усталость {session.fatigue_level}/10")
    if session.notes:
        extras.append(f"заметки: {session.notes}")

    joined_exercises = "; ".join(exercise_fragments) if exercise_fragments else "без упражнений"
    joined_extras = "; ".join(extras)
    return f"{session.date}: {joined_exercises}" + (f"; {joined_extras}" if joined_extras else "")


def build_training_context(args: BuildTrainingContextInput) -> BuildTrainingContextOutput:
    snapshot = args.request
    mode = "adaptation" if (snapshot.session_history or snapshot.current_session) else "initial_plan"
    goal_value = getattr(snapshot.user_profile.goal, "value", str(snapshot.user_profile.goal))
    level_value = getattr(snapshot.user_profile.experience_level, "value", str(snapshot.user_profile.experience_level))
    equipment_summary = ", ".join(snapshot.user_profile.equipment) if snapshot.user_profile.equipment else "без оборудования"
    restrictions_summary = ", ".join(snapshot.user_profile.restrictions) if snapshot.user_profile.restrictions else "нет ограничений"

    return BuildTrainingContextOutput(
        mode=mode,
        brief_goal=goal_value,
        experience_level=level_value,
        equipment_summary=equipment_summary,
        restrictions_summary=restrictions_summary,
        has_history=bool(snapshot.session_history),
        has_current_session=snapshot.current_session is not None,
        latest_session_excerpt=_summarize_latest_session(snapshot),
        history_size=len(snapshot.session_history),
    )


def retrieve_training_knowledge(args: RetrieveTrainingKnowledgeInput) -> RetrieveTrainingKnowledgeOutput:
    snapshot_dict = args.request.model_dump()
    documents = retrieve_for_request(snapshot_dict, top_k=args.top_k)
    query = build_retrieval_query(snapshot_dict)

    normalized_docs = [
        RetrievedKnowledgeItem(
            title=str(item.get("title", "")),
            category=str(item.get("category", "general")),
            tags=[str(tag) for tag in item.get("tags", [])],
            content=str(item.get("content", "")),
            score=float(item.get("score", 0.0)),
        )
        for item in documents
    ]

    return RetrieveTrainingKnowledgeOutput(
        query=query,
        documents=normalized_docs,
        formatted_knowledge=format_retrieved_knowledge(documents),
    )


def assess_restrictions(args: AssessRestrictionsInput) -> AssessRestrictionsOutput:
    snapshot = args.request
    restrictions = snapshot.user_profile.restrictions
    if not restrictions:
        return AssessRestrictionsOutput(
            restrictions_present=False,
            limiting_factors=[],
            restriction_impact_summary="Явные ограничения пользователя отсутствуют.",
            suggested_exercise_cautions=[],
        )

    joined = " ".join(restrictions).lower()
    limiting_factors = [item for item in restrictions]
    cautions: list[str] = []

    for pattern in RESTRICTION_PATTERNS.values():
        if any(keyword in joined for keyword in pattern["keywords"]):
            cautions.extend(pattern["cautions"])

    summary = "Ограничения нужно учитывать при выборе упражнений и объёма нагрузки."
    if cautions:
        summary += " Требуются безопасные модификации упражнений и осторожное повышение нагрузки."

    return AssessRestrictionsOutput(
        restrictions_present=True,
        limiting_factors=limiting_factors,
        restriction_impact_summary=summary,
        suggested_exercise_cautions=list(dict.fromkeys(cautions)),
    )


def assess_training_load(args: AssessTrainingLoadInput) -> AssessTrainingLoadOutput:
    snapshot = args.request
    session = snapshot.current_session or (snapshot.session_history[-1] if snapshot.session_history else None)
    if session is None:
        return AssessTrainingLoadOutput(
            progress_detected=False,
            supporting_facts=[],
            recommended_progression=None,
            overload_detected=False,
            overload_signals=[],
            recommended_adjustment=None,
            session_assessment="История и текущая тренировка отсутствуют, оценивается только стартовый план.",
        )

    overload_signals: list[str] = []
    supporting_facts: list[str] = []
    all_completed_low_rpe = True

    for exercise in session.exercises:
        if exercise.sets_completed < exercise.sets_planned:
            overload_signals.append(
                f"{exercise.name}: выполнено {exercise.sets_completed} из {exercise.sets_planned} подходов"
            )
            all_completed_low_rpe = False

        if exercise.rpe is None or exercise.rpe > 7:
            all_completed_low_rpe = False

        if exercise.rpe is not None and exercise.rpe > 9:
            overload_signals.append(f"{exercise.name}: высокий RPE {exercise.rpe}")

        if exercise.rpe is not None and exercise.rpe <= 7 and exercise.sets_completed == exercise.sets_planned:
            supporting_facts.append(
                f"{exercise.name}: все подходы выполнены при RPE {exercise.rpe}"
            )

    if session.sleep_hours is not None and session.sleep_hours < 6:
        overload_signals.append(f"Недостаток сна: {session.sleep_hours} ч")
        all_completed_low_rpe = False

    if session.fatigue_level is not None and session.fatigue_level > 7:
        overload_signals.append(f"Высокая усталость: {session.fatigue_level}/10")
        all_completed_low_rpe = False

    overload_detected = bool(overload_signals)
    progress_detected = bool(supporting_facts) and not overload_detected and all_completed_low_rpe

    recommended_adjustment = None
    if overload_detected:
        if any("RPE" in signal for signal in overload_signals):
            recommended_adjustment = "reduce_intensity"
        else:
            recommended_adjustment = "reduce_volume"

    recommended_progression = None
    if progress_detected:
        recommended_progression = (
            "Можно повысить нагрузку на 2.5-5% или добавить 1 повторение в основных упражнениях."
        )

    session_assessment_parts: list[str] = []
    if overload_detected:
        session_assessment_parts.append("Есть признаки перегрузки")
    elif progress_detected:
        session_assessment_parts.append("Есть признаки готовности к небольшой прогрессии")
    else:
        session_assessment_parts.append("Явных сигналов для прогрессии нет, уместен консервативный подход")

    if session.sleep_hours is not None:
        session_assessment_parts.append(f"сон {session.sleep_hours} ч")
    if session.fatigue_level is not None:
        session_assessment_parts.append(f"усталость {session.fatigue_level}/10")

    return AssessTrainingLoadOutput(
        progress_detected=progress_detected,
        supporting_facts=supporting_facts,
        recommended_progression=recommended_progression,
        overload_detected=overload_detected,
        overload_signals=overload_signals,
        recommended_adjustment=recommended_adjustment,
        session_assessment="; ".join(session_assessment_parts),
    )


def assess_medical_risk(args: AssessMedicalRiskInput) -> AssessMedicalRiskOutput:
    snapshot = args.request
    text = _notes_text(snapshot)
    matched = _contains_any(text, ACUTE_MEDICAL_KEYWORDS)

    refusal_required = bool(matched)
    refuse_reason = None
    if refusal_required:
        refuse_reason = (
            "Обнаружены признаки острой боли, травмы или иного медицинского риска. "
            "Нужна очная оценка состояния, а не тренировочная рекомендация."
        )

    risk_signals = [f"Обнаружен маркер: {item}" for item in matched]

    return AssessMedicalRiskOutput(
        medical_risk_detected=refusal_required,
        risk_signals=risk_signals,
        refusal_required=refusal_required,
        refuse_reason=refuse_reason,
    )


def request_confirmation(args: RequestConfirmationInput) -> RequestConfirmationOutput:
    snapshot = args.request
    text = _notes_text(snapshot)
    ambiguous = _contains_any(text, AMBIGUOUS_DISCOMFORT_KEYWORDS)

    if args.medical_risk_detected:
        return RequestConfirmationOutput(
            confirmation_required=False,
            confirmation_reason=None,
            safe_default_action="refuse",
        )

    if ambiguous:
        return RequestConfirmationOutput(
            confirmation_required=True,
            confirmation_reason=(
                "Во входе есть неоднозначные жалобы на дискомфорт без явного описания острой травмы. "
                "Без подтверждения состояния нельзя повышать нагрузку."
            ),
            safe_default_action="maintain",
        )

    return RequestConfirmationOutput(
        confirmation_required=False,
        confirmation_reason=None,
        safe_default_action="maintain",
    )


TOOL_SPECS: dict[str, ToolSpec] = {
    "build_training_context": ToolSpec(
        name="build_training_context",
        description="Нормализует входные данные, определяет режим работы и кратко summarises контекст пользователя.",
        input_model=BuildTrainingContextInput,
        output_model=BuildTrainingContextOutput,
        handler=build_training_context,
    ),
    "retrieve_training_knowledge": ToolSpec(
        name="retrieve_training_knowledge",
        description="Извлекает из локальной базы знаний вспомогательные тренировочные знания по текущему запросу.",
        input_model=RetrieveTrainingKnowledgeInput,
        output_model=RetrieveTrainingKnowledgeOutput,
        handler=retrieve_training_knowledge,
    ),
    "assess_restrictions": ToolSpec(
        name="assess_restrictions",
        description="Анализирует ограничения пользователя и возвращает факторы, влияющие на выбор упражнений.",
        input_model=AssessRestrictionsInput,
        output_model=AssessRestrictionsOutput,
        handler=assess_restrictions,
    ),
    "assess_training_load": ToolSpec(
        name="assess_training_load",
        description="Оценивает признаки прогрессии и перегрузки по текущей тренировке и истории.",
        input_model=AssessTrainingLoadInput,
        output_model=AssessTrainingLoadOutput,
        handler=assess_training_load,
    ),
    "assess_medical_risk": ToolSpec(
        name="assess_medical_risk",
        description="Проверяет входные данные на признаки острой боли, травмы и других медицинских рисков.",
        input_model=AssessMedicalRiskInput,
        output_model=AssessMedicalRiskOutput,
        handler=assess_medical_risk,
    ),
    "request_confirmation": ToolSpec(
        name="request_confirmation",
        description="Определяет, нужно ли подтверждение пользователя перед потенциально рискованным повышением нагрузки.",
        input_model=RequestConfirmationInput,
        output_model=RequestConfirmationOutput,
        handler=request_confirmation,
    ),
}


def get_openai_tool_definitions() -> list[dict[str, Any]]:
    definitions: list[dict[str, Any]] = []
    for spec in TOOL_SPECS.values():
        definitions.append(
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.input_model.model_json_schema(),
                },
            }
        )
    return definitions


def execute_tool(tool_name: str, arguments: dict[str, Any]) -> BaseModel:
    if tool_name not in TOOL_SPECS:
        raise ValueError(f"Неизвестный инструмент: {tool_name}")

    spec = TOOL_SPECS[tool_name]
    validated_args = spec.input_model.model_validate(arguments)
    result = spec.handler(validated_args)
    spec.output_model.model_validate(result.model_dump())
    return result


def run_local_tool_pipeline(request_data: dict[str, Any]) -> tuple[dict[str, dict], AgentExecutionTrace]:
    snapshot = ToolRequestSnapshot.model_validate(request_data)
    trace = AgentExecutionTrace(tool_calls=[])
    outputs: dict[str, dict] = {}

    ordered_calls: list[tuple[str, dict[str, Any]]] = [
        ("build_training_context", {"request": snapshot.model_dump()}),
        ("retrieve_training_knowledge", {"request": snapshot.model_dump(), "top_k": 3}),
        ("assess_restrictions", {"request": snapshot.model_dump()}),
        ("assess_training_load", {"request": snapshot.model_dump()}),
        ("assess_medical_risk", {"request": snapshot.model_dump()}),
    ]

    for tool_name, arguments in ordered_calls:
        result = execute_tool(tool_name, arguments)
        outputs[tool_name] = result.model_dump()
        trace.tool_calls.append(
            ToolCallRecord(
                tool_name=tool_name,
                arguments=arguments,
                result=result.model_dump(),
                source="local_fallback",
            )
        )

    confirmation_args = {
        "request": snapshot.model_dump(),
        "medical_risk_detected": outputs["assess_medical_risk"]["medical_risk_detected"],
    }
    confirmation_result = execute_tool("request_confirmation", confirmation_args)
    outputs["request_confirmation"] = confirmation_result.model_dump()
    trace.tool_calls.append(
        ToolCallRecord(
            tool_name="request_confirmation",
            arguments=confirmation_args,
            result=confirmation_result.model_dump(),
            source="local_fallback",
        )
    )

    return outputs, trace


def dump_tool_result(model: BaseModel) -> str:
    return json.dumps(model.model_dump(), ensure_ascii=False)
