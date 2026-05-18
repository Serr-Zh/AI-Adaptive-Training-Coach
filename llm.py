import json
import logging
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langfuse import get_client, propagate_attributes
from langfuse.openai import AsyncOpenAI
from openai import BadRequestError

from models import (
    AgentExecutionTrace,
    CoachResponse,
    CoachSGRResponse,
    ToolCallRecord,
    sgr_to_coach_response,
)
from prompts import (
    FINAL_SYSTEM_PROMPT,
    TOOL_SYSTEM_PROMPT,
    build_final_user_prompt,
    build_tool_user_prompt,
)
from tools import (
    dump_tool_result,
    execute_tool,
    get_openai_tool_definitions,
    run_local_tool_pipeline,
)


load_dotenv(Path(__file__).resolve().parent / ".env")

logger = logging.getLogger(__name__)

_openai_client: AsyncOpenAI | None = None


def get_training_llm_client() -> AsyncOpenAI:
    global _openai_client

    if _openai_client is None:
        api_key = os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL")

        if not api_key or not base_url:
            raise RuntimeError(
                "Не заданы LLM_API_KEY или LLM_BASE_URL — проверь .env файл"
            )

        _openai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    return _openai_client


def _get_max_tokens() -> int:
    raw_value = os.getenv("LLM_MAX_TOKENS", "2048")

    try:
        max_tokens = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(
            f"Некорректное значение LLM_MAX_TOKENS={raw_value!r}. "
            "Ожидалось целое число."
        ) from exc

    if max_tokens <= 0:
        raise RuntimeError(
            f"Некорректное значение LLM_MAX_TOKENS={raw_value!r}. "
            "Значение должно быть больше нуля."
        )

    return max_tokens


def _flush_langfuse() -> None:
    try:
        get_client().flush()
    except Exception as exc:
        logger.warning("Не удалось выполнить Langfuse flush: %s", exc)


def extract_json_from_model_answer(raw_answer: str) -> str:
    if not isinstance(raw_answer, str) or not raw_answer.strip():
        raise ValueError(
            "Ожидалась непустая строка с JSON-ответом модели, "
            f"получено: {raw_answer!r}"
        )

    raw_answer = re.sub(r"```(?:json)?\s*", "", raw_answer).strip()
    raw_answer = raw_answer.replace("```", "").strip()

    json_start = raw_answer.find("{")
    json_end = raw_answer.rfind("}")

    if json_start == -1 or json_end == -1 or json_end < json_start:
        raise ValueError(f"JSON не найден в ответе модели: {raw_answer[:300]}")

    return raw_answer[json_start : json_end + 1]


def _as_list(value):
    if value is None:
        return []

    if isinstance(value, list):
        return value

    return [value]


def _as_bool(value, default=False):
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "да"}

    if value is None:
        return default

    return bool(value)


def _as_str(value, default=""):
    if value is None:
        return default

    if isinstance(value, str):
        return value.strip()

    return str(value)


def _normalize_sgr_exercise_changes(value):
    if not isinstance(value, list):
        return []

    normalized = []

    for item in value:
        if not isinstance(item, dict):
            normalized.append(
                {
                    "exercise_name": "Не указано",
                    "change_type": "modify",
                    "details": str(item),
                }
            )
            continue

        exercise_name = (
            item.get("exercise_name")
            or item.get("exercise")
            or item.get("name")
            or "Не указано"
        )

        change_type = item.get("change_type") or item.get("type")
        details = item.get("details") or item.get("description")

        if not change_type:
            if "new_weight_kg" in item:
                change_type = "increase_weight"
            elif "remove" in item:
                change_type = "remove_exercise"
            elif "replace_with" in item:
                change_type = "replace_exercise"
            else:
                change_type = "modify"

        if not details:
            if "new_weight_kg" in item:
                details = f"Новый рабочий вес: {item['new_weight_kg']} кг"
            elif "replace_with" in item:
                details = f"Заменить на {item['replace_with']}"
            elif "remove" in item:
                details = "Убрать упражнение"
            else:
                extra_parts = []

                for key, val in item.items():
                    if key not in {
                        "exercise_name",
                        "exercise",
                        "name",
                        "change_type",
                        "type",
                    }:
                        extra_parts.append(f"{key}: {val}")

                details = "; ".join(extra_parts) if extra_parts else "Без деталей"

        normalized.append(
            {
                "exercise_name": str(exercise_name),
                "change_type": str(change_type),
                "details": str(details),
            }
        )

    return normalized


def normalize_sgr_response_shape(response_data: dict) -> dict:
    if not isinstance(response_data, dict):
        return response_data

    mode = _as_str(response_data.get("mode"), "initial_plan").lower()

    if mode not in {"initial_plan", "adaptation"}:
        mode = "initial_plan"

    response_data["mode"] = mode

    input_summary = response_data.get("input_summary", {})

    if not isinstance(input_summary, dict):
        input_summary = {}

    response_data["input_summary"] = {
        "brief_goal": _as_str(
            input_summary.get("brief_goal")
            or input_summary.get("goal")
            or input_summary.get("goal_summary"),
            "Не указано",
        ),
        "experience_level": _as_str(input_summary.get("experience_level"), "beginner"),
        "equipment_summary": _as_str(
            input_summary.get("equipment_summary") or input_summary.get("equipment"),
            "Не указано",
        ),
        "restrictions_summary": _as_str(
            input_summary.get("restrictions_summary")
            or input_summary.get("restrictions"),
            "нет",
        ),
        "has_history": _as_bool(
            input_summary.get("has_history")
            if "has_history" in input_summary
            else input_summary.get("history_exists"),
            False,
        ),
        "has_current_session": _as_bool(
            input_summary.get("has_current_session")
            if "has_current_session" in input_summary
            else input_summary.get("current_session"),
            False,
        ),
    }

    progress = response_data.get("progress_assessment", {})

    if not isinstance(progress, dict):
        progress = {}

    response_data["progress_assessment"] = {
        "progress_detected": _as_bool(
            progress.get("progress_detected")
            if "progress_detected" in progress
            else progress.get("progress_signs_exist"),
            False,
        ),
        "supporting_facts": _as_list(
            progress.get("supporting_facts") or progress.get("progress_signs") or []
        ),
        "recommended_progression": (
            _as_str(
                progress.get("recommended_progression")
                or progress.get("progression")
                or progress.get("suggested_progression"),
                "",
            )
            or None
        ),
    }

    overload = response_data.get("overload_assessment", {})

    if not isinstance(overload, dict):
        overload = {}

    recommended_adjustment = (
        overload.get("recommended_adjustment")
        or overload.get("adjustment_type")
        or overload.get("suggested_adjustment")
    )
    recommended_adjustment = _as_str(recommended_adjustment, "")

    if recommended_adjustment not in {"reduce_intensity", "reduce_volume"}:
        recommended_adjustment = None

    response_data["overload_assessment"] = {
        "overload_detected": _as_bool(
            overload.get("overload_detected")
            if "overload_detected" in overload
            else overload.get("overload_signs_exist"),
            False,
        ),
        "overload_signals": _as_list(
            overload.get("overload_signals") or overload.get("signals") or []
        ),
        "recommended_adjustment": recommended_adjustment,
    }

    medical = response_data.get("medical_risk_assessment", {})

    if not isinstance(medical, dict):
        medical = {}

    medical_risk_detected = _as_bool(
        medical.get("medical_risk_detected")
        if "medical_risk_detected" in medical
        else medical.get("risk_detected"),
        False,
    )
    refusal_required = _as_bool(medical.get("refusal_required"), medical_risk_detected)

    response_data["medical_risk_assessment"] = {
        "medical_risk_detected": medical_risk_detected,
        "risk_signals": _as_list(
            medical.get("risk_signals") or medical.get("medical_signals") or []
        ),
        "refusal_required": refusal_required,
        "refuse_reason": (_as_str(medical.get("refuse_reason"), "") or None),
    }

    restriction = response_data.get("restriction_assessment", {})

    if not isinstance(restriction, dict):
        restriction = {}

    response_data["restriction_assessment"] = {
        "restrictions_present": _as_bool(
            restriction.get("restrictions_present")
            if "restrictions_present" in restriction
            else restriction.get("restrictions_exist"),
            False,
        ),
        "limiting_factors": _as_list(
            restriction.get("limiting_factors")
            or restriction.get("restriction_factors")
            or []
        ),
        "restriction_impact_summary": _as_str(
            restriction.get("restriction_impact_summary")
            or restriction.get("impact_summary"),
            "Ограничения не влияют на решение",
        ),
    }

    trace = response_data.get("decision_trace", {})

    if not isinstance(trace, dict):
        trace = {}

    selected_policy = _as_str(trace.get("selected_policy") or trace.get("main_rule"), "")

    policy_map = {
        "medical_refusal": "medical_refusal",
        "restriction_limited": "restriction_limited",
        "overload_reduction": "overload_reduction",
        "progressive_overload": "progressive_overload",
        "maintain_plan": "maintain_plan",
        "initial_plan_generation": "initial_plan_generation",
        "medical safety": "medical_refusal",
        "medical_refusal_policy": "medical_refusal",
        "progression": "progressive_overload",
        "progressive overload": "progressive_overload",
        "overload": "overload_reduction",
        "initial plan": "initial_plan_generation",
    }

    selected_policy = policy_map.get(selected_policy.lower(), selected_policy)

    if selected_policy not in {
        "medical_refusal",
        "restriction_limited",
        "overload_reduction",
        "progressive_overload",
        "maintain_plan",
        "initial_plan_generation",
    }:
        if response_data["medical_risk_assessment"]["medical_risk_detected"]:
            selected_policy = "medical_refusal"
        elif response_data["mode"] == "initial_plan":
            selected_policy = "initial_plan_generation"
        elif response_data["overload_assessment"]["overload_detected"]:
            selected_policy = "overload_reduction"
        elif response_data["restriction_assessment"]["restrictions_present"]:
            selected_policy = "restriction_limited"
        elif response_data["progress_assessment"]["progress_detected"]:
            selected_policy = "progressive_overload"
        else:
            selected_policy = "maintain_plan"

    final_action = _as_str(trace.get("final_action"), "")

    action_map = {
        "proceed": "create_initial_plan"
        if response_data["mode"] == "initial_plan"
        else "maintain",
        "continue": "maintain",
        "adapt": "maintain",
        "increase": "increase_load",
        "increase_weight": "increase_load",
        "reduce_load": "reduce_intensity",
        "reduce_intensity": "reduce_intensity",
        "reduce_volume": "reduce_volume",
        "maintain": "maintain",
        "refuse": "refuse",
        "create_initial_plan": "create_initial_plan",
        "modify_for_restrictions": "modify_for_restrictions",
    }

    final_action = action_map.get(final_action, final_action)

    if final_action not in {
        "refuse",
        "create_initial_plan",
        "increase_load",
        "reduce_intensity",
        "reduce_volume",
        "maintain",
        "modify_for_restrictions",
    }:
        if response_data["medical_risk_assessment"]["medical_risk_detected"]:
            final_action = "refuse"
        elif response_data["mode"] == "initial_plan":
            final_action = "create_initial_plan"
        elif response_data["overload_assessment"]["recommended_adjustment"] == "reduce_volume":
            final_action = "reduce_volume"
        elif response_data["overload_assessment"]["overload_detected"]:
            final_action = "reduce_intensity"
        elif response_data["restriction_assessment"]["restrictions_present"]:
            final_action = "modify_for_restrictions"
        elif response_data["progress_assessment"]["progress_detected"]:
            final_action = "increase_load"
        else:
            final_action = "maintain"

    response_data["decision_trace"] = {
        "selected_policy": selected_policy,
        "final_action": final_action,
        "policy_reasoning": _as_str(
            trace.get("policy_reasoning")
            or trace.get("rule_reasoning")
            or trace.get("main_rule")
            or "Решение выбрано на основе анализа входных данных",
            "Решение выбрано на основе анализа входных данных",
        ),
    }

    final = response_data.get("final_recommendation", {})

    if not isinstance(final, dict):
        final = {}

    exercise_changes = _normalize_sgr_exercise_changes(
        final.get("exercise_changes") or final.get("changes") or []
    )

    final_decision = _as_str(
        final.get("decision")
        or final.get("final_decision")
        or final.get("recommendation")
        or trace.get("final_decision"),
        "",
    )

    response_data["final_recommendation"] = {
        "session_assessment": (
            _as_str(final.get("session_assessment") or final.get("session_evaluation"), "")
            or None
        ),
        "decision": final_decision or "Решение сформировано на основе анализа",
        "exercise_changes": exercise_changes,
        "reasoning": _as_str(
            final.get("reasoning")
            or final.get("explanation")
            or response_data["decision_trace"]["policy_reasoning"],
            response_data["decision_trace"]["policy_reasoning"],
        ),
        "long_term_recommendation": (
            _as_str(final.get("long_term_recommendation"), "") or None
        ),
        "safety_warnings": _as_list(final.get("safety_warnings") or []),
        "refused": _as_bool(
            final.get("refused"),
            response_data["medical_risk_assessment"]["refusal_required"],
        ),
        "refuse_reason": (
            _as_str(
                final.get("refuse_reason")
                or response_data["medical_risk_assessment"]["refuse_reason"],
                "",
            )
            or None
        ),
    }

    if response_data["medical_risk_assessment"]["medical_risk_detected"]:
        response_data["decision_trace"]["selected_policy"] = "medical_refusal"
        response_data["decision_trace"]["final_action"] = "refuse"
        response_data["final_recommendation"]["refused"] = True
        response_data["final_recommendation"]["exercise_changes"] = []

        if not response_data["final_recommendation"]["refuse_reason"]:
            response_data["final_recommendation"]["refuse_reason"] = "Обнаружен медицинский риск"

        if not response_data["final_recommendation"]["decision"]:
            response_data["final_recommendation"]["decision"] = (
                "Отказ от тренировочной рекомендации из-за медицинского риска"
            )

    return response_data


def _extract_tool_calls(message: Any) -> list[Any]:
    return list(getattr(message, "tool_calls", None) or [])


def _assistant_message_to_dict(message: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": "assistant",
        "content": message.content or "",
    }

    tool_calls = _extract_tool_calls(message)

    if tool_calls:
        payload["tool_calls"] = [
            {
                "id": item.id,
                "type": "function",
                "function": {
                    "name": item.function.name,
                    "arguments": item.function.arguments,
                },
            }
            for item in tool_calls
        ]

    return payload


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}

    if isinstance(value, list | tuple | set):
        return [_to_jsonable(item) for item in value]

    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump())

    return str(value)


def _extract_message_content(response: Any, stage: str) -> str:
    choice = response.choices[0]
    message = choice.message
    content = getattr(message, "content", None)

    if isinstance(content, str) and content.strip():
        return content

    finish_reason = getattr(choice, "finish_reason", None)
    tool_calls = getattr(message, "tool_calls", None)

    raise ValueError(
        "Модель вернула пустой message.content на этапе "
        f"{stage}. finish_reason={finish_reason}, tool_calls={tool_calls}"
    )


def _tool_call_record_to_dict(record: ToolCallRecord) -> dict[str, Any]:
    if hasattr(record, "model_dump"):
        return _to_jsonable(record.model_dump())

    return _to_jsonable(
        {
            "tool_name": getattr(record, "tool_name", None),
            "arguments": getattr(record, "arguments", None),
            "result": getattr(record, "result", None),
            "source": getattr(record, "source", None),
        }
    )


def _record_tool_observation(record: ToolCallRecord) -> None:
    langfuse = get_client()
    tool_name = getattr(record, "tool_name", "unknown_tool")

    with langfuse.start_as_current_observation(
        as_type="tool",
        name=f"tool.{tool_name}",
        input=_to_jsonable(getattr(record, "arguments", {})),
        metadata={"source": getattr(record, "source", "unknown")},
    ) as tool_span:
        tool_span.update(output=_to_jsonable(getattr(record, "result", {})))


async def _run_tool_calling_phase(
    client: AsyncOpenAI,
    model_name: str,
    request_data: dict,
) -> tuple[dict[str, dict], AgentExecutionTrace]:
    langfuse = get_client()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": TOOL_SYSTEM_PROMPT},
        {"role": "user", "content": build_tool_user_prompt(request_data)},
    ]

    trace = AgentExecutionTrace(tool_calls=[])
    outputs: dict[str, dict] = {}
    tools = get_openai_tool_definitions()
    max_tokens = _get_max_tokens()

    with langfuse.start_as_current_observation(
        as_type="span",
        name="tool_calling_phase",
        input={
            "request_data": _to_jsonable(request_data),
            "initial_messages": _to_jsonable(messages),
        },
        metadata={
            "model": model_name,
            "max_tokens": max_tokens,
            "tool_count": len(tools),
        },
    ) as phase_span:
        try:
            for iteration in range(6):
                with langfuse.start_as_current_observation(
                    as_type="span",
                    name="tool_calling_model_iteration",
                    input={
                        "iteration": iteration + 1,
                        "messages_count": len(messages),
                    },
                    metadata={
                        "model": model_name,
                        "temperature": 0,
                        "max_tokens": max_tokens,
                    },
                ) as iteration_span:
                    response = await client.chat.completions.create(
                        model=model_name,
                        temperature=0,
                        messages=messages,
                        max_tokens=max_tokens,
                        tools=tools,
                        tool_choice="auto",
                    )

                    message = response.choices[0].message
                    tool_calls = _extract_tool_calls(message)
                    tool_names = [tool_call.function.name for tool_call in tool_calls]

                    iteration_span.update(
                        output={
                            "tool_call_count": len(tool_calls),
                            "tool_names": tool_names,
                            "assistant_content": message.content,
                        }
                    )

                if not tool_calls:
                    break

                messages.append(_assistant_message_to_dict(message))

                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    raw_arguments = tool_call.function.arguments or "{}"
                    parsed_arguments = json.loads(raw_arguments)

                    with langfuse.start_as_current_observation(
                        as_type="tool",
                        name=f"tool.{tool_name}",
                        input=_to_jsonable(parsed_arguments),
                        metadata={"source": "model_function_call"},
                    ) as tool_span:
                        result_model = execute_tool(tool_name, parsed_arguments)
                        result = result_model.model_dump()
                        tool_span.update(output=_to_jsonable(result))

                    outputs[tool_name] = result

                    trace.tool_calls.append(
                        ToolCallRecord(
                            tool_name=tool_name,
                            arguments=parsed_arguments,
                            result=result,
                            source="model_function_call",
                        )
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": dump_tool_result(result_model),
                        }
                    )

        except (
            BadRequestError,
            NotImplementedError,
            KeyError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            logger.warning("Function Calling недоступен или завершился ошибкой: %s", exc)

            local_outputs, local_trace = run_local_tool_pipeline(request_data)

            for record in local_trace.tool_calls:
                _record_tool_observation(record)

            phase_span.update(
                output={
                    "fallback": True,
                    "fallback_reason": str(exc),
                    "tool_outputs": _to_jsonable(local_outputs),
                    "tool_calls": [
                        _tool_call_record_to_dict(record)
                        for record in local_trace.tool_calls
                    ],
                },
                metadata={"fallback_to_local_pipeline": True},
            )

            return local_outputs, local_trace

        required_tools = [
            "build_training_context",
            "retrieve_training_knowledge",
            "assess_restrictions",
            "assess_training_load",
            "assess_medical_risk",
        ]

        local_outputs_cache: dict[str, dict] | None = None
        local_trace_cache: AgentExecutionTrace | None = None

        def get_local_pipeline() -> tuple[dict[str, dict], AgentExecutionTrace]:
            nonlocal local_outputs_cache, local_trace_cache

            if local_outputs_cache is None or local_trace_cache is None:
                local_outputs_cache, local_trace_cache = run_local_tool_pipeline(
                    request_data
                )

            return local_outputs_cache, local_trace_cache

        for tool_name in required_tools:
            if tool_name in outputs:
                continue

            local_outputs, local_trace = get_local_pipeline()
            outputs[tool_name] = local_outputs[tool_name]

            for record in local_trace.tool_calls:
                if record.tool_name == tool_name:
                    forced_record = ToolCallRecord(
                        tool_name=record.tool_name,
                        arguments=record.arguments,
                        result=record.result,
                        source="forced_completion",
                    )
                    trace.tool_calls.append(forced_record)
                    _record_tool_observation(forced_record)
                    break

        if "request_confirmation" not in outputs:
            local_outputs, local_trace = get_local_pipeline()
            outputs["request_confirmation"] = local_outputs["request_confirmation"]

            for record in local_trace.tool_calls:
                if record.tool_name == "request_confirmation":
                    forced_record = ToolCallRecord(
                        tool_name=record.tool_name,
                        arguments=record.arguments,
                        result=record.result,
                        source="forced_completion",
                    )
                    trace.tool_calls.append(forced_record)
                    _record_tool_observation(forced_record)
                    break

        phase_span.update(
            output={
                "tool_outputs": _to_jsonable(outputs),
                "tool_calls": [
                    _tool_call_record_to_dict(record) for record in trace.tool_calls
                ],
            }
        )

    return outputs, trace


async def _request_model_response(
    client: AsyncOpenAI,
    model_name: str,
    temperature: float,
    messages: list[dict],
) -> str:
    response_schema = CoachSGRResponse.model_json_schema()
    max_tokens = _get_max_tokens()

    try:
        response = await client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            messages=messages,
            max_tokens=max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "CoachSGRResponse",
                    "strict": True,
                    "schema": response_schema,
                },
            },
        )

        return _extract_message_content(response, "json_schema")

    except (BadRequestError, ValueError) as exc:
        logger.warning(
            "Провайдер не поддерживает structured output через json_schema "
            "или вернул пустой content. Пробую режим json_object. Ошибка: %s",
            exc,
        )

    try:
        response = await client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            messages=messages,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

        return _extract_message_content(response, "json_object")

    except (BadRequestError, ValueError) as exc:
        logger.warning(
            "Провайдер не поддерживает json_object или вернул пустой content. "
            "Перехожу к обычному запросу с инструкцией вернуть только JSON. Ошибка: %s",
            exc,
        )

    plain_json_messages = messages[:-1] + [
        {
            "role": "user",
            "content": messages[-1]["content"]
            + "\n\nВерни только JSON по схеме SGR, без markdown и без дополнительных пояснений.",
        }
    ]

    response = await client.chat.completions.create(
        model=model_name,
        temperature=temperature,
        messages=plain_json_messages,
        max_tokens=max_tokens,
    )

    return _extract_message_content(response, "plain_json")


async def get_sgr_response_with_trace(
    request_data: dict,
) -> tuple[CoachSGRResponse, AgentExecutionTrace]:
    langfuse = get_client()
    client = get_training_llm_client()

    model_name = os.getenv("LLM_MODEL")

    if not model_name:
        raise RuntimeError("Не задан LLM_MODEL — проверь .env файл")

    temperature = request_data.get("temperature", 0.3)
    max_tokens = _get_max_tokens()

    with langfuse.start_as_current_observation(
        as_type="span",
        name="coach_request",
        input=_to_jsonable(request_data),
        metadata={
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    ) as root_span:
        with propagate_attributes(
            trace_name="coach_request",
            tags=["ai-training-coach", "langfuse", "manual-spans"],
        ):
            try:
                tool_outputs, trace = await _run_tool_calling_phase(
                    client, model_name, request_data
                )

                dialog = [
                    {"role": "system", "content": FINAL_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_final_user_prompt(request_data, tool_outputs),
                    },
                ]

                with langfuse.start_as_current_observation(
                    as_type="span",
                    name="final_response_phase",
                    input={
                        "dialog": _to_jsonable(dialog),
                        "tool_outputs": _to_jsonable(tool_outputs),
                    },
                    metadata={
                        "model": model_name,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                ) as final_phase_span:
                    raw_answer = await _request_model_response(
                        client=client,
                        model_name=model_name,
                        temperature=temperature,
                        messages=dialog,
                    )
                    final_phase_span.update(output={"raw_answer": raw_answer})

                with langfuse.start_as_current_observation(
                    as_type="span",
                    name="sgr_response_parsing",
                    input={"raw_answer": raw_answer},
                ) as parsing_span:
                    sgr_data = normalize_sgr_response_shape(
                        json.loads(extract_json_from_model_answer(raw_answer))
                    )
                    sgr_response = CoachSGRResponse(**sgr_data)
                    parsing_span.update(
                        output={
                            "sgr_data": _to_jsonable(sgr_data),
                            "sgr_response": _to_jsonable(sgr_response),
                        }
                    )

                root_span.update(
                    output={
                        "sgr_response": _to_jsonable(sgr_response),
                        "tool_outputs": _to_jsonable(tool_outputs),
                        "tool_calls": [
                            _tool_call_record_to_dict(record)
                            for record in trace.tool_calls
                        ],
                    }
                )

                _flush_langfuse()

                return sgr_response, trace

            except Exception as exc:
                root_span.update(
                    output={"error": str(exc)},
                    metadata={"error_type": type(exc).__name__},
                )
                _flush_langfuse()
                raise


async def get_sgr_response(request_data: dict) -> CoachSGRResponse:
    sgr_response, _ = await get_sgr_response_with_trace(request_data)

    return sgr_response


async def get_coach_response_with_trace(
    request_data: dict,
) -> tuple[CoachResponse, AgentExecutionTrace]:
    sgr_response, trace = await get_sgr_response_with_trace(request_data)
    coach_response = sgr_to_coach_response(sgr_response)

    langfuse = get_client()

    with langfuse.start_as_current_observation(
        as_type="span",
        name="coach_response_mapping",
        input={"sgr_response": _to_jsonable(sgr_response)},
    ) as mapping_span:
        mapping_span.update(output={"coach_response": _to_jsonable(coach_response)})

    _flush_langfuse()

    return coach_response, trace


async def get_coach_response(request_data: dict) -> CoachResponse:
    response, _ = await get_coach_response_with_trace(request_data)

    return response
