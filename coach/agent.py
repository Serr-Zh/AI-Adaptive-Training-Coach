import json
import logging
from typing import Any

from langfuse import get_client, propagate_attributes
from langfuse.openai import AsyncOpenAI
from openai import BadRequestError

from coach.client import flush_langfuse, get_max_tokens, get_model_name, get_training_llm_client
from coach.exceptions import InvalidModelResponseError
from coach.models import (
    AgentExecutionTrace,
    CoachResponse,
    CoachSGRResponse,
    ToolCallRecord,
    sgr_to_coach_response,
)
from coach.parse import extract_json_from_model_answer, normalize_sgr_response_shape
from coach.prompts import (
    FINAL_SYSTEM_PROMPT,
    TOOL_SYSTEM_PROMPT,
    build_final_user_prompt,
    build_tool_user_prompt,
)
from coach.retry import retry_with_backoff
from coach.tools import (
    dump_tool_result,
    execute_tool,
    get_openai_tool_definitions,
    run_local_tool_pipeline,
)

logger = logging.getLogger(__name__)

_MAX_TOOL_CALLING_ITERATIONS = 6
_REQUIRED_TOOL_NAMES = (
    "build_training_context",
    "retrieve_training_knowledge",
    "assess_restrictions",
    "assess_training_load",
    "assess_medical_risk",
)
_REQUEST_CONFIRMATION_TOOL = "request_confirmation"
_OPENAI_TOOL_DEFINITIONS = get_openai_tool_definitions()
_SGR_RESPONSE_SCHEMA = CoachSGRResponse.model_json_schema()

_RESPONSE_FORMAT_STAGES: list[tuple[str, dict]] = [
    (
        "json_schema",
        {
            "type": "json_schema",
            "json_schema": {
                "name": "CoachSGRResponse",
                "strict": True,
                "schema": _SGR_RESPONSE_SCHEMA,
            },
        },
    ),
    ("json_object", {"type": "json_object"}),
]


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump())
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [_to_jsonable(item) for item in value]
    return str(value)


def _extract_message_content(response: Any, stage: str) -> str:
    choice = response.choices[0]
    message = choice.message
    content = getattr(message, "content", None)

    if isinstance(content, str) and content.strip():
        return content

    finish_reason = getattr(choice, "finish_reason", None)
    tool_calls = getattr(message, "tool_calls", None)

    raise InvalidModelResponseError(
        "Модель вернула пустой message.content на этапе "
        f"{stage}. finish_reason={finish_reason}, tool_calls={tool_calls}"
    )


def _tool_call_record_to_dict(record: ToolCallRecord) -> dict[str, Any]:
    return _to_jsonable(record.model_dump())


def _record_tool_observation(record: ToolCallRecord) -> None:
    langfuse = get_client()

    with langfuse.start_as_current_observation(
        as_type="tool",
        name=f"tool.{record.tool_name}",
        input=_to_jsonable(record.arguments),
        metadata={"source": record.source},
    ) as tool_span:
        tool_span.update(output=_to_jsonable(record.result))


def _build_trace_index(trace: AgentExecutionTrace) -> dict[str, ToolCallRecord]:
    return {record.tool_name: record for record in trace.tool_calls}


def _append_forced_tool_completion(
    trace: AgentExecutionTrace,
    local_trace_index: dict[str, ToolCallRecord],
    tool_name: str,
) -> None:
    record = local_trace_index.get(tool_name)
    if record is None:
        return

    forced_record = ToolCallRecord(
        tool_name=record.tool_name,
        arguments=record.arguments,
        result=record.result,
        source="forced_completion",
    )
    trace.tool_calls.append(forced_record)
    _record_tool_observation(forced_record)


def _fill_missing_tool_outputs(
    outputs: dict[str, dict],
    trace: AgentExecutionTrace,
    request_data: dict,
) -> None:
    missing = [t for t in _REQUIRED_TOOL_NAMES if t not in outputs]
    if _REQUEST_CONFIRMATION_TOOL not in outputs:
        missing.append(_REQUEST_CONFIRMATION_TOOL)

    if not missing:
        return

    local_outputs, local_trace = run_local_tool_pipeline(request_data)
    local_trace_index = _build_trace_index(local_trace)

    for tool_name in missing:
        outputs[tool_name] = local_outputs[tool_name]
        _append_forced_tool_completion(trace, local_trace_index, tool_name)


def _fallback_to_local_pipeline(
    request_data: dict,
    phase_span: Any,
) -> tuple[dict[str, dict], AgentExecutionTrace]:
    local_outputs, local_trace = run_local_tool_pipeline(request_data)

    for record in local_trace.tool_calls:
        _record_tool_observation(record)

    phase_span.update(
        output={
            "fallback": True,
            "fallback_reason": "tool_calling_failed",
            "tool_outputs": _to_jsonable(local_outputs),
            "tool_calls": [
                _tool_call_record_to_dict(record)
                for record in local_trace.tool_calls
            ],
        },
        metadata={"fallback_to_local_pipeline": True},
    )

    return local_outputs, local_trace


async def _execute_single_tool_call(
    tool_call: Any,
    messages: list[dict[str, Any]],
    outputs: dict[str, dict],
    trace: AgentExecutionTrace,
) -> None:
    tool_name = tool_call.function.name
    raw_arguments = tool_call.function.arguments or "{}"
    parsed_arguments = json.loads(raw_arguments)

    langfuse = get_client()

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
    tools = _OPENAI_TOOL_DEFINITIONS
    max_tokens = get_max_tokens()

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
            for iteration in range(_MAX_TOOL_CALLING_ITERATIONS):
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
                    response = await retry_with_backoff(
                        client.chat.completions.create,
                        model=model_name,
                        temperature=0,
                        messages=messages,
                        max_tokens=max_tokens,
                        tools=tools,
                        tool_choice="auto",
                    )

                    message = response.choices[0].message
                    tool_calls = list(getattr(message, "tool_calls", None) or [])
                    tool_names = [tc.function.name for tc in tool_calls]

                    iteration_span.update(
                        output={
                            "tool_call_count": len(tool_calls),
                            "tool_names": tool_names,
                            "assistant_content": message.content,
                        }
                    )

                if not tool_calls:
                    break

                dumped = message.model_dump(exclude_none=True, exclude_unset=True)
                dumped.setdefault("content", "")
                messages.append(dumped)

                for tool_call in tool_calls:
                    await _execute_single_tool_call(
                        tool_call, messages, outputs, trace
                    )

        except (
            BadRequestError,
            NotImplementedError,
            KeyError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            logger.warning("Function Calling недоступен или завершился ошибкой: %s", exc)
            return _fallback_to_local_pipeline(request_data, phase_span)

        _fill_missing_tool_outputs(outputs, trace, request_data)

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
    max_tokens = get_max_tokens()

    for stage, response_format in _RESPONSE_FORMAT_STAGES:
        try:
            response = await retry_with_backoff(
                client.chat.completions.create,
                model=model_name,
                temperature=temperature,
                messages=messages,
                max_tokens=max_tokens,
                response_format=response_format,
            )
            return _extract_message_content(response, stage)
        except (BadRequestError, InvalidModelResponseError) as exc:
            logger.warning(
                "Провайдер не поддерживает %s или вернул пустой content. "
                "Пробую следующий формат. Ошибка: %s",
                stage,
                exc,
            )

    plain_json_messages = messages[:-1] + [
        {
            "role": "user",
            "content": messages[-1]["content"]
            + "\n\nВерни только JSON по схеме SGR, без markdown и без дополнительных пояснений.",
        }
    ]

    response = await retry_with_backoff(
        client.chat.completions.create,
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
    model_name = get_model_name()
    temperature = request_data.get("temperature", 0.3)
    max_tokens = get_max_tokens()

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

                flush_langfuse()

                return sgr_response, trace

            except Exception as exc:
                root_span.update(
                    output={"error": str(exc)},
                    metadata={"error_type": type(exc).__name__},
                )
                flush_langfuse()
                raise


async def get_sgr_response(request_data: dict) -> CoachSGRResponse:
    sgr_response, _ = await get_sgr_response_with_trace(request_data)
    return sgr_response


async def get_coach_response_with_trace(
    request_data: dict,
) -> tuple[CoachResponse, AgentExecutionTrace]:
    sgr_response, trace = await get_sgr_response_with_trace(request_data)
    return sgr_to_coach_response(sgr_response), trace


async def get_coach_response(request_data: dict) -> CoachResponse:
    response, _ = await get_coach_response_with_trace(request_data)
    return response
