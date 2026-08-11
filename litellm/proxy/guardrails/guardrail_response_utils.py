"""Small, dependency-light helpers for synthetic guardrail responses."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Optional, cast
from uuid import uuid4

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import ModifyResponseException
from litellm.litellm_core_utils.thread_pool_executor import executor as thread_pool_executor
from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse
from litellm.types.utils import Usage


LOCAL_USAGE_GUARDRAIL_NAMES = frozenset({"shortlab-cyber-abuse"})


def records_local_usage(exception: ModifyResponseException) -> bool:
    """True for guardrails that block locally and bill the refusal as a gateway success.

    Every other ``ModifyResponseException`` keeps the pre-existing path: the
    failure hook runs and the refusal is not recorded as a successful call.
    """
    return exception.guardrail_name in LOCAL_USAGE_GUARDRAIL_NAMES


def guardrail_response_usage(original_response: Optional[Any]) -> Usage:
    """Read usage preserved on a synthetic or provider response safely.

    A pre-call guardrail can carry locally measured usage; a post-call
    guardrail can carry the provider's original usage.  Both should be kept in
    the response instead of silently becoming zero.
    """
    usage = getattr(original_response, "usage", None)
    if isinstance(usage, Usage):
        return usage

    return Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)


def build_guardrail_responses_response(
    exception: ModifyResponseException,
    fallback_model: Optional[str],
) -> ResponsesAPIResponse:
    """Build an OpenAI-compatible completed response for a guardrail block."""
    usage = guardrail_response_usage(exception.original_response)
    return ResponsesAPIResponse(
        id=f"resp_{uuid4()}",
        object="response",
        created_at=int(time.time()),
        model=exception.model or fallback_model,
        output=cast(
            Any,
            [
                {
                    "id": f"msg_{uuid4()}",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": exception.message,
                            "annotations": [],
                        }
                    ],
                }
            ],
        ),
        status="completed",
        usage=ResponseAPIUsage(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        ),
    )


def _logging_input(request_data: dict[str, Any]) -> Any:
    """Return the inbound content in the shape expected by LiteLLM logging."""
    if "messages" in request_data:
        return request_data["messages"]
    if "input" in request_data:
        return request_data["input"]
    return request_data.get("prompt", "")


async def record_local_guardrail_success(
    *,
    proxy_logging_obj: Any,
    user_api_key_dict: Any,
    request_data: dict[str, Any],
    response: Any,
) -> Any:
    """Record a local refusal as a successful, billable gateway request.

    The provider is not contacted. The response usage is locally measured, then
    passed through the normal success handlers so client usage, spend logs, and
    reports use the same numbers. A logging error must never turn the refusal
    into a 5xx response.
    """
    logging_obj = request_data.get("litellm_logging_obj")
    if logging_obj is None:
        return response

    try:
        # The local response is complete even if the client requested SSE. Log
        # it once now; there is no provider stream that can later flush usage.
        logging_obj.stream = False
        logging_obj.pre_call(input=_logging_input(request_data), api_key="")

        modified_response = await proxy_logging_obj.post_call_success_hook(
            data=request_data,
            user_api_key_dict=user_api_key_dict,
            response=response,
        )
        if modified_response is not None:
            response = modified_response

        await proxy_logging_obj.update_request_status(
            litellm_call_id=request_data.get("litellm_call_id", ""),
            status="success",
        )

        end_time = datetime.now()
        await logging_obj.async_success_handler(
            result=response,
            start_time=logging_obj.start_time,
            end_time=end_time,
            cache_hit=False,
        )
        thread_pool_executor.submit(
            logging_obj.success_handler,
            response,
            logging_obj.start_time,
            end_time,
            False,
        )
    except Exception:
        verbose_proxy_logger.exception("Unable to record local guardrail usage")

    return response
