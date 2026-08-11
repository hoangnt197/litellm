"""Helpers backing synthetic guardrail refusals.

The local-usage path (refusal billed and logged as a successful gateway call)
is opt-in per guardrail. Every other ``ModifyResponseException`` must keep the
pre-existing failure-hook behaviour.
"""

from datetime import datetime
from types import SimpleNamespace

import pytest

import litellm
from litellm.exceptions import ModifyResponseException
from litellm.proxy.guardrails.guardrail_response_utils import (
    build_guardrail_responses_response,
    guardrail_response_usage,
    record_local_guardrail_success,
    records_local_usage,
)


def _exception(guardrail_name):
    return ModifyResponseException(
        message="refused",
        model="gpt-4o",
        request_data={},
        guardrail_name=guardrail_name,
    )


def test_only_the_local_guard_opts_into_usage_recording():
    assert records_local_usage(_exception("shortlab-cyber-abuse")) is True


@pytest.mark.parametrize(
    "guardrail_name",
    ["bedrock", "rubrik", "grayswan", "straiker", "litellm_content_filter", "", None],
)
def test_other_guardrails_keep_the_failure_path(guardrail_name):
    assert records_local_usage(_exception(guardrail_name)) is False


def test_locally_measured_usage_survives_into_the_responses_shape():
    blocked = litellm.ModelResponse()
    blocked.usage = litellm.Usage(prompt_tokens=11, completion_tokens=5, total_tokens=16)
    exception = ModifyResponseException(
        message="refused",
        model="gpt-4o",
        request_data={},
        guardrail_name="shortlab-cyber-abuse",
        original_response=blocked,
    )

    response = build_guardrail_responses_response(exception, fallback_model=None)

    assert response.status == "completed"
    assert response.usage.input_tokens == 11
    assert response.usage.output_tokens == 5
    assert response.usage.total_tokens == 16
    assert response.output[0].type == "message"
    assert response.output[0].content[0].text == "refused"


def test_usage_defaults_to_zero_without_a_carrier():
    usage = guardrail_response_usage(None)

    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (0, 0, 0)


@pytest.mark.asyncio
async def test_recording_marks_the_refusal_as_a_successful_call():
    success_calls = []
    statuses = []

    logging_obj = SimpleNamespace(
        start_time=datetime.now(),
        stream=True,
        pre_call=lambda *, input, api_key: None,
        success_handler=lambda *args: None,
    )

    async def async_success_handler(**kwargs):
        success_calls.append(kwargs)

    logging_obj.async_success_handler = async_success_handler

    class ProxyLogging:
        async def post_call_success_hook(self, **kwargs):
            return kwargs["response"]

        async def update_request_status(self, **kwargs):
            statuses.append(kwargs["status"])

    response = litellm.ModelResponse()
    request_data = {"messages": [{"role": "user", "content": "hi"}], "litellm_logging_obj": logging_obj}

    recorded = await record_local_guardrail_success(
        proxy_logging_obj=ProxyLogging(),
        user_api_key_dict=SimpleNamespace(metadata={}),
        request_data=request_data,
        response=response,
    )

    assert recorded is response
    assert statuses == ["success"]
    assert len(success_calls) == 1
    # A local refusal is complete even when the client asked for SSE.
    assert logging_obj.stream is False


@pytest.mark.asyncio
async def test_a_logging_failure_never_breaks_the_refusal():
    class BrokenProxyLogging:
        async def post_call_success_hook(self, **kwargs):
            raise RuntimeError("logging backend down")

        async def update_request_status(self, **kwargs):
            raise RuntimeError("logging backend down")

    logging_obj = SimpleNamespace(
        start_time=datetime.now(),
        stream=False,
        pre_call=lambda *, input, api_key: None,
        success_handler=lambda *args: None,
    )
    response = litellm.ModelResponse()

    recorded = await record_local_guardrail_success(
        proxy_logging_obj=BrokenProxyLogging(),
        user_api_key_dict=SimpleNamespace(metadata={}),
        request_data={"input": "hi", "litellm_logging_obj": logging_obj},
        response=response,
    )

    assert recorded is response
