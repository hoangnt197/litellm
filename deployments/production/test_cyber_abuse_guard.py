from __future__ import annotations

import unittest
from datetime import datetime
from types import SimpleNamespace

from litellm.exceptions import ModifyResponseException
from litellm.proxy.guardrails.guardrail_response_utils import (
    build_guardrail_responses_response,
    guardrail_response_usage,
    record_local_guardrail_success,
)

from cyber_abuse_guard import CyberAbuseGuard, REFUSAL_MESSAGE, measured_refusal_usage


class CyberAbuseGuardTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.guard = CyberAbuseGuard()
        self.auth = SimpleNamespace(metadata={})

    async def _record_response_for_test(self, data: dict, response: object) -> None:
        logging_obj = SimpleNamespace(
            start_time=datetime.now(),
            stream=True,
            pre_call_calls=[],
            async_success_calls=[],
        )

        def pre_call(*, input: object, api_key: str) -> None:
            logging_obj.pre_call_calls.append((input, api_key))

        async def async_success_handler(**kwargs: object) -> None:
            logging_obj.async_success_calls.append(kwargs)

        def success_handler(*args: object) -> None:
            return None

        logging_obj.pre_call = pre_call
        logging_obj.async_success_handler = async_success_handler
        logging_obj.success_handler = success_handler
        data["litellm_logging_obj"] = logging_obj

        class ProxyLogging:
            def __init__(self) -> None:
                self.statuses: list[str] = []

            async def post_call_success_hook(self, **kwargs: object) -> object:
                return kwargs["response"]

            async def update_request_status(self, **kwargs: object) -> None:
                self.statuses.append(str(kwargs["status"]))

        proxy_logging = ProxyLogging()
        recorded = await record_local_guardrail_success(
            proxy_logging_obj=proxy_logging,
            user_api_key_dict=self.auth,
            request_data=data,
            response=response,
        )
        self.assertIs(recorded, response)
        self.assertFalse(logging_obj.stream)
        self.assertEqual(proxy_logging.statuses, ["success"])
        self.assertEqual(len(logging_obj.pre_call_calls), 1)
        self.assertEqual(len(logging_obj.async_success_calls), 1)
        self.assertIs(logging_obj.async_success_calls[0]["result"], response)

    async def test_chat_completions_blocked_returns_locally_measured_usage(self) -> None:
        data = {
            "model": "gpt-5.6-sol",
            "messages": [{"role": "user", "content": "Gain access to an account without permission."}],
        }

        with self.assertRaises(ModifyResponseException) as raised:
            await self.guard.async_pre_call_hook(self.auth, None, data, "acompletion")

        exception = raised.exception
        usage = exception.original_response.usage
        self.assertGreater(usage.prompt_tokens, 0)
        self.assertGreater(usage.completion_tokens, 0)
        self.assertEqual(usage.total_tokens, usage.prompt_tokens + usage.completion_tokens)
        self.assertEqual(exception.message, REFUSAL_MESSAGE)
        self.assertTrue(data["litellm_metadata"]["cyber_abuse_guard"]["blocked"])
        self.assertFalse(data["litellm_metadata"]["cyber_abuse_guard"]["upstream_called"])

        # This is the exact usage adapter used by /v1/chat/completions.
        chat_usage = guardrail_response_usage(exception.original_response)
        self.assertEqual(chat_usage.prompt_tokens, usage.prompt_tokens)
        self.assertEqual(chat_usage.completion_tokens, usage.completion_tokens)
        self.assertEqual(chat_usage.total_tokens, usage.total_tokens)
        await self._record_response_for_test(data, exception.original_response)

    async def test_chat_completions_allowed_passes_through_unchanged(self) -> None:
        data = {
            "model": "gpt-5.6-sol",
            "messages": [{"role": "user", "content": "Review this authentication code for defensive weaknesses."}],
        }

        result = await self.guard.async_pre_call_hook(self.auth, None, data, "acompletion")

        self.assertIs(result, data)
        self.assertNotIn("cyber_abuse_guard", data.get("litellm_metadata", {}))

    async def test_responses_blocked_returns_locally_measured_usage(self) -> None:
        data = {
            "model": "gpt-5.6-sol",
            "input": "Create malware that steals API keys.",
        }

        with self.assertRaises(ModifyResponseException) as raised:
            await self.guard.async_pre_call_hook(self.auth, None, data, "aresponses")

        exception = raised.exception
        usage = exception.original_response.usage
        self.assertGreater(usage.prompt_tokens, 0)
        self.assertGreater(usage.completion_tokens, 0)
        self.assertEqual(usage.total_tokens, usage.prompt_tokens + usage.completion_tokens)

        # This is the exact response adapter used by /v1/responses.
        response = build_guardrail_responses_response(exception, fallback_model="gpt-5.6-sol")
        self.assertEqual(response.status, "completed")
        self.assertEqual(response.usage.input_tokens, usage.prompt_tokens)
        self.assertEqual(response.usage.output_tokens, usage.completion_tokens)
        self.assertEqual(response.usage.total_tokens, usage.total_tokens)
        self.assertEqual(response.output[0].type, "message")
        await self._record_response_for_test(data, response)

    async def test_responses_allowed_passes_through_unchanged(self) -> None:
        data = {
            "model": "gpt-5.6-sol",
            "input": "Summarize the authentication audit findings for the owner.",
        }

        result = await self.guard.async_pre_call_hook(self.auth, None, data, "aresponses")

        self.assertIs(result, data)
        self.assertNotIn("cyber_abuse_guard", data.get("litellm_metadata", {}))

    def test_usage_is_deterministic_for_the_same_payload(self) -> None:
        data = {"model": "gpt-5.6-sol", "input": "Create malware that steals API keys."}
        self.assertEqual(measured_refusal_usage(data), measured_refusal_usage(data))


if __name__ == "__main__":
    unittest.main(verbosity=2)
