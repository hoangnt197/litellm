from __future__ import annotations

import unittest
from types import SimpleNamespace

from litellm.types.llms.openai import (
    ResponseCompletedEvent,
    ResponsesAPIResponse,
)

from reasoning_policy import (
    ReasoningPolicyHandler,
    apply_reasoning_policy,
    restore_requested_effort,
)


POLICY = {
    "reasoning_policy": {
        "enabled": True,
        "rules": [
            {
                "models": ["gpt-5.6-sol", "gpt-5.6-luna"],
                "fallbacks": {"ultra": "high", "max": "high", "xhigh": "high"},
            }
        ],
    }
}


class ReasoningPolicyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.auth = SimpleNamespace(metadata=POLICY)

    def test_rewrites_supported_fields_without_changing_model(self) -> None:
        data = {
            "model": "gpt-5.6-sol",
            "reasoning_effort": "max",
            "reasoning": {"effort": "xhigh", "summary": "auto"},
            "output_config": {"effort": "ultra"},
        }
        apply_reasoning_policy(self.auth, data, "aresponses")

        self.assertEqual(data["model"], "gpt-5.6-sol")
        self.assertEqual(data["reasoning_effort"], "high")
        self.assertEqual(data["reasoning"]["effort"], "high")
        self.assertEqual(data["reasoning"]["summary"], "auto")
        self.assertEqual(data["output_config"]["effort"], "high")

    def test_model_scope_and_first_match(self) -> None:
        out_of_scope = {"model": "gpt-5.5", "reasoning": {"effort": "xhigh"}}
        apply_reasoning_policy(self.auth, out_of_scope, "aresponses")
        self.assertEqual(out_of_scope["reasoning"]["effort"], "xhigh")
        self.assertNotIn("litellm_metadata", out_of_scope)

        auth = SimpleNamespace(
            metadata={
                "reasoning_policy": {
                    "enabled": True,
                    "rules": [
                        {"models": ["gpt-*"], "fallbacks": {"xhigh": "medium"}},
                        {"models": ["*"], "fallbacks": {"xhigh": "high"}},
                    ],
                }
            }
        )
        first_match = {"model": "gpt-5.6-sol", "reasoning": {"effort": "xhigh"}}
        apply_reasoning_policy(auth, first_match, "aresponses")
        self.assertEqual(first_match["reasoning"]["effort"], "medium")

    def test_disabled_or_unmapped_policy_is_noop(self) -> None:
        disabled = SimpleNamespace(metadata={"reasoning_policy": {"enabled": False, "rules": []}})
        data = {"model": "gpt-5.6-sol", "reasoning": {"effort": "xhigh"}}
        apply_reasoning_policy(disabled, data, "aresponses")
        self.assertEqual(data["reasoning"]["effort"], "xhigh")

        unmapped = {"model": "gpt-5.6-sol", "reasoning": {"effort": "medium"}}
        apply_reasoning_policy(self.auth, unmapped, "aresponses")
        self.assertEqual(unmapped["reasoning"]["effort"], "medium")

    def test_non_streaming_response_restores_requested_effort_only(self) -> None:
        data = {
            "model": "gpt-5.6-sol",
            "reasoning": {"effort": "xhigh", "summary": "auto"},
        }
        apply_reasoning_policy(self.auth, data, "aresponses")
        response = {
            "object": "response",
            "reasoning": {"effort": "high", "summary": "auto"},
            "usage": {"output_tokens": 123, "output_tokens_details": {"reasoning_tokens": 45}},
            "output": [{"type": "reasoning", "summary": []}],
        }
        restored = restore_requested_effort(data, response)

        self.assertEqual(restored["reasoning"]["effort"], "xhigh")
        self.assertEqual(restored["reasoning"]["summary"], "auto")
        self.assertEqual(restored["usage"]["output_tokens_details"]["reasoning_tokens"], 45)
        self.assertEqual(restored["output"], [{"type": "reasoning", "summary": []}])

    async def test_stream_lifecycle_event_restores_pydantic_response(self) -> None:
        data = {"model": "gpt-5.6-sol", "reasoning": {"effort": "xhigh"}}
        apply_reasoning_policy(self.auth, data, "aresponses")
        event = ResponseCompletedEvent(
            type="response.completed",
            response=ResponsesAPIResponse(
                id="resp_test",
                created_at=1,
                object="response",
                output=[],
                reasoning={"effort": "high"},
            ),
        )

        async def source():
            yield event

        handler = ReasoningPolicyHandler()
        chunks = [
            chunk
            async for chunk in handler.async_post_call_streaming_iterator_hook(
                self.auth,
                source(),
                data,
            )
        ]
        self.assertEqual(chunks[0].response.reasoning["effort"], "xhigh")


if __name__ == "__main__":
    unittest.main(verbosity=2)
