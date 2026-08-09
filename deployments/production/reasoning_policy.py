"""Per-key, model-scoped reasoning fallback for the LiteLLM proxy.

Virtual-key metadata schema::

    {
      "reasoning_policy": {
        "enabled": true,
        "rules": [
          {
            "models": ["gpt-5.6-sol", "gpt-5.6-luna"],
            "fallbacks": {"ultra": "high", "max": "high", "xhigh": "high"}
          }
        ]
      }
    }

The request sent upstream uses the fallback effort. Responses API clients keep
seeing the effort they requested in full responses and lifecycle SSE events.
"""

from __future__ import annotations

import fnmatch
import json
from typing import Any, AsyncGenerator, Optional

from litellm.integrations.custom_logger import CustomLogger


_CONTEXT_KEY = "_reasoning_policy_context"
_RESPONSE_CALL_TYPES = frozenset({"responses", "aresponses"})


def _non_empty_string(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _key_metadata(user_api_key_dict: Any) -> dict[str, Any]:
    """Return virtual-key metadata without making a database call."""
    metadata = getattr(user_api_key_dict, "metadata", None)
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (TypeError, ValueError):
            return {}
    return metadata if isinstance(metadata, dict) else {}


def _model_matches(model: str, patterns: Any) -> bool:
    if not isinstance(patterns, list):
        return False
    normalized_model = model.casefold()
    for pattern in patterns:
        candidate = _non_empty_string(pattern)
        if candidate and fnmatch.fnmatchcase(normalized_model, candidate.casefold()):
            return True
    return False


def _matching_fallbacks(user_api_key_dict: Any, model: str) -> Optional[dict[str, str]]:
    policy = _key_metadata(user_api_key_dict).get("reasoning_policy")
    if not isinstance(policy, dict) or policy.get("enabled", True) is not True:
        return None

    rules = policy.get("rules")
    if not isinstance(rules, list):
        return None

    # Rules are intentionally first-match so a key can put narrow model rules
    # before a final wildcard rule.
    for rule in rules:
        if not isinstance(rule, dict) or not _model_matches(model, rule.get("models")):
            continue
        raw_fallbacks = rule.get("fallbacks")
        if not isinstance(raw_fallbacks, dict):
            return None
        fallbacks: dict[str, str] = {}
        for requested, effective in raw_fallbacks.items():
            requested_value = _non_empty_string(requested)
            effective_value = _non_empty_string(effective)
            if requested_value and effective_value:
                fallbacks[requested_value.casefold()] = effective_value
        return fallbacks
    return None


def _request_effort_fields(data: dict[str, Any]) -> list[tuple[str, dict[str, Any], str, str]]:
    """Return (field name, owner, key, original value) tuples."""
    fields: list[tuple[str, dict[str, Any], str, str]] = []

    root_effort = _non_empty_string(data.get("reasoning_effort"))
    if root_effort:
        fields.append(("reasoning_effort", data, "reasoning_effort", root_effort))

    reasoning = data.get("reasoning")
    if isinstance(reasoning, dict):
        effort = _non_empty_string(reasoning.get("effort"))
        if effort:
            fields.append(("reasoning.effort", reasoning, "effort", effort))

    output_config = data.get("output_config")
    if isinstance(output_config, dict):
        effort = _non_empty_string(output_config.get("effort"))
        if effort:
            fields.append(("output_config.effort", output_config, "effort", effort))

    return fields


def _is_responses_call(data: dict[str, Any], call_type: str) -> bool:
    if str(call_type or "").casefold() in _RESPONSE_CALL_TYPES:
        return True
    proxy_request = data.get("proxy_server_request")
    if isinstance(proxy_request, dict):
        route = str(proxy_request.get("url") or proxy_request.get("path") or "")
        return "/responses" in route.split("?", 1)[0]
    return False


def apply_reasoning_policy(
    user_api_key_dict: Any,
    data: dict[str, Any],
    call_type: str,
) -> dict[str, Any]:
    """Apply a policy to a request in place and attach private restore context."""
    model = _non_empty_string(data.get("model"))
    if not model:
        return data

    fallbacks = _matching_fallbacks(user_api_key_dict, model)
    if not fallbacks:
        return data

    fields = _request_effort_fields(data)
    if not fields:
        return data

    # Responses API returns reasoning.effort. Prefer its original value when a
    # caller provided more than one compatible request field.
    response_requested = next(
        (original for name, _owner, _key, original in fields if name == "reasoning.effort"),
        fields[0][3],
    )

    applied_fields: list[str] = []
    effective_for_response: Optional[str] = None
    for field_name, owner, key, original in fields:
        effective = fallbacks.get(original.casefold())
        if not effective or effective == original:
            continue
        owner[key] = effective
        applied_fields.append(field_name)
        if original == response_requested and effective_for_response is None:
            effective_for_response = effective

    if not applied_fields:
        return data

    litellm_metadata = data.get("litellm_metadata")
    if not isinstance(litellm_metadata, dict):
        litellm_metadata = {}
        data["litellm_metadata"] = litellm_metadata
    litellm_metadata[_CONTEXT_KEY] = {
        "requested": response_requested,
        "effective": effective_for_response or fields[0][3],
        "model": model,
        "applied_fields": applied_fields,
        "restore_responses_api": _is_responses_call(data, call_type),
    }
    return data


def _policy_context(data: Any) -> Optional[dict[str, Any]]:
    if not isinstance(data, dict):
        return None
    litellm_metadata = data.get("litellm_metadata")
    if not isinstance(litellm_metadata, dict):
        return None
    context = litellm_metadata.get(_CONTEXT_KEY)
    if not isinstance(context, dict) or context.get("restore_responses_api") is not True:
        return None
    return context


def _set_effort_on_response(response: Any, requested: str) -> Any:
    """Mutate only top-level Responses API reasoning.effort fields."""
    if response is None:
        return response

    if isinstance(response, dict):
        nested_response = response.get("response")
        if nested_response is not None:
            _set_effort_on_response(nested_response, requested)

        if response.get("object") == "response" or "reasoning" in response:
            reasoning = response.get("reasoning")
            if isinstance(reasoning, dict):
                reasoning["effort"] = requested
            elif reasoning is None:
                response["reasoning"] = {"effort": requested}
            elif hasattr(reasoning, "effort"):
                setattr(reasoning, "effort", requested)
        return response

    nested_response = getattr(response, "response", None)
    if nested_response is not None:
        _set_effort_on_response(nested_response, requested)

    if getattr(response, "object", None) == "response" or hasattr(response, "reasoning"):
        reasoning = getattr(response, "reasoning", None)
        if isinstance(reasoning, dict):
            reasoning["effort"] = requested
        elif reasoning is None:
            setattr(response, "reasoning", {"effort": requested})
        elif hasattr(reasoning, "effort"):
            setattr(reasoning, "effort", requested)
    return response


def restore_requested_effort(data: dict[str, Any], response: Any) -> Any:
    context = _policy_context(data)
    if context is None:
        return response
    requested = _non_empty_string(context.get("requested"))
    if not requested:
        return response
    return _set_effort_on_response(response, requested)


class ReasoningPolicyHandler(CustomLogger):
    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict[str, Any],
        call_type: str,
    ) -> dict[str, Any]:
        try:
            if isinstance(data, dict):
                return apply_reasoning_policy(user_api_key_dict, data, call_type)
        except Exception as exc:
            print(f"[REASONING-POLICY] request rewrite failed open: {exc}")
        return data

    async def async_post_call_success_hook(
        self,
        data: dict[str, Any],
        user_api_key_dict: Any,
        response: Any,
    ) -> Any:
        try:
            return restore_requested_effort(data, response)
        except Exception as exc:
            print(f"[REASONING-POLICY] response restore failed open: {exc}")
            return response

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: Any,
        response: Any,
        request_data: dict[str, Any],
    ) -> AsyncGenerator[Any, None]:
        # Resolve the request context once per stream. Chat streams and
        # Responses requests without an applied fallback remain a minimal
        # pass-through and do not repeat policy parsing for every token.
        context = _policy_context(request_data)
        requested = _non_empty_string(context.get("requested")) if context else None
        if not requested:
            async for chunk in response:
                yield chunk
            return

        async for chunk in response:
            try:
                yield _set_effort_on_response(chunk, requested)
            except Exception as exc:
                print(f"[REASONING-POLICY] stream restore failed open: {exc}")
                yield chunk


proxy_handler_instance = ReasoningPolicyHandler()
