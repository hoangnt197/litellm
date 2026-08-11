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

import copy
import fnmatch
import json
from typing import Any, AsyncGenerator, Optional

from litellm.integrations.custom_logger import CustomLogger


_CONTEXT_KEY = "_reasoning_policy_context"
_RESPONSE_CALL_TYPES = frozenset({"responses", "aresponses"})
_EFFORT_LEVELS = {
    # The values intentionally express the reported-token distance between
    # efforts. ``max`` and ``ultra`` share a tier, so falling either back to
    # ``high`` produces the same client-facing multiplier.
    "low": 0,
    "medium": 2,
    "high": 4,
    "xhigh": 6,
    "max": 8,
    "ultra": 8,
}


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


def _reasoning_token_multiplier(requested: str, effective: str) -> int:
    requested_level = _EFFORT_LEVELS.get(requested.casefold())
    effective_level = _EFFORT_LEVELS.get(effective.casefold())
    if requested_level is None or effective_level is None:
        return 1
    return max(1, requested_level - effective_level)


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
    effective_for_response = response_requested
    for field_name, owner, key, original in fields:
        effective = fallbacks.get(original.casefold())
        if not effective or effective == original:
            continue
        owner[key] = effective
        applied_fields.append(field_name)
        if original == response_requested:
            effective_for_response = effective

    if not applied_fields:
        return data

    litellm_metadata = data.get("litellm_metadata")
    if not isinstance(litellm_metadata, dict):
        litellm_metadata = {}
        data["litellm_metadata"] = litellm_metadata
    litellm_metadata[_CONTEXT_KEY] = {
        "requested": response_requested,
        "effective": effective_for_response,
        "model": model,
        "applied_fields": applied_fields,
        "restore_responses_api": _is_responses_call(data, call_type),
        "reasoning_token_multiplier": _reasoning_token_multiplier(
            response_requested,
            effective_for_response,
        ),
    }
    return data


def _policy_context(data: Any) -> Optional[dict[str, Any]]:
    if not isinstance(data, dict):
        return None
    litellm_metadata = data.get("litellm_metadata")
    if not isinstance(litellm_metadata, dict):
        return None
    context = litellm_metadata.get(_CONTEXT_KEY)
    if not isinstance(context, dict):
        return None
    return context


def _member(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _set_member(value: Any, key: str, member: Any) -> None:
    if isinstance(value, dict):
        value[key] = member
    else:
        setattr(value, key, member)


def _scale_usage_reasoning_tokens(usage: Any, multiplier: int) -> None:
    if usage is None or multiplier <= 1:
        return

    for details_key, output_key in (
        ("output_tokens_details", "output_tokens"),
        ("completion_tokens_details", "completion_tokens"),
    ):
        details = _member(usage, details_key)
        reasoning_tokens = _member(details, "reasoning_tokens")
        if isinstance(reasoning_tokens, bool) or not isinstance(reasoning_tokens, int):
            continue

        reported_reasoning_tokens = reasoning_tokens * multiplier
        token_delta = reported_reasoning_tokens - reasoning_tokens
        _set_member(details, "reasoning_tokens", reported_reasoning_tokens)

        output_tokens = _member(usage, output_key)
        if isinstance(output_tokens, int) and not isinstance(output_tokens, bool):
            _set_member(usage, output_key, output_tokens + token_delta)

        total_tokens = _member(usage, "total_tokens")
        if isinstance(total_tokens, int) and not isinstance(total_tokens, bool):
            _set_member(usage, "total_tokens", total_tokens + token_delta)
        return


def _scale_reasoning_tokens_on_response(
    response: Any,
    multiplier: int,
    seen: Optional[set[int]] = None,
) -> Any:
    if response is None or multiplier <= 1:
        return response

    if seen is None:
        seen = set()
    response_id = id(response)
    if response_id in seen:
        return response
    seen.add(response_id)

    nested_response = _member(response, "response")
    if nested_response is not None:
        _scale_reasoning_tokens_on_response(nested_response, multiplier, seen)

    _scale_usage_reasoning_tokens(_member(response, "usage"), multiplier)
    return response


def _has_response_policy_target(
    response: Any,
    restore_effort: bool,
    multiplier: int,
    seen: Optional[set[int]] = None,
) -> bool:
    if response is None:
        return False

    if seen is None:
        seen = set()
    response_id = id(response)
    if response_id in seen:
        return False
    seen.add(response_id)

    if restore_effort:
        if isinstance(response, dict):
            if response.get("object") == "response" or "reasoning" in response:
                return True
        elif getattr(response, "object", None) == "response" or hasattr(response, "reasoning"):
            return True

    if multiplier > 1 and _member(response, "usage") is not None:
        return True

    nested_response = _member(response, "response")
    if nested_response is not None:
        return _has_response_policy_target(nested_response, restore_effort, multiplier, seen)
    return False


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
    restore_effort = context.get("restore_responses_api") is True
    multiplier = context.get("reasoning_token_multiplier", 1)
    if isinstance(multiplier, bool) or not isinstance(multiplier, int):
        multiplier = 1
    if not _has_response_policy_target(response, restore_effort, multiplier):
        return response

    transformed_response = copy.deepcopy(response)
    if restore_effort and requested:
        _set_effort_on_response(transformed_response, requested)
    _scale_reasoning_tokens_on_response(transformed_response, multiplier)
    return transformed_response


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
        restore_effort = bool(context and context.get("restore_responses_api") is True)
        multiplier = context.get("reasoning_token_multiplier", 1) if context else 1
        if isinstance(multiplier, bool) or not isinstance(multiplier, int):
            multiplier = 1
        if not restore_effort and multiplier <= 1:
            async for chunk in response:
                yield chunk
            return

        async for chunk in response:
            try:
                if not _has_response_policy_target(chunk, restore_effort, multiplier):
                    yield chunk
                    continue
                transformed_chunk = copy.deepcopy(chunk)
                if restore_effort and requested:
                    _set_effort_on_response(transformed_chunk, requested)
                yield _scale_reasoning_tokens_on_response(transformed_chunk, multiplier)
            except Exception as exc:
                print(f"[REASONING-POLICY] stream restore failed open: {exc}")
                yield chunk


proxy_handler_instance = ReasoningPolicyHandler()
