"""Fast, request-only cyber-abuse guard for the production LiteLLM proxy.

The guard runs before routing, so a blocked request never reaches an upstream
provider.  It deliberately returns a ``ModifyResponseException`` instead of a
4xx error: the proxy turns that into an OpenAI-compatible HTTP 200 refusal.

The synthetic refusal carries token usage measured from the actual inbound
payload and the refusal text.  This is local gateway usage, not upstream
provider usage (the provider was never called).
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Iterable, Optional

import litellm
from litellm.exceptions import ModifyResponseException
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import Usage


GUARDRAIL_NAME = "shortlab-cyber-abuse"
REFUSAL_MESSAGE = (
    "Xin lỗi, tôi không thể hỗ trợ yêu cầu này. Tôi có thể hỗ trợ các phương án "
    "bảo mật phòng thủ, hợp pháp và có ủy quyền."
)

# These are intentionally high-confidence combinations. The guard is not a
# keyword blacklist: defensive requests such as code review and mitigation
# remain on the normal route. Broader / ambiguous classifications belong in a
# separately reviewed semantic classifier, not this low-latency hot path.
_BLOCK_PATTERNS = (
    re.compile(
        r"\b(?:gain|obtain|break into|take over|access)\b.{0,96}"
        r"\b(?:without (?:permission|authorization)|unauthorized)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(?:steal|exfiltrate|harvest|dump)\b.{0,96}"
        r"\b(?:credentials?|passwords?|api[ _-]?keys?|tokens?)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(?:create|build|write|deploy)\b.{0,96}"
        r"\b(?:malware|ransomware|keylogger)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(?:bypass|disable|evade)\b.{0,96}"
        r"\b(?:authentication|security controls?|antivirus|monitoring)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(?:truy\s*cập|xâm\s*nhập|chiếm\s*quyền)\b.{0,96}"
        r"\b(?:trái\s*phép|không\s*(?:có\s*)?quyền|không\s*được\s*phép)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(?:đánh\s*cắp|trích\s*xuất)\b.{0,96}"
        r"\b(?:mật\s*khẩu|thông\s*tin\s*đăng\s*nhập|api[ _-]?key|token)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(?:tạo|viết|triển\s*khai)\b.{0,96}"
        r"\b(?:mã\s*độc|ransomware|keylogger)\b",
        re.IGNORECASE | re.DOTALL,
    ),
)


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = normalized.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    return re.sub(r"\s+", " ", normalized).strip()


def _iter_text_values(value: Any) -> Iterable[str]:
    """Yield textual request content without inspecting model/config metadata."""
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, list):
        for item in value:
            yield from _iter_text_values(item)
        return
    if isinstance(value, dict):
        for key in ("content", "input", "text", "arguments"):
            if key in value:
                yield from _iter_text_values(value[key])


def request_text(data: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("messages", "input", "prompt", "tools"):
        if key in data:
            parts.extend(_iter_text_values(data[key]))
    return "\n".join(part for part in parts if part)


def blocking_rule(data: dict[str, Any]) -> Optional[str]:
    normalized = _normalize_text(request_text(data))
    if not normalized:
        return None
    for index, pattern in enumerate(_BLOCK_PATTERNS, start=1):
        if pattern.search(normalized):
            return f"high_confidence_cyber_rule_{index}"
    return None


def _count_input_tokens(data: dict[str, Any], model: str) -> int:
    try:
        messages = data.get("messages")
        if isinstance(messages, list):
            return litellm.token_counter(
                model=model,
                messages=messages,
                tools=data.get("tools") if isinstance(data.get("tools"), list) else None,
                tool_choice=data.get("tool_choice"),
            )

        input_value = data.get("input", data.get("prompt", ""))
        if isinstance(input_value, str):
            return litellm.token_counter(model=model, text=input_value)
        if isinstance(input_value, list) and all(isinstance(item, dict) for item in input_value):
            response_messages = [
                item
                for item in input_value
                if isinstance(item.get("role"), str) and "content" in item
            ]
            if response_messages:
                return litellm.token_counter(model=model, messages=response_messages)
        if input_value:
            return litellm.token_counter(
                model=model,
                text=json.dumps(input_value, ensure_ascii=False, separators=(",", ":"), default=str),
            )
    except Exception:
        # A guard must never fail open merely because a provider-specific
        # tokenizer is unavailable. The response remains valid with a zero
        # local count, and the request remains blocked.
        return 0
    return 0


def measured_refusal_usage(data: dict[str, Any], refusal_message: str = REFUSAL_MESSAGE) -> Usage:
    """Return locally measured input + synthetic refusal usage for one request."""
    model = str(data.get("model") or "")
    prompt_tokens = _count_input_tokens(data, model)
    try:
        completion_tokens = litellm.token_counter(
            model=model,
            text=refusal_message,
            count_response_tokens=True,
        )
    except Exception:
        completion_tokens = 0
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def _usage_carrier(usage: Usage) -> Any:
    response = litellm.ModelResponse()
    response.usage = usage
    return response


class CyberAbuseGuard(CustomLogger):
    """Block only the current request; API keys remain active."""

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict[str, Any],
        call_type: str,
    ) -> dict[str, Any]:
        rule = blocking_rule(data)
        if rule is None:
            return data

        metadata = data.get("litellm_metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            data["litellm_metadata"] = metadata
        metadata["cyber_abuse_guard"] = {
            "blocked": True,
            "rule": rule,
            "upstream_called": False,
            "local_usage_measured": True,
        }

        raise ModifyResponseException(
            message=REFUSAL_MESSAGE,
            model=str(data.get("model") or ""),
            request_data=data,
            guardrail_name=GUARDRAIL_NAME,
            detection_info={"rule": rule, "upstream_called": False},
            original_response=_usage_carrier(measured_refusal_usage(data)),
        )


proxy_handler_instance = CyberAbuseGuard()
