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

_WINDOW = 96
_NON_USER_ROLES = frozenset({"system", "developer", "assistant", "tool", "function"})

_HARMFUL_NOUN = r"malware|ransomware|keylogger|trojans?|spyware|botnets?|rootkits?|phishing"

# Deliberately not a context window and deliberately not generic security words:
# "for training", "for analysis" or "secure" next to a request must never make it
# allowed. Only a fixed compound where the defensive word binds directly to the
# harmful noun ("malware detection", "remove ransomware") suppresses a match.
_DEFENSIVE_COMPOUND = re.compile(
    rf"\b(?:{_HARMFUL_NOUN})\s+"
    r"(?:detection|detector|scanners?|signatures?|removal|recovery|mitigation|prevention|quarantine)\b"
    rf"|\b(?:detect(?:s|ing)?|prevent(?:s|ing)?|remov(?:e|es|ing|al\s+of)|mitigat(?:e|es|ing)|"
    rf"quarantin(?:e|es|ing))\s+(?:{_HARMFUL_NOUN})\b",
    re.IGNORECASE,
)

_HARMFUL_NOUN_OCCURRENCE = re.compile(rf"\b(?:{_HARMFUL_NOUN})\b", re.IGNORECASE)


def _near(left: str, right: str, window: int = _WINDOW) -> re.Pattern[str]:
    return re.compile(
        rf"(?:{left}).{{0,{window}}}?(?:{right})|(?:{right}).{{0,{window}}}?(?:{left})",
        re.IGNORECASE | re.DOTALL,
    )


_BLOCK_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "unauthorized_access",
        _near(
            r"\b(?:gain|obtain|break\s+into|take\s+over|hack(?:ing)?|compromise)\b|"
            r"\b(?:truy\s*cap|xam\s*nhap|chiem\s*quyen)\b",
            r"\b(?:without\s+(?:permission|authorization)|unauthori[sz]ed)\b|"
            r"\b(?:trai\s*phep|khong\s*(?:co\s*)?quyen|khong\s*duoc\s*phep)\b",
        ),
    ),
    (
        "unauthorized_access",
        re.compile(
            r"\bunauthori[sz]ed\s+access\b|"
            r"\b(?:truy\s*cap|xam\s*nhap)\s+(?:trai\s*phep|khong\s*(?:co\s*)?quyen)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "credential_theft",
        _near(
            r"\b(?:steal|exfiltrate|harvest|dump|phish)\b|"
            r"\b(?:danh\s*cap|trich\s*xuat)\b",
            r"\b(?:credentials?|passwords?|api[ _-]?keys?|tokens?|sessions?)\b|"
            r"\b(?:mat\s*khau|thong\s*tin\s*dang\s*nhap|api[ _-]?key|token)\b",
        ),
    ),
    (
        "malware_creation",
        _near(
            r"\b(?:create|build|write|deploy|spread|infect|distribute|craft)\b|"
            r"\b(?:tao|viet|trien\s*khai)\b",
            r"\b(?:malware|ransomware|keylogger|trojan|spyware|botnet|rootkit|rat)\b|"
            r"\b(?:ma\s*doc)\b",
        ),
    ),
    (
        "exploit_development",
        _near(
            r"\b(?:create|build|write|develop|generate|craft)\b|"
            r"\b(?:tao|viet|xay\s*dung)\b",
            r"\b(?:exploits?|exploit\s*poc|poc\s*exploit|attack\s+tools?|weaponi[sz]e)\b|"
            r"\b(?:cong\s*cu\s*tan\s*cong|ma\s*tan\s*cong)\b",
        ),
    ),
    (
        "phishing_kit",
        _near(
            r"\b(?:create|build|write|craft|clone|host)\b|"
            r"\b(?:tao|viet|dung)\b",
            r"\b(?:phishing|spear[- ]?phish(?:ing)?|credential[- ]?harvest(?:ing)?)\b|"
            r"\b(?:lua\s*dao|gia\s*mao\s*(?:trang|web|email))\b",
        ),
    ),
    (
        "security_bypass",
        _near(
            r"\b(?:bypass|evade)\b",
            r"\b(?:authentication|security\s+controls?|antivirus|edr|monitoring|detection)\b|"
            r"\b(?:xac\s*thuc|antivirus|giam\s*sat)\b",
        ),
    ),
    (
        "security_bypass",
        _near(
            r"\b(?:disable|turn\s+off)\b|"
            r"\b(?:tat|vo\s*hieu\s*hoa)\b",
            r"\b(?:authentication|security\s+controls?|monitoring|detection)\b",
        ),
    ),
    (
        "attack_automation",
        _near(
            r"\b(?:automat(?:e|ing|ion)|script(?:ing)?)\b|"
            r"\b(?:tu\s*dong\s*hoa)\b",
            r"\b(?:attacks?|attacking|exploits?|exploitation|bruteforce|brute[- ]force|"
            r"credential\s+stuffing)\b|"
            r"\b(?:tan\s*cong|bruteforce)\b",
        ),
    ),
    (
        "attack_automation",
        re.compile(
            r"\b(?:scan(?:ning)?\s+and\s+attack|attack\s+these\s+hosts|"
            r"mass[- ]?(?:scan|attack)|quet\s+lo\s+hong.{0,96}tan\s*cong)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
)


def _fold_diacritics(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value)
    without_marks = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return without_marks.replace("đ", "d").replace("Đ", "d")


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = normalized.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    normalized = _fold_diacritics(normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _iter_text_values(value: Any) -> Iterable[str]:
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


def _is_user_authored_item(item: dict[str, Any]) -> bool:
    role = item.get("role")
    return role not in _NON_USER_ROLES


def request_text(data: dict[str, Any]) -> str:
    """Extract only user-authored text so system/developer safety prompts cannot self-trip."""
    parts: list[str] = []

    messages = data.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and _is_user_authored_item(message):
                parts.extend(_iter_text_values(message.get("content")))

    input_value = data.get("input")
    if isinstance(input_value, str):
        parts.append(input_value)
    elif isinstance(input_value, list):
        for item in input_value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and _is_user_authored_item(item):
                parts.extend(_iter_text_values(item))

    prompt = data.get("prompt")
    if isinstance(prompt, str):
        parts.append(prompt)

    return "\n".join(part for part in parts if part)


def _overlaps(match: re.Match[str], compounds: tuple[re.Match[str], ...]) -> bool:
    return any(compound.start() < match.end() and compound.end() > match.start() for compound in compounds)


def _every_harmful_noun_is_defensive(normalized: str, compounds: tuple[re.Match[str], ...]) -> bool:
    """True only when the text never names a harmful artifact outside a defensive compound.

    One unbound mention ("malware detection rules, then a ransomware dropper")
    keeps the request blockable, so a defensive prefix cannot launder the rest.
    """
    if not compounds:
        return False
    return all(
        any(compound.start() <= noun.start() and compound.end() >= noun.end() for compound in compounds)
        for noun in _HARMFUL_NOUN_OCCURRENCE.finditer(normalized)
    )


def blocking_rule(data: dict[str, Any]) -> Optional[str]:
    normalized = _normalize_text(request_text(data))
    if not normalized:
        return None
    compounds = tuple(_DEFENSIVE_COMPOUND.finditer(normalized))
    defensive_only = _every_harmful_noun_is_defensive(normalized, compounds)
    for rule_name, pattern in _BLOCK_RULES:
        for match in pattern.finditer(normalized):
            if defensive_only and _overlaps(match, compounds):
                continue
            return rule_name
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
                item for item in input_value if isinstance(item.get("role"), str) and "content" in item
            ]
            if response_messages:
                return litellm.token_counter(model=model, messages=response_messages)
        if input_value:
            return litellm.token_counter(
                model=model,
                text=json.dumps(input_value, ensure_ascii=False, separators=(",", ":"), default=str),
            )
    except Exception:
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
