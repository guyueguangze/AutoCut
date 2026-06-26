import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from autocut.semantic import EditPlan


class LLMError(RuntimeError):
    """Raised when edit-plan generation fails."""


@dataclass(frozen=True)
class LLMConfig:
    model: str
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    wire_api: str = "chat_completions"
    reasoning_effort: str | None = None
    disable_response_storage: bool = True
    temperature: float = 0.2
    timeout: float = 120
    json_mode: bool = True


def config_from_env(
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    api_key_env: str = "AUTOCUT_LLM_API_KEY",
    wire_api: str | None = None,
    reasoning_effort: str | None = None,
    disable_response_storage: bool | None = None,
    temperature: float = 0.2,
    timeout: float = 120,
    json_mode: bool = True,
) -> LLMConfig:
    load_env_file(Path(".env"))
    resolved_base_url = base_url or os.getenv("AUTOCUT_LLM_BASE_URL") or LLMConfig.base_url
    resolved_model = model or os.getenv("AUTOCUT_LLM_MODEL")
    resolved_api_key = api_key or os.getenv(api_key_env) or os.getenv("OPENAI_API_KEY")
    resolved_wire_api = wire_api or os.getenv("AUTOCUT_LLM_WIRE_API") or "chat_completions"
    resolved_reasoning_effort = reasoning_effort or os.getenv("AUTOCUT_LLM_REASONING_EFFORT")
    resolved_disable_storage = (
        disable_response_storage
        if disable_response_storage is not None
        else _env_bool("AUTOCUT_LLM_DISABLE_RESPONSE_STORAGE", default=True)
    )

    if not resolved_model:
        raise LLMError("Missing LLM model. Set AUTOCUT_LLM_MODEL or pass --model.")
    if not resolved_api_key and "api.openai.com" in resolved_base_url:
        raise LLMError(
            f"Missing API key. Set {api_key_env}, AUTOCUT_LLM_API_KEY, or pass --api-key."
        )

    return LLMConfig(
        model=resolved_model,
        base_url=resolved_base_url,
        api_key=resolved_api_key,
        wire_api=_normalize_wire_api(resolved_wire_api),
        reasoning_effort=resolved_reasoning_effort,
        disable_response_storage=resolved_disable_storage,
        temperature=temperature,
        timeout=timeout,
        json_mode=json_mode,
    )


def generate_edit_plan(markdown: str, config: LLMConfig) -> EditPlan:
    content = _request_chat_completion(
        system=EDIT_PLAN_SYSTEM_PROMPT,
        user=build_edit_plan_prompt(markdown),
        config=config,
    )
    payload = extract_json_object(content)
    try:
        return EditPlan.model_validate(payload)
    except ValidationError as exc:
        raise LLMError(f"LLM returned invalid edit_plan JSON: {exc}") from exc


def build_edit_plan_prompt(markdown: str) -> str:
    return (
        "请根据下面的 transcript blocks 生成自动剪辑决策。\n\n"
        "要求：\n"
        "1. 只能使用输入里出现过的 block_id，不要编造时间戳。\n"
        "2. edit_decisions 中每个对象必须包含 block_id、decision、reason、title、confidence。\n"
        "3. decision 只能是 keep 或 delete。\n"
        "4. title 用中文短标题，适合视频章节或片段标题。\n"
        "5. confidence 是 0 到 1 的数字。\n"
        "6. 输出必须是 JSON 对象，不要输出 Markdown 或解释文字。\n\n"
        "JSON 格式：\n"
        '{ "chapters": [{ "title": "", "start_block_id": "B0001", '
        '"end_block_id": "B0001", "summary": "" }], '
        '"edit_decisions": [{ "block_id": "B0001", "decision": "keep", '
        '"reason": "", "title": "", "confidence": 0.9 }] }\n\n'
        f"{markdown}"
    )


def extract_json_object(content: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else content.strip()

    try:
        loaded = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise LLMError("LLM response does not contain a JSON object.")
        try:
            loaded = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"Could not parse LLM JSON response: {exc}") from exc

    if not isinstance(loaded, dict):
        raise LLMError("LLM response JSON must be an object.")
    return loaded


def _request_chat_completion(
    *,
    system: str,
    user: str,
    config: LLMConfig,
) -> str:
    if config.wire_api == "responses":
        return _request_response(system=system, user=user, config=config)

    payload: dict[str, object] = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": config.temperature,
    }
    if config.json_mode:
        payload["response_format"] = {"type": "json_object"}

    request = urllib.request.Request(
        _chat_completions_url(config.base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers=_headers(config),
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=config.timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise LLMError(f"LLM API HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"Could not reach LLM API: {exc}") from exc

    try:
        data = json.loads(raw)
        choices = data["choices"]
        message = choices[0]["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise LLMError("LLM API response did not match Chat Completions format.") from exc

    if not isinstance(content, str) or not content.strip():
        raise LLMError("LLM API returned empty content.")
    return content


def _request_response(
    *,
    system: str,
    user: str,
    config: LLMConfig,
) -> str:
    payload: dict[str, object] = {
        "model": config.model,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": config.temperature,
        "store": not config.disable_response_storage,
    }
    if config.reasoning_effort:
        payload["reasoning"] = {"effort": config.reasoning_effort}
    if config.json_mode:
        payload["text"] = {"format": {"type": "json_object"}}

    request = urllib.request.Request(
        _responses_url(config.base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers=_headers(config),
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=config.timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise LLMError(f"LLM API HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"Could not reach LLM API: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMError("LLM API response was not valid JSON.") from exc

    content = _response_output_text(data)
    if not content:
        raise LLMError("LLM API returned empty content.")
    return content


def _chat_completions_url(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith("/chat/completions"):
        return stripped
    return f"{stripped}/chat/completions"


def _responses_url(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith("/responses"):
        return stripped
    return f"{stripped}/responses"


def _headers(config: LLMConfig) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    return headers


def _response_output_text(data: dict) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]

    chunks = []
    for item in data.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            if isinstance(content.get("text"), str):
                chunks.append(content["text"])
            elif isinstance(content.get("content"), str):
                chunks.append(content["content"])
    return "".join(chunks)


def _normalize_wire_api(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "chat": "chat_completions",
        "chat_completions": "chat_completions",
        "responses": "responses",
        "response": "responses",
    }
    if normalized not in aliases:
        raise LLMError("LLM wire API must be chat_completions or responses.")
    return aliases[normalized]


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


EDIT_PLAN_SYSTEM_PROMPT = (
    "你是一个视频剪辑决策助手。你的任务是根据分块后的字幕内容，"
    "输出严格 JSON，用于自动粗剪。不要输出解释，不要输出 Markdown。"
)
