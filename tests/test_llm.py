import pytest

from autocut.llm import (
    LLMConfig,
    LLMError,
    _response_output_text,
    config_from_env,
    extract_json_object,
    generate_edit_plan,
)


def test_extract_json_object_from_fenced_response() -> None:
    payload = extract_json_object(
        """
        ```json
        {"edit_decisions": [{"block_id": "B0001", "decision": "keep"}]}
        ```
        """
    )

    assert payload["edit_decisions"][0]["block_id"] == "B0001"


def test_extract_json_object_rejects_non_json() -> None:
    with pytest.raises(LLMError):
        extract_json_object("not json")


def test_generate_edit_plan_validates_llm_response(monkeypatch) -> None:
    def fake_request_chat_completion(**kwargs) -> str:
        return (
            '{"chapters": [{"title": "开场", "start_block_id": "B0001", '
            '"end_block_id": "B0001", "summary": "说明开场"}], '
            '"edit_decisions": [{"block_id": "B0001", "decision": "keep", '
            '"reason": "核心内容", "title": "开场", "confidence": 0.9}]}'
        )

    monkeypatch.setattr("autocut.llm._request_chat_completion", fake_request_chat_completion)

    plan = generate_edit_plan("B0001: 内容", LLMConfig(model="test-model"))

    assert plan.chapters[0].title == "开场"
    assert plan.edit_decisions[0].decision == "keep"
    assert plan.edit_decisions[0].confidence == 0.9


def test_config_from_env_loads_dotenv_for_responses(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AUTOCUT_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("AUTOCUT_LLM_MODEL", raising=False)
    monkeypatch.delenv("AUTOCUT_LLM_WIRE_API", raising=False)
    monkeypatch.delenv("AUTOCUT_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "AUTOCUT_LLM_BASE_URL=https://example.test",
                "AUTOCUT_LLM_MODEL=test-model",
                "AUTOCUT_LLM_WIRE_API=responses",
                "AUTOCUT_LLM_REASONING_EFFORT=xhigh",
                "AUTOCUT_LLM_DISABLE_RESPONSE_STORAGE=true",
                "OPENAI_API_KEY=test-key",
            ]
        ),
        encoding="utf-8",
    )

    config = config_from_env()

    assert config.base_url == "https://example.test"
    assert config.model == "test-model"
    assert config.wire_api == "responses"
    assert config.api_key == "test-key"
    assert config.reasoning_effort == "xhigh"
    assert config.disable_response_storage is True


def test_response_output_text_reads_responses_shape() -> None:
    content = _response_output_text(
        {
            "output": [
                {
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"edit_decisions": []}',
                        }
                    ]
                }
            ]
        }
    )

    assert content == '{"edit_decisions": []}'
