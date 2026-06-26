import json
from pathlib import Path

from typer.testing import CliRunner

from autocut.cli import _default_job_name, _job_paths, _safe_job_name, app
from autocut.semantic import EditPlan


def test_safe_job_name_removes_path_sensitive_characters() -> None:
    assert _safe_job_name('  demo video: 01/02?*  ') == "demo_video_01_02"


def test_default_job_name_uses_video_stem() -> None:
    assert _default_job_name(Path("inputs/01 - 猴王初问世.flv")) == "01_-_猴王初问世"


def test_job_paths_use_canonical_run_layout() -> None:
    paths = _job_paths("monkey king")

    assert paths.job == "monkey_king"
    assert paths.run_dir == Path("runs/monkey_king")
    assert paths.transcript == Path("runs/monkey_king/transcript.json")
    assert paths.llm_input == Path("runs/monkey_king/llm_input.md")
    assert paths.cut_report == Path("runs/monkey_king/cut_report.md")
    assert paths.preview_srt == Path("outputs/monkey_king.rough_cut.srt")
    assert paths.subtitle_srt == Path("outputs/monkey_king.subtitles.srt")
    assert paths.render_output == Path("outputs/monkey_king.rough_cut.mp4")


def test_apply_plan_writes_timeline_for_job(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "runs" / "demo"
    run_dir.mkdir(parents=True)
    (run_dir / "topic_blocks.json").write_text(
        json.dumps(
            {
                "blocks": [
                    {
                        "id": "B0001",
                        "start_sentence_id": "S0001",
                        "end_sentence_id": "S0001",
                        "start": 1.0,
                        "end": 3.0,
                        "text": "核心内容。",
                        "sentence_count": 1,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_dir / "edit_plan.json").write_text(
        json.dumps(
            {
                "edit_decisions": [
                    {
                        "block_id": "B0001",
                        "decision": "keep",
                        "reason": "核心内容",
                        "title": "保留片段",
                        "confidence": 0.9,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["apply-plan", "--job", "demo"])

    assert result.exit_code == 0
    timeline = json.loads((run_dir / "timeline.json").read_text(encoding="utf-8"))
    assert timeline["segments"][0]["id"] == "B0001"
    assert timeline["segments"][0]["decision"] == "keep"
    assert timeline["segments"][0]["text"] == "保留片段"
    report = (run_dir / "cut_report.md").read_text(encoding="utf-8")
    assert "Estimated rough-cut duration" in report
    assert "B0001" in report


def test_llm_plan_writes_edit_plan_for_job(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "runs" / "demo"
    run_dir.mkdir(parents=True)
    (run_dir / "llm_input.md").write_text("[B0001] 核心内容", encoding="utf-8")

    def fake_generate_edit_plan(markdown, config) -> EditPlan:
        assert "B0001" in markdown
        assert config.model == "test-model"
        return EditPlan.model_validate(
            {
                "edit_decisions": [
                    {
                        "block_id": "B0001",
                        "decision": "keep",
                        "reason": "核心内容",
                        "title": "保留片段",
                        "confidence": 0.9,
                    }
                ]
            }
        )

    monkeypatch.setattr("autocut.llm.generate_edit_plan", fake_generate_edit_plan)

    result = CliRunner().invoke(
        app,
        [
            "llm-plan",
            "--job",
            "demo",
            "--model",
            "test-model",
            "--base-url",
            "http://localhost:11434/v1",
        ],
    )

    assert result.exit_code == 0
    edit_plan = json.loads((run_dir / "edit_plan.json").read_text(encoding="utf-8"))
    assert edit_plan["edit_decisions"][0]["block_id"] == "B0001"
    assert edit_plan["edit_decisions"][0]["title"] == "保留片段"
