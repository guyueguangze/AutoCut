from autocut.report import build_cut_report
from autocut.semantic import EditPlan, TopicBlocks
from autocut.timeline import Timeline


def test_build_cut_report_summarizes_keep_and_delete() -> None:
    topic_blocks = TopicBlocks.model_validate(
        {
            "source": "input.mp4",
            "blocks": [
                {
                    "id": "B0001",
                    "start_sentence_id": "S0001",
                    "end_sentence_id": "S0001",
                    "start": 0,
                    "end": 10,
                    "text": "核心内容。",
                    "sentence_count": 1,
                },
                {
                    "id": "B0002",
                    "start_sentence_id": "S0002",
                    "end_sentence_id": "S0002",
                    "start": 10,
                    "end": 15,
                    "text": "重复内容。",
                    "sentence_count": 1,
                },
            ],
        }
    )
    edit_plan = EditPlan.model_validate(
        {
            "chapters": [
                {
                    "title": "开场",
                    "start_block_id": "B0001",
                    "end_block_id": "B0001",
                    "summary": "保留开场核心内容。",
                }
            ],
            "edit_decisions": [
                {
                    "block_id": "B0001",
                    "decision": "keep",
                    "reason": "核心内容",
                    "title": "开场",
                    "confidence": 0.9,
                },
                {
                    "block_id": "B0002",
                    "decision": "delete",
                    "reason": "重复",
                    "title": "重复段",
                    "confidence": 0.8,
                },
            ],
        }
    )
    timeline = Timeline.model_validate(
        {
            "segments": [
                {
                    "id": "B0001",
                    "start": 0,
                    "end": 10,
                    "decision": "keep",
                    "text": "开场",
                    "reason": "核心内容",
                    "confidence": 0.9,
                },
                {
                    "id": "B0002",
                    "start": 10,
                    "end": 15,
                    "decision": "delete",
                    "text": "重复段",
                    "reason": "重复",
                    "confidence": 0.8,
                },
            ]
        }
    )

    report = build_cut_report(
        job="demo",
        topic_blocks=topic_blocks,
        edit_plan=edit_plan,
        timeline=timeline,
    )

    assert "# Cut Report: demo" in report
    assert "Estimated rough-cut duration: 10s" in report
    assert "Removed duration: 5s" in report
    assert "B0001" in report
    assert "B0002" in report
