import json

import pytest

from sregym.conductor.oracles.llm_as_a_judge.judge import (
    ChecklistParseError,
    DiagnosisJudge,
)

EXPECTED_IDS = ["D1-Q1", "D1-Q2"]


def _response():
    return [
        {"id": "D1-Q1", "answer": "Yes", "evidence": "cause", "confidence": "High"},
        {"id": "D1-Q2", "answer": "No", "evidence": "scope", "confidence": "Medium"},
    ]


def test_parser_accepts_reasoning_after_json_array():
    payload = json.dumps(_response()) + "\n\nThe JSON above is my final answer."
    assert DiagnosisJudge._parse_response(payload, EXPECTED_IDS) == _response()


def test_parser_accepts_reasoning_before_fenced_json_array():
    payload = "I checked the evidence first.\n```json\n" + json.dumps(_response()) + "\n```"
    assert DiagnosisJudge._parse_response(payload, EXPECTED_IDS) == _response()


def test_parser_still_rejects_missing_questions():
    with pytest.raises(ChecklistParseError, match="Missing"):
        DiagnosisJudge._parse_response(json.dumps(_response()[:1]), EXPECTED_IDS)
