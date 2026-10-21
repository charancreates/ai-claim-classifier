from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

from .evaluator import GoldClaim
from .models import AnswerInput, BrandFact


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def compute_file_hash(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_answers(path: str | Path) -> list[AnswerInput]:
    rows = read_csv_rows(path)
    required = {"answer_id", "brand_id", "question", "ai_answer"}
    if rows and not required.issubset(rows[0]):
        raise ValueError(f"answers CSV is missing columns: {sorted(required - set(rows[0]))}")
    return [
        AnswerInput(
            answer_id=r["answer_id"],
            brand_id=r["brand_id"],
            question=r["question"],
            ai_answer=r["ai_answer"],
            split=r.get("split", "development"),
        )
        for r in rows
    ]


def _typed_value(raw: str, value_type: str) -> Any:
    if value_type == "boolean":
        if raw == "true":
            return True
        if raw == "false":
            return False
        raise ValueError(f"invalid boolean value: {raw!r}")
    if value_type == "number":
        return float(raw) if any(c in raw for c in ".eE") else int(raw)
    if value_type == "string":
        return raw
    raise ValueError(f"invalid value_type: {value_type!r}")


def load_facts(path: str | Path, fact_version: str = "speaker-catalogue-v4") -> list[BrandFact]:
    rows = read_csv_rows(path)
    required = {"fact_id", "brand_id", "model_id", "field", "value", "value_type"}
    if rows and not required.issubset(rows[0]):
        raise ValueError(f"facts CSV is missing columns: {sorted(required - set(rows[0]))}")
    return [
        BrandFact(
            fact_id=r["fact_id"],
            brand_id=r["brand_id"],
            model_id=r["model_id"],
            field=r["field"],
            value=_typed_value(r["value"], r["value_type"]),
            value_type=r["value_type"],  # type: ignore[arg-type]
            unit=r.get("unit") or None,
            condition=r.get("condition") or None,
            valid_from=r.get("valid_from") or None,
            valid_to=r.get("valid_to") or None,
            fact_version=fact_version,
        )
        for r in rows
    ]


def load_gold(path: str | Path) -> list[GoldClaim]:
    rows = read_csv_rows(path)
    required = {"answer_id", "claim_id", "start_char", "end_char", "quote", "expected_label"}
    if rows and not required.issubset(rows[0]):
        raise ValueError(f"gold CSV is missing columns: {sorted(required - set(rows[0]))}")
    return [
        GoldClaim(
            answer_id=r["answer_id"],
            claim_id=r["claim_id"],
            start_char=int(r["start_char"]),
            end_char=int(r["end_char"]),
            quote=r["quote"],
            expected_label=r["expected_label"],
        )
        for r in rows
    ]
