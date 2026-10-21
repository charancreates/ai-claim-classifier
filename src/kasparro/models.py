from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Label = Literal["supported", "contradicted", "unknown"]


@dataclass(frozen=True)
class AnswerInput:
    answer_id: str
    brand_id: str
    question: str
    ai_answer: str
    split: str = "development"


@dataclass(frozen=True)
class BrandFact:
    fact_id: str
    brand_id: str
    model_id: str
    field: str
    value: Any
    value_type: Literal["number", "boolean", "string"]
    unit: str | None = None
    condition: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    fact_version: str = "speaker-catalogue-v4"


@dataclass(frozen=True)
class ProposedClaim:
    quote: str
    label: Label
    fact_ids: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class ValidatedClaim:
    answer_id: str
    claim_id: str
    start_char: int
    end_char: int
    quote: str
    label: Label
    fact_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class AgentResult:
    answer_id: str
    claims: tuple[ValidatedClaim, ...] = ()
    error_type: str | None = None
    error_message: str | None = None
    turns: int = 0


@dataclass(frozen=True)
class Score:
    gold: int
    predictions: int
    matches: int
    correct: int
    precision: float
    recall: float
    f1: float


@dataclass
class EvaluationResult:
    split: str
    extraction: Score | None
    joint: Score | None
    gold_coverage: float | None = None
    missed_gold_ids: list[str] = field(default_factory=list)
    unmatched_prediction_ids: list[str] = field(default_factory=list)
    wrong_label_ids: list[str] = field(default_factory=list)
    answer_errors: int = 0
    answer_count: int = 0
