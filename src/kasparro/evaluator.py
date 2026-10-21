from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .models import AgentResult, EvaluationResult, Score


@dataclass(frozen=True)
class GoldClaim:
    answer_id: str
    claim_id: str
    start_char: int
    end_char: int
    quote: str
    expected_label: str


def _metric(g: int, p: int, numerator: int, matches: int) -> Score:
    return Score(
        g,
        p,
        matches,
        numerator,
        numerator / p if p else 0.0,
        numerator / g if g else 0.0,
        2 * numerator / (p + g) if p + g else 0.0,
    )


def score_results(results: Iterable[AgentResult], gold: Iterable[GoldClaim], split: str) -> EvaluationResult:
    result_list = list(results)
    gold_list = list(gold)
    if split == "check":
        return EvaluationResult(
            split=split,
            extraction=None,
            joint=None,
            gold_coverage=None,
            answer_errors=sum(r.error_type is not None for r in result_list),
            answer_count=len(result_list),
        )
    gold_by_key = {(g.answer_id, g.start_char, g.end_char): g for g in gold_list}
    matched: set[tuple[str, int, int]] = set()
    prediction_count = 0
    correct = 0
    unmatched: list[str] = []
    wrong: list[str] = []
    for result in result_list:
        for claim in result.claims:
            prediction_count += 1
            key = (claim.answer_id, claim.start_char, claim.end_char)
            gold_claim = gold_by_key.get(key)
            if gold_claim is None or claim.quote != gold_claim.quote or key in matched:
                unmatched.append(claim.claim_id)
                continue
            matched.add(key)
            if claim.label == gold_claim.expected_label:
                correct += 1
            else:
                wrong.append(claim.claim_id)
    missed = [g.claim_id for g in gold_list if (g.answer_id, g.start_char, g.end_char) not in matched]
    coverage = correct / len(gold_list) if gold_list else 0.0
    return EvaluationResult(
        split=split,
        extraction=_metric(len(gold_list), prediction_count, len(matched), len(matched)),
        joint=_metric(len(gold_list), prediction_count, correct, len(matched)),
        gold_coverage=coverage,
        missed_gold_ids=missed,
        unmatched_prediction_ids=unmatched,
        wrong_label_ids=wrong,
        answer_errors=sum(r.error_type is not None for r in result_list),
        answer_count=len(result_list),
    )
