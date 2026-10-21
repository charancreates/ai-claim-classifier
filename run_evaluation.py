from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from kasparro.agent import Agent
from kasparro.database import AnswerOutcome, ClaimOutcome, EvaluationRun, make_session_factory
from kasparro.evaluator import score_results
from kasparro.models import AgentResult, AnswerInput
from kasparro.parsing import compute_file_hash, load_answers, load_facts, load_gold
from kasparro.provider import create_provider


def run_evaluation(
    agent_version: str,
    split: str,
    data_dir: str = "data",
    output_dir: str = "reports",
    save_exchange: bool = False,
) -> dict[str, Any]:
    data_path = Path(data_dir)
    answers_file = data_path / "answers.csv"
    facts_file = data_path / "brand_facts.csv"
    gold_file = data_path / "development_gold.csv"

    all_answers = load_answers(answers_file)
    facts = load_facts(facts_file)
    gold = load_gold(gold_file) if gold_file.exists() and split == "development" else []

    target_answers = [a for a in all_answers if a.split == split]
    if not target_answers:
        if split == "development":
            target_answers = [a for a in all_answers if int(a.answer_id[1:]) <= 40]
        else:
            target_answers = [a for a in all_answers if int(a.answer_id[1:]) > 40]

    provider = create_provider(
        provider_type=os.getenv("LLM_PROVIDER", "gemini"),
        api_key=os.getenv("LLM_API_KEY"),
        model=os.getenv("LLM_MODEL", "gemini-flash-lite-latest"),
    )
    timeout_sec = float(os.getenv("PROVIDER_TIMEOUT_SECONDS", "30"))
    agent = Agent(provider, facts, version=agent_version, timeout_seconds=timeout_sec)

    file_hashes = {
        "answers_hash": compute_file_hash(answers_file) if answers_file.exists() else None,
        "facts_hash": compute_file_hash(facts_file) if facts_file.exists() else None,
        "gold_hash": compute_file_hash(gold_file) if gold_file.exists() else None,
    }

    metadata = {
        "split": split,
        "agent_version": agent_version,
        "dataset_version": "long-brand-answers-50-v4",
        "fact_version": "speaker-catalogue-v4",
        "evaluator_version": "sentence-spans-and-labels-v4",
        "provider": os.getenv("LLM_PROVIDER", "gemini"),
        "model": os.getenv("LLM_MODEL", "gemini-flash-lite-latest"),
        "timeout_seconds": timeout_sec,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hashes": file_hashes,
    }

    print(f"\n=======================================================")
    print(f"Starting Evaluation Run: Version={agent_version}, Split={split}")
    print(f"Target answers count: {len(target_answers)}")
    print(f"=======================================================")

    sessions = make_session_factory() if os.getenv("DATABASE_URL") else None
    run_id = None
    if sessions:
        with sessions.begin() as db:
            run_row = EvaluationRun(
                split=split,
                agent_version=agent_version,
                status="processing",
                metadata_json=metadata,
            )
            db.add(run_row)
            db.flush()
            run_id = str(run_row.id)

    results: list[AgentResult] = []
    answer_reports = []

    for idx, ans in enumerate(target_answers, 1):
        t0 = time.perf_counter()
        res = agent.classify(ans)
        latency = round((time.perf_counter() - t0) * 1000)
        results.append(res)

        print(f"[{idx}/{len(target_answers)}] {ans.answer_id} ({ans.brand_id}): {len(res.claims)} claims, {res.turns} turns, error={res.error_type} ({latency}ms)")

        ans_dict = {
            "answer_id": ans.answer_id,
            "status": "error" if res.error_type else "completed",
            "error_type": res.error_type,
            "error_message": res.error_message,
            "latency_ms": latency,
            "claims": [
                {
                    "claim_id": c.claim_id,
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                    "quote": c.quote,
                    "label": c.label,
                    "fact_ids": list(c.fact_ids),
                    "reason": c.reason,
                }
                for c in sorted(res.claims, key=lambda x: x.start_char)
            ],
        }
        answer_reports.append(ans_dict)

        # Save into PostgreSQL
        if sessions and run_id:
            with sessions.begin() as db:
                ans_outcome = AnswerOutcome(
                    run_id=run_id,
                    answer_id=ans.answer_id,
                    status="error" if res.error_type else "completed",
                    error_type=res.error_type,
                    error_message=res.error_message,
                    latency_ms=latency,
                )
                db.add(ans_outcome)
                db.flush()

                for c in res.claims:
                    db.add(
                        ClaimOutcome(
                            answer_outcome_id=ans_outcome.id,
                            run_id=run_id,
                            answer_id=c.answer_id,
                            start_char=c.start_char,
                            end_char=c.end_char,
                            quote=c.quote,
                            label=c.label,
                            fact_ids=list(c.fact_ids),
                            reason=c.reason,
                        )
                    )

        time.sleep(2.0)

    # Score run
    if split == "development":
        eval_metrics = score_results(results, gold, "development")
        scores_dict = {
            "extraction": eval_metrics.extraction.__dict__ if eval_metrics.extraction else None,
            "joint": eval_metrics.joint.__dict__ if eval_metrics.joint else None,
            "gold_coverage": eval_metrics.gold_coverage,
            "missed_gold_ids": eval_metrics.missed_gold_ids,
            "unmatched_prediction_ids": eval_metrics.unmatched_prediction_ids,
            "wrong_label_ids": eval_metrics.wrong_label_ids,
            "answer_errors": eval_metrics.answer_errors,
            "answer_count": eval_metrics.answer_count,
        }
        print(f"\n--- Metrics for {agent_version} on {split} ---")
        if eval_metrics.extraction:
            print(f"Extraction: Precision={eval_metrics.extraction.precision:.4f}, Recall={eval_metrics.extraction.recall:.4f}, F1={eval_metrics.extraction.f1:.4f}")
        if eval_metrics.joint:
            print(f"Joint:      Precision={eval_metrics.joint.precision:.4f}, Recall={eval_metrics.joint.recall:.4f}, F1={eval_metrics.joint.f1:.4f}")
        print(f"Gold Claim Coverage (C/G): {eval_metrics.gold_coverage:.4f}")
        print(f"Missed Gold: {len(eval_metrics.missed_gold_ids)}, Unmatched Pred: {len(eval_metrics.unmatched_prediction_ids)}, Wrong Labels: {len(eval_metrics.wrong_label_ids)}, Errors: {eval_metrics.answer_errors}")
    else:
        scores_dict = None
        print(f"\n--- Check split completed: quality metrics stay null ---")

    if sessions and run_id:
        with sessions.begin() as db:
            active_run = db.get(EvaluationRun, run_id)
            if active_run:
                active_run.status = "completed"
                active_run.scores = scores_dict

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / f"{agent_version}_{split}.json"

    report_payload = {
        "run_id": run_id,
        "metadata": metadata,
        "metrics": scores_dict,
        "answers": answer_reports,
    }

    with report_file.open("w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    print(f"Saved run report to {report_file}")

    return report_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kasparro evaluation runner")
    parser.add_argument("--version", choices=["v1", "v2"], default="v1", help="Agent version")
    parser.add_argument("--split", choices=["development", "check"], default="development", help="Dataset split")
    parser.add_argument("--all", action="store_true", help="Run all 4 runs (v1_dev, v2_dev, v1_check, v2_check)")
    args = parser.parse_args()

    if args.all:
        print("Running all four runs sequentially...")
        for v in ["v1", "v2"]:
            for s in ["development", "check"]:
                run_evaluation(v, s)
                time.sleep(2)
    else:
        run_evaluation(args.version, args.split)
