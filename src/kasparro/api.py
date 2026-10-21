from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from .agent import Agent
from .database import AnswerOutcome, ClaimOutcome, EvaluationRun, make_session_factory
from .evaluator import score_results
from .models import AgentResult, AnswerInput
from .parsing import compute_file_hash, load_answers, load_gold


class ClassifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer_id: str
    brand_id: str
    question: str
    ai_answer: str
    agent_version: str = Field(pattern="^v[12]$")


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent_version: str = Field(pattern="^v[12]$")
    split: str = Field(pattern="^(development|check)$")


def create_app(agent: Agent, session_factory=None) -> FastAPI:
    app = FastAPI(title="Kasparro FSE Assessment")
    sessions = session_factory

    @app.post("/classify")
    def classify(request: ClassifyRequest) -> dict[str, Any]:
        target_agent = (
            agent
            if agent.version == request.agent_version
            else Agent(agent.provider, agent.facts, version=request.agent_version, timeout_seconds=agent.timeout_seconds)
        )
        if not any(f.brand_id == request.brand_id for f in target_agent.facts):
            raise HTTPException(status_code=404, detail="unknown brand")

        started = time.perf_counter()
        result = target_agent.classify(
            AnswerInput(
                answer_id=request.answer_id,
                brand_id=request.brand_id,
                question=request.question,
                ai_answer=request.ai_answer,
            )
        )
        if result.error_type:
            code = 504 if result.error_type == "timeout" else 502
            raise HTTPException(status_code=code, detail={"type": result.error_type, "message": result.error_message})

        return {
            "answer_id": result.answer_id,
            "claims": [c.__dict__ for c in result.claims],
            "latency_ms": round((time.perf_counter() - started) * 1000),
        }

    @app.post("/evaluation-runs", status_code=status.HTTP_201_CREATED)
    def start_run(request: EvaluationRequest) -> dict[str, Any]:
        if sessions is None:
            raise HTTPException(status_code=503, detail="database is not configured")

        data_dir = Path(os.getenv("DATA_DIR", "data"))
        answers_file = data_dir / "answers.csv"
        facts_file = data_dir / "brand_facts.csv"
        gold_file = data_dir / "development_gold.csv"

        if not answers_file.exists():
            raise HTTPException(status_code=500, detail="answers.csv not found")

        all_answers = load_answers(answers_file)
        target_answers = [a for a in all_answers if a.split == request.split]
        if not target_answers:
            # Fallback by ID prefix if split column absent
            if request.split == "development":
                target_answers = [a for a in all_answers if int(a.answer_id[1:]) <= 40]
            else:
                target_answers = [a for a in all_answers if int(a.answer_id[1:]) > 40]

        run_agent = Agent(
            agent.provider,
            agent.facts,
            version=request.agent_version,
            timeout_seconds=agent.timeout_seconds,
        )

        file_hashes = {
            "answers_hash": compute_file_hash(answers_file) if answers_file.exists() else None,
            "facts_hash": compute_file_hash(facts_file) if facts_file.exists() else None,
            "gold_hash": compute_file_hash(gold_file) if gold_file.exists() else None,
        }

        run = EvaluationRun(
            split=request.split,
            agent_version=request.agent_version,
            status="processing",
            metadata_json={
                "split": request.split,
                "agent_version": request.agent_version,
                "dataset_version": "long-brand-answers-50-v4",
                "fact_version": "speaker-catalogue-v4",
                "evaluator_version": "sentence-spans-and-labels-v4",
                "provider": os.getenv("LLM_PROVIDER", "configured"),
                "model": os.getenv("LLM_MODEL", "gemini-flash-lite-latest"),
                "timeout_seconds": agent.timeout_seconds,
                "hashes": file_hashes,
            },
        )

        with sessions.begin() as db:
            db.add(run)
            db.flush()
            run_id = run.id

        agent_results: list[AgentResult] = []
        with sessions.begin() as db:
            active_run = db.get(EvaluationRun, run_id)
            for item in target_answers:
                t0 = time.perf_counter()
                res = run_agent.classify(item)
                latency = round((time.perf_counter() - t0) * 1000)
                agent_results.append(res)

                ans_outcome = AnswerOutcome(
                    run_id=run_id,
                    answer_id=item.answer_id,
                    status="error" if res.error_type else "completed",
                    error_type=res.error_type,
                    error_message=res.error_message,
                    latency_ms=latency,
                )
                db.add(ans_outcome)
                db.flush()

                for c in res.claims:
                    claim_outcome = ClaimOutcome(
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
                    db.add(claim_outcome)

            if request.split == "development" and gold_file.exists():
                gold_claims = load_gold(gold_file)
                eval_metrics = score_results(agent_results, gold_claims, "development")
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
            else:
                eval_metrics = score_results(agent_results, [], "check")
                scores_dict = None

            active_run.status = "completed"
            active_run.scores = scores_dict

        return {
            "run_id": str(run_id),
            "status": "completed",
            "split": request.split,
            "agent_version": request.agent_version,
            "answer_count": len(target_answers),
            "metrics": scores_dict,
        }

    @app.get("/evaluation-runs/{run_id}")
    def get_run(run_id: UUID) -> dict[str, Any]:
        if sessions is None:
            raise HTTPException(status_code=503, detail="database is not configured")
        with sessions() as db:
            run = db.get(EvaluationRun, run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="evaluation run not found")
            return {
                "run_id": str(run.id),
                "split": run.split,
                "agent_version": run.agent_version,
                "status": run.status,
                "metadata": run.metadata_json,
                "metrics": run.scores,
                "answers": [
                    {
                        "answer_id": a.answer_id,
                        "status": a.status,
                        "error_type": a.error_type,
                        "error_message": a.error_message,
                        "latency_ms": a.latency_ms,
                        "claims": [
                            {
                                "start_char": c.start_char,
                                "end_char": c.end_char,
                                "quote": c.quote,
                                "label": c.label,
                                "fact_ids": c.fact_ids,
                                "reason": c.reason,
                            }
                            for c in sorted(a.claims, key=lambda x: x.start_char)
                        ],
                    }
                    for a in sorted(run.answers, key=lambda x: x.answer_id)
                ],
            }

    return app
