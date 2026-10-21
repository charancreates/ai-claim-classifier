from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from .models import AgentResult, AnswerInput, BrandFact, ValidatedClaim
from .prompts import build_prompt


class ProviderError(Exception):
    def __init__(self, error_type: str, message: str):
        super().__init__(message)
        self.error_type = error_type


class Provider(Protocol):
    def complete(self, prompt: str, tools: list[dict[str, Any]], timeout_seconds: float) -> dict[str, Any]:
        ...


@dataclass
class FactTool:
    facts: list[BrandFact]

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        if tool_name != "get_brand_facts":
            raise ProviderError("illegal_tool", f"The requested tool '{tool_name}' is not permitted.")
        brand_id = arguments.get("brand_id") if isinstance(arguments, dict) else None
        if not isinstance(brand_id, str) or not any(f.brand_id == brand_id for f in self.facts):
            raise ProviderError("unknown_brand", f"The requested brand '{brand_id}' is not in the supplied catalogue.")
        return [fact.__dict__ for fact in self.facts if fact.brand_id == brand_id]


def _count_sentences(text: str) -> int:
    clean = text.strip()
    if not clean:
        return 0
    # Split on sentence ending punctuation followed by whitespace, avoiding splits on abbreviations/digits
    sentences = [s for s in re.split(r'(?<=[.!?])\s+', clean) if s]
    return len(sentences)


def validate_claims(
    answer: AnswerInput,
    raw_claims: Iterable[dict[str, Any]],
    known_fact_ids: set[str],
) -> tuple[ValidatedClaim, ...]:
    temp_claims: list[dict[str, Any]] = []
    cursor = 0
    for raw in raw_claims:
        if not isinstance(raw, dict):
            raise ValueError("each claim must be an object")
        quote = raw.get("quote")
        label = raw.get("label")
        fact_ids = raw.get("fact_ids", [])
        reason = raw.get("reason", "")

        if not isinstance(quote, str) or not quote or quote != quote.strip():
            raise ValueError("quote must be non-empty and have no leading or trailing whitespace")
        if label not in {"supported", "contradicted", "unknown"}:
            raise ValueError(f"invalid label '{label}'; must be supported, contradicted, or unknown")
        if not isinstance(fact_ids, (list, tuple)) or any(not isinstance(f, str) or f not in known_fact_ids for f in fact_ids):
            raise ValueError("claim contains an invalid fact ID")
        if not isinstance(reason, str) or _count_sentences(reason) > 2:
            raise ValueError("claim reason must contain at most two sentences")

        # Locate quote in ai_answer
        start = answer.ai_answer.find(quote, cursor)
        if start < 0:
            start = answer.ai_answer.find(quote, 0)
        if start < 0:
            raise ValueError(f"quote does not occur in the answer: {quote!r}")
        end = start + len(quote)
        if answer.ai_answer[start:end] != quote:
            raise ValueError("quote offset substring mismatch")

        temp_claims.append({
            "start": start,
            "end": end,
            "quote": quote,
            "label": label,
            "fact_ids": tuple(fact_ids),
            "reason": reason,
        })
        cursor = end

    # Sort claims by start_char to ensure strict source order
    temp_claims.sort(key=lambda x: (x["start"], x["end"]))

    result: list[ValidatedClaim] = []
    last_end = 0
    seen_ids: set[str] = set()
    for item in temp_claims:
        start, end = item["start"], item["end"]
        if start < last_end:
            raise ValueError(f"claims overlap or are not in source order: span [{start}:{end}] overlaps previous end {last_end}")
        claim_id = f"{answer.answer_id}:{start}:{end}"
        if claim_id in seen_ids:
            raise ValueError(f"duplicate claim ID: {claim_id}")
        seen_ids.add(claim_id)
        result.append(ValidatedClaim(answer.answer_id, claim_id, start, end, item["quote"], item["label"], item["fact_ids"], item["reason"]))
        last_end = end

    return tuple(result)


class Agent:
    def __init__(self, provider: Provider, facts: list[BrandFact], version: str = "v1", timeout_seconds: float = 30):
        if version not in {"v1", "v2"}:
            raise ValueError("agent version must be v1 or v2")
        self.provider = provider
        self.tool = FactTool(facts)
        self.facts = facts
        self.version = version
        self.timeout_seconds = timeout_seconds

    def classify(self, answer: AnswerInput) -> AgentResult:
        base_prompt = build_prompt(answer, self.version)
        tools = [{"name": "get_brand_facts", "description": "Read-only catalogue lookup", "parameters": {"brand_id": "string"}}]
        tool_results: list[dict[str, Any]] = []
        turn = 0
        try:
            for turn in range(1, 4):
                if not tool_results:
                    current_prompt = base_prompt
                else:
                    current_prompt = (
                        f"{base_prompt}\n\n"
                        f"Catalogue facts returned from get_brand_facts:\n"
                        f"{json.dumps(tool_results, indent=2)}\n\n"
                        "Now, extract all complete factual sentences from the target answer and classify each against the catalogue facts above.\n"
                        "Output ONLY a valid JSON object matching the claims schema."
                    )
                response = self.provider.complete(current_prompt, tools, self.timeout_seconds)
                if not isinstance(response, dict):
                    raise ProviderError("invalid_output", "Provider returned a non-object response.")

                actions = response.get("actions", [])
                if not isinstance(actions, list):
                    raise ProviderError("invalid_output", "Provider actions must be a list.")

                if actions:
                    for action in actions:
                        if not isinstance(action, dict):
                            raise ProviderError("invalid_output", "Provider action must be an object.")
                        tool_name = action.get("tool", "")
                        args = action.get("arguments", {"brand_id": action.get("brand_id")})
                        res = self.tool.execute(tool_name, args)
                        tool_results.append({
                            "tool": tool_name,
                            "arguments": args,
                            "result": res,
                        })
                    continue

                raw_claims = response.get("claims")
                if not isinstance(raw_claims, list):
                    raise ProviderError("invalid_output", "Provider claims must be a list.")
                claims = validate_claims(answer, raw_claims, {f.fact_id for f in self.facts})
                return AgentResult(answer.answer_id, claims, turns=turn)

            raise ProviderError("turn_limit", "The agent exceeded the maximum of three turns.")
        except ProviderError as exc:
            return AgentResult(answer.answer_id, error_type=exc.error_type, error_message=str(exc), turns=turn)
        except TimeoutError as exc:
            return AgentResult(answer.answer_id, error_type="timeout", error_message=str(exc), turns=turn)
        except (ValueError, TypeError, KeyError) as exc:
            return AgentResult(answer.answer_id, error_type="invalid_output", error_message=str(exc), turns=turn)
