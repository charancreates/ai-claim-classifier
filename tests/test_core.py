import io
import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from kasparro.agent import Agent, FactTool, ProviderError, validate_claims
from kasparro.api import create_app
from kasparro.database import Base, EvaluationRun, AnswerOutcome, ClaimOutcome
from kasparro.evaluator import GoldClaim, score_results
from kasparro.models import AgentResult, AnswerInput, BrandFact, ValidatedClaim
from kasparro.parsing import load_answers, load_facts, read_csv_rows


# ---------------------------------------------------------
# 1. CSV Parsing & Exact Text Preservation Checks
# ---------------------------------------------------------

def test_csv_parser_embedded_newlines_and_quoted_commas(tmp_path):
    csv_file = tmp_path / "answers.csv"
    content = (
        'answer_id,split,brand_id,question,ai_answer\n'
        'A01,development,cleartrail,"Question, with comma","Line 1 with, comma.\n\nLine 2 after newline."\n'
    )
    csv_file.write_text(content, encoding="utf-8", newline="")
    answers = load_answers(csv_file)
    assert len(answers) == 1
    assert answers[0].answer_id == "A01"
    assert answers[0].brand_id == "cleartrail"
    assert answers[0].question == "Question, with comma"
    assert answers[0].ai_answer == "Line 1 with, comma.\n\nLine 2 after newline."


def test_strict_boolean_and_number_parsing(tmp_path):
    csv_file = tmp_path / "facts.csv"
    content = (
        "fact_id,brand_id,model_id,field,value,value_type\n"
        "f1,b1,m1,active,true,boolean\n"
        "f2,b1,m1,flag,false,boolean\n"
        "f3,b1,m1,count,42,number\n"
        "f4,b1,m1,ratio,3.14,number\n"
        "f5,b1,m1,name,test,string\n"
    )
    csv_file.write_text(content, encoding="utf-8")
    facts = load_facts(csv_file)
    assert facts[0].value is True
    assert facts[1].value is False
    assert facts[2].value == 42
    assert facts[3].value == 3.14
    assert facts[4].value == "test"

    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("fact_id,brand_id,model_id,field,value,value_type\nf,b,m,k,yes,boolean\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid boolean value"):
        load_facts(bad_csv)


# ---------------------------------------------------------
# 2. Span Validation, Quote Equality & Offset Checks
# ---------------------------------------------------------

def test_exact_quote_and_unicode_offsets():
    answer = AnswerInput("A01", "b1", "q", "Intro. Café is supported.")
    claims = validate_claims(
        answer,
        [{"quote": "Café is supported.", "label": "supported", "fact_ids": [], "reason": "Verified."}],
        set(),
    )
    assert claims[0].start_char == 7
    assert claims[0].end_char == len(answer.ai_answer)
    assert answer.ai_answer[claims[0].start_char:claims[0].end_char] == claims[0].quote
    assert claims[0].claim_id == f"A01:7:{len(answer.ai_answer)}"


def test_invalid_quote_not_in_answer_is_rejected():
    answer = AnswerInput("A01", "b1", "q", "Actual answer text.")
    with pytest.raises(ValueError, match="quote does not occur"):
        validate_claims(answer, [{"quote": "Nonexistent claim.", "label": "supported"}], set())


def test_overlapping_spans_are_rejected():
    answer = AnswerInput("A01", "b1", "q", "Sentence one. Sentence two.")
    with pytest.raises(ValueError, match="overlap"):
        validate_claims(
            answer,
            [
                {"quote": "Sentence one. Sentence two.", "label": "supported"},
                {"quote": "Sentence two.", "label": "supported"},
            ],
            set(),
        )


def test_duplicate_claims_are_rejected():
    answer = AnswerInput("A01", "b1", "q", "Only one sentence.")
    with pytest.raises(ValueError):
        validate_claims(
            answer,
            [
                {"quote": "Only one sentence.", "label": "supported"},
                {"quote": "Only one sentence.", "label": "supported"},
            ],
            set(),
        )


def test_invalid_label_and_invented_fact_ids_are_rejected():
    answer = AnswerInput("A01", "b1", "q", "Sentence one.")
    with pytest.raises(ValueError, match="invalid label"):
        validate_claims(answer, [{"quote": "Sentence one.", "label": "maybe"}], set())

    with pytest.raises(ValueError, match="invalid fact ID"):
        validate_claims(answer, [{"quote": "Sentence one.", "label": "supported", "fact_ids": ["invented.id"]}], {"valid.id"})


# ---------------------------------------------------------
# 3. Scorer Arithmetic Checks (Spec Mandatory Arithmetic)
# ---------------------------------------------------------

def test_scorer_spec_arithmetic_example():
    # Spec: G=10, P=10, M=8, C=7 -> Extraction F1=0.8, Joint F1=0.7
    gold = [GoldClaim("A", str(i), i, i + 1, "x", "supported") for i in range(10)]
    claims = tuple(
        ValidatedClaim("A", f"A:{i}:{i+1}", i, i + 1, "x", "unknown" if i == 7 else "supported", (), "")
        for i in range(8)
    ) + tuple(
        ValidatedClaim("A", f"A:{i}:{i+1}", i, i + 1, "x", "unknown", (), "")
        for i in range(100, 102)
    )
    result = score_results([AgentResult("A", claims)], gold, "development")
    assert result.extraction.f1 == pytest.approx(0.8)
    assert result.extraction.precision == pytest.approx(0.8)
    assert result.extraction.recall == pytest.approx(0.8)
    assert result.joint.f1 == pytest.approx(0.7)
    assert result.joint.precision == pytest.approx(0.7)
    assert result.joint.recall == pytest.approx(0.7)
    assert result.gold_coverage == pytest.approx(0.7)


def test_scorer_perfect_output():
    gold = [GoldClaim("A", f"A:{i}:{i+1}", i, i + 1, "s", "supported") for i in range(5)]
    claims = tuple(
        ValidatedClaim("A", f"A:{i}:{i+1}", i, i + 1, "s", "supported", (), "")
        for i in range(5)
    )
    res = score_results([AgentResult("A", claims)], gold, "development")
    assert res.extraction.f1 == pytest.approx(1.0)
    assert res.joint.f1 == pytest.approx(1.0)
    assert res.missed_gold_ids == []
    assert res.unmatched_prediction_ids == []
    assert res.wrong_label_ids == []


def test_scorer_provider_error_answer_retains_gold_as_missed():
    gold = [GoldClaim("A", "c1", 0, 10, "text", "supported")]
    error_result = AgentResult("A", (), error_type="timeout", error_message="deadline")
    res = score_results([error_result], gold, "development")
    assert res.extraction.precision == 0.0
    assert res.extraction.recall == 0.0
    assert res.extraction.f1 == 0.0
    assert res.joint.f1 == 0.0
    assert res.missed_gold_ids == ["c1"]
    assert res.answer_errors == 1


def test_scorer_valid_empty_predictions():
    gold = [GoldClaim("A", "c1", 0, 10, "text", "supported")]
    empty_result = AgentResult("A", ())
    res = score_results([empty_result], gold, "development")
    assert res.extraction.f1 == 0.0
    assert res.joint.f1 == 0.0
    assert res.missed_gold_ids == ["c1"]


def test_scorer_check_split_has_null_quality():
    res = score_results(
        [AgentResult("A", ())],
        [GoldClaim("A", "c", 0, 5, "x", "supported")],
        "check",
    )
    assert res.extraction is None
    assert res.joint is None
    assert res.gold_coverage is None


# ---------------------------------------------------------
# 4. Fake Provider Failure Mode Checks
# ---------------------------------------------------------

class MalformedJsonProvider:
    def complete(self, prompt, tools, timeout_seconds):
        return "not a valid json object"  # type: ignore


class TurnExhaustionProvider:
    def complete(self, prompt, tools, timeout_seconds):
        return {"actions": [{"tool": "get_brand_facts", "arguments": {"brand_id": "b1"}}]}


class LeakCheckingProvider:
    def __init__(self):
        self.received_prompts = []

    def complete(self, prompt, tools, timeout_seconds):
        self.received_prompts.append(prompt)
        return {"claims": []}


def test_fake_provider_malformed_output():
    facts = [BrandFact("f1", "b1", "m1", "f", "v", "string")]
    agent = Agent(MalformedJsonProvider(), facts, version="v1")
    res = agent.classify(AnswerInput("A", "b1", "q", "Text."))
    assert res.error_type == "invalid_output"


def test_fake_provider_turn_exhaustion():
    facts = [BrandFact("f1", "b1", "m1", "f", "v", "string")]
    agent = Agent(TurnExhaustionProvider(), facts, version="v1")
    res = agent.classify(AnswerInput("A", "b1", "q", "Text."))
    assert res.error_type == "turn_limit"
    assert res.turns == 3


def test_gold_never_leaks_to_model_or_tools():
    facts = [BrandFact("f1", "b1", "m1", "f", "v", "string")]
    provider = LeakCheckingProvider()
    agent = Agent(provider, facts, version="v1")
    agent.classify(AnswerInput("A01", "b1", "q", "Factual answer."))
    assert len(provider.received_prompts) > 0
    for p in provider.received_prompts:
        assert "expected_label" not in p
        assert "difficulty_tags" not in p
        assert "gold" not in p.lower()


# ---------------------------------------------------------
# 5. FastAPI & Database Persistence Checks
# ---------------------------------------------------------

def test_api_classify_and_validation():
    facts = [BrandFact("f1", "cleartrail", "mini", "field", "val", "string")]
    
    class DummyProvider:
        def complete(self, prompt, tools, timeout):
            return {"claims": []}

    app = create_app(Agent(DummyProvider(), facts, version="v1"))
    client = TestClient(app)

    # Unknown brand -> 404
    resp = client.post(
        "/classify",
        json={"answer_id": "A1", "brand_id": "unknown_brand", "question": "q", "ai_answer": "ans", "agent_version": "v1"},
    )
    assert resp.status_code == 404

    # Extra fields (e.g. gold) -> 422
    resp = client.post(
        "/classify",
        json={"answer_id": "A1", "brand_id": "cleartrail", "question": "q", "ai_answer": "ans", "agent_version": "v1", "gold": "forbidden"},
    )
    assert resp.status_code == 422

    # Success -> 200
    resp = client.post(
        "/classify",
        json={"answer_id": "A1", "brand_id": "cleartrail", "question": "q", "ai_answer": "ans", "agent_version": "v1"},
    )
    assert resp.status_code == 200
    assert resp.json()["answer_id"] == "A1"


def test_api_database_persistence_and_retrieval(tmp_path):
    db_file = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_file}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(engine, expire_on_commit=False)

    facts = [BrandFact("f1", "cleartrail", "mini", "price", 1299, "number")]

    class MockAppProvider:
        def complete(self, prompt, tools, timeout):
            return {"claims": []}

    app = create_app(Agent(MockAppProvider(), facts, version="v1"), session_factory=session_factory)
    client = TestClient(app)

    # Invalid UUID -> 422
    assert client.get("/evaluation-runs/not-a-uuid").status_code == 422

    # Missing UUID -> 404
    assert client.get(f"/evaluation-runs/{uuid.uuid4()}").status_code == 404

    # Run check split with mock provider
    post_resp = client.post("/evaluation-runs", json={"agent_version": "v1", "split": "check"})
    assert post_resp.status_code == 201
    run_id = post_resp.json()["run_id"]
    assert post_resp.json()["status"] == "completed"
    assert post_resp.json()["metrics"] is None

    # Retrieve run and verify joined retrieval
    get_resp = client.get(f"/evaluation-runs/{run_id}")
    assert get_resp.status_code == 200
    data = get_resp.json()
    assert data["run_id"] == run_id
    assert data["split"] == "check"
    assert len(data["answers"]) == 10  # A41 to A50
    # Verify answers are sorted by answer_id
    answer_ids = [a["answer_id"] for a in data["answers"]]
    assert answer_ids == sorted(answer_ids)
