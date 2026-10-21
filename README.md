# Kasparro FSE Intern Pathway Assessment: AI Claim Extraction & Classification

**Repository Link**: [https://github.com/charancreates/ai-claim-classifier](https://github.com/charancreates/ai-claim-classifier)

Automated pipeline to extract factual product claim sentences from long, unstructured AI answers, validate character offsets against original text, and classify each claim against an authoritative product catalogue (`supported`, `contradicted`, or `unknown`).

---

## 1. Project Structure

```text
├── data/                       # CSV datasets
│   ├── answers.csv             # 50 answers (A01-A40 dev, A41-A50 check)
│   ├── brand_facts.csv         # 132 catalogue facts (6 models across 3 brands)
│   └── development_gold.csv    # 376 reference gold claims for dev split
├── reports/                    # Live artifacts and test evidence
│   ├── v1_development.json     # Baseline v1 run on development split (A01-A40)
│   ├── v2_development.json     # Candidate v2 run on development split (A01-A40)
│   ├── v1_check.json           # Baseline v1 run on check split (A41-A50)
│   ├── v2_check.json           # Candidate v2 run on check split (A41-A50)
│   ├── sanitized_exchange.json # Sanitized prompt & tool-calling exchange
│   └── test_output.txt         # Real executed pytest output (17/17 passed)
├── src/kasparro/
│   ├── agent.py                # Agent loop, turn limit (<=3), span & quote validation
│   ├── api.py                  # Thin FastAPI endpoints (POST /classify, POST/GET /evaluation-runs)
│   ├── database.py             # PostgreSQL SQLAlchemy schema (runs, answers, claims)
│   ├── evaluator.py            # Deterministic exact-span extraction & joint scorer
│   ├── main.py                 # FastAPI application factory
│   ├── models.py               # Strict domain contracts & dataclasses
│   ├── parsing.py              # UTF-8 CSV parsing with strict typed values
│   ├── prompts.py              # Frozen V1 baseline & single targeted V2 prompt
│   └── provider.py             # LLM provider adapter with function calling
├── tests/
│   └── test_core.py            # Comprehensive unit and integration test suite
├── .env.example                # Secret-free environment template
├── pyproject.toml              # Project metadata & build configuration
├── requirements.txt            # Pinned dependency versions
└── run_evaluation.py           # CLI evaluation runner
```

---

## 2. Dataset Verification & SHA-256 Hashes

The dataset is loaded with strict UTF-8 decoding, preserving original whitespace, quotes, and punctuation without normalization.

| File | Records | SHA-256 Hash |
| :--- | :--- | :--- |
| `data/answers.csv` | 50 records (A01–A50) | `fa28e813ea54b4e5ea985585f22f084d826241e1182accbc27ace17066125724` |
| `data/brand_facts.csv` | 132 facts (6 models) | `2190d40fad2dd167ef2437956c43d595aaf2554ea673a4789b9343cb7e2936f2` |
| `data/development_gold.csv` | 376 gold claims | `afe84cc78f304afb6a4aaa505dd7504af80fcbf8bc507806aa5ff0aab6bf4850` |

---

## 3. Environment & Database Setup

### Prerequisites
* Python 3.11+
* PostgreSQL 15+

### Installation
```bash
git clone https://github.com/charancreates/ai-claim-classifier.git
cd ai-claim-classifier

# Create virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

### Secret-Free Environment Configuration
Copy the template to `.env` (note: `.env` is ignored by `.gitignore` and must never be committed):
```bash
cp .env.example .env
```

Configure `.env`:
```env
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/kasparro
LLM_PROVIDER=gemini
LLM_MODEL=gemini-flash-latest
LLM_API_KEY=your_gemini_api_key_here
PROVIDER_TIMEOUT_SECONDS=30
DATA_DIR=data
```

### PostgreSQL Setup
Create the database:
```sql
CREATE DATABASE kasparro;
```
Tables (`evaluation_runs`, `answer_outcomes`, `claim_outcomes`) and relational constraints with cascading deletes are automatically initialized by SQLAlchemy on startup.

* **Provider Timeout**: Fixed at **30 seconds** per provider call (configured via `PROVIDER_TIMEOUT_SECONDS`).

---

## 4. Execution & Testing Commands

### A. Run Controlled Tests
Execute the full test suite and verify parser, scorer arithmetic, provider error handling, and API persistence:
```bash
python -m pytest -v
```
*(The actual verified output of all 17 passing tests is saved in `reports/test_output.txt`)*.

### B. Run the FastAPI Web Service
```bash
uvicorn kasparro.main:app --reload --port 8000
```
#### API Endpoints:
* `POST /classify`: Classifies a single input answer. Rejects invalid input or forbidden fields (`gold`) with HTTP 422; unknown brand with HTTP 404; provider error with HTTP 502; timeout with HTTP 504.
* `POST /evaluation-runs`: Accepts `{"agent_version": "v1"|"v2", "split": "development"|"check"}`. Returns HTTP 201 Created and saves runs and predictions into PostgreSQL.
* `GET /evaluation-runs/{run_id}`: Returns the complete joined run, answer outcomes, and predicted claims in stable source order. Returns HTTP 404 for missing runs and HTTP 422 for invalid UUIDs.

### C. Run the Evaluation Pipeline
```bash
# Run v1 on development
python run_evaluation.py --version v1 --split development

# Run v2 on development
python run_evaluation.py --version v2 --split development

# Run v1 on check
python run_evaluation.py --version v1 --split check

# Run v2 on check
python run_evaluation.py --version v2 --split check
```

---

## 5. Controlled Test Suite Coverage

All 17 tests passed in 1.09s (recorded in `reports/test_output.txt`):
1. **CSV & Encoding**: Quoted commas and embedded newlines preserved exactly (`test_csv_parser_embedded_newlines_and_quoted_commas`).
2. **Type Parsing**: Strict boolean (`true`/`false`, rejects `bool("false")`) and numeric conversions (`test_strict_boolean_and_number_parsing`).
3. **Offset Validation**: Unicode offsets located in `ai_answer`; quote equality verified; non-overlapping ordered spans; unique IDs (`test_exact_quote_and_unicode_offsets`, `test_invalid_quote_not_in_answer_is_rejected`, `test_overlapping_spans_are_rejected`, `test_duplicate_claims_are_rejected`).
4. **Scorer Arithmetic**: Verified with specification example ($G=10, P=10, M=8, C=7 \to \text{Extraction } F_1 = 0.80, \text{Joint } F_1 = 0.70$), perfect scores, provider errors, and empty predictions (`test_scorer_spec_arithmetic_example`, `test_scorer_perfect_output`, `test_scorer_provider_error_answer_retains_gold_as_missed`, `test_scorer_valid_empty_predictions`).
5. **Private Benchmark Integrity**: Check split quality metrics remain `null` (`test_scorer_check_split_has_null_quality`).
6. **Provider & Tool Fakes**: Malformed JSON handling, turn exhaustion hard limit (3 turns), illegal tool rejection, prompt leakage prevention (`test_fake_provider_malformed_output`, `test_fake_provider_turn_exhaustion`, `test_gold_never_leaks_to_model_or_tools`).
7. **FastAPI & PostgreSQL**: Validation error handling, 404 unknown brands, 422 extra fields, database persistence and joined retrieval (`test_api_classify_and_validation`, `test_api_database_persistence_and_retrieval`).

---

## 6. Baseline (V1) vs Candidate (V2): Targeted Improvement & Results

### Failure Diagnosis (V1)
In the baseline V1 run on the 40 development answers, the model frequently generated conversational explanations exceeding two sentences. Under the domain contract (`reason must contain at most two sentences`), these caused complete answer-level validation failures (`error_type: "invalid_output"`):
* Answers **A05**, **A06**, **A34**, and **A35** failed validation entirely due to verbose reasons.
* Because an answer error awards 0 accepted predictions while all its reference claims remain in $G$ as missed, this severely penalized both Extraction Recall and Joint Recall.

### One Targeted Improvement (V2)
To preserve identical tool contracts, evaluator rules, and model settings, V2 modified only the prompt template by appending an explicit reason brevity instruction:
```python
V2_PROMPT_TEMPLATE = V1_PROMPT_TEMPLATE.replace(
    "Output format:",
    "REASON LENGTH RULE: Every claim's reason must be at most TWO short sentences. "
    "Never write three or more. Prefer one short sentence. Count your sentences "
    "before answering.\n\nOutput format:",
)
```

### Measured Comparison on Development Split (A01–A40, 376 Gold Claims)

| Metric | Baseline V1 | Candidate V2 | Delta |
| :--- | :---: | :---: | :---: |
| **Accepted Predictions ($P$)** | 327 | 333 | +6 |
| **Exact Span Matches ($M$)** | 325 | 331 | +6 |
| **Correctly Labeled Matches ($C$)** | 322 | 326 | +4 |
| **Extraction Precision** | 0.9939 | 0.9940 | +0.0001 |
| **Extraction Recall** | 0.8644 | 0.8803 | **+1.59%** |
| **Extraction $F_1$** | 0.9246 | **0.9337** | **+0.0091** |
| **Joint Precision** | 0.9847 | 0.9790 | -0.0057 |
| **Joint Recall** | 0.8564 | 0.8670 | **+1.06%** |
| **Joint $F_1$ (Primary Quality Metric)** | 0.9161 | **0.9196** | **+0.0035** |
| **Gold Coverage ($C / G$)** | 85.64% | **86.70%** | **+1.06%** |
| **Answer Errors** | 5 | 0 | **-5 (Resolved)** |

### Analysis & Regressions
* **Gain**: The targeted brevity rule prevented validation crashes on multi-sentence answers, allowing previously dropped valid claims to be extracted and classified, lifting Joint Recall from 85.64% to 86.70%.
* **Regression / Tradeoff**: Joint Precision dropped slightly from 98.47% to 97.90% due to 2 additional edge-case claims where shorter explanations occasionally omitted contextual caveats, but the overall Joint $F_1$ improved from 0.9161 to 0.9196.

---

## 7. Submission Artifacts

All four run reports are saved in [`reports/`](file:///c:/Users/charan/Desktop/kasparro%20assessment/reports):
1. `reports/v1_development.json`: 40 answers, predictions, missed gold IDs, wrong labels, and scores.
2. `reports/v2_development.json`: 40 answers, candidate predictions, and improved scores.
3. `reports/v1_check.json`: 10 final-check answers (A41–A50) with predictions; quality metrics are `null`.
4. `reports/v2_check.json`: 10 final-check answers (A41–A50) with predictions; quality metrics are `null`.
5. `reports/sanitized_exchange.json`: Full model/tool conversation transcript for answer A01, verified free of API keys or secrets.
6. `reports/test_output.txt`: Actual pytest execution console log.

---

## 8. Time Spent, Limitations & AI Assistance Disclosure

* **Time Spent**: ~2.5 to 3 hours total building the agent, parser, database schema, evaluation harness, API endpoints, tests, and running artifacts.
* **Limitations & Unfinished Work**:
  * Agent execution is currently sequential; parallel batch processing with rate-limiting queues could accelerate runs.
  * Sentence boundary tokenization relies on punctuation heuristics; integration of spaCy or a formal NLP sentence segmenter would provide even tighter boundary precision on rare abbreviations.
* **AI Assistance Disclosure**:
  * **Claude Code**: Used for initial architectural decomposition, requirement review, and rubric planning.
  * **Google Gemini / Antigravity**: Used for code implementation, test scaffolding, and debugging.
  * **Manual Human Verification**: All Unicode offset calculations, exact-span matcher logic, arithmetic metrics ($G, P, M, C$), database uniqueness constraints, and live-run artifact verification were independently checked.
