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
│   └── test_core.py            # Controlled parser, span, scorer & failure tests
├── .env.example                # Secret-free environment template
├── pyproject.toml              # Project metadata & build configuration
├── requirements.txt            # Pinned dependency versions
└── run_evaluation.py           # CLI evaluation runner
```

---

## 2. Design Decisions

* **Python locates offsets**: The agent locates quotes in the original `ai_answer` via zero-based Python Unicode code points, so the LLM never counts characters.
* **Preserve error fidelity**: Any invalid output, unknown tool, or malformed quote becomes an answer-level error; it is never silently masked as "unknown".
* **Strict tool validation**: Only `get_brand_facts` is callable, and only for existing brands in the catalogue.
* **Zero data leakage**: Gold spans, labels, and reference IDs are strictly isolated to the evaluation harness; they never enter model prompts or tool calls.
* **Constrained execution**: Enforces a strict maximum of 3 model turns and a configurable per-request provider timeout (30 seconds).
* **Targeted prompt separation**: V1 and V2 are distinct, frozen prompt templates; V2 is strictly V1 plus one targeted brevity rule.

---

## 3. Dataset Verification & Hashes

The dataset is loaded with strict UTF-8 decoding, preserving original whitespace, quotes, and punctuation without normalization.

| File | Records | SHA-256 Hash |
| :--- | :--- | :--- |
| `data/answers.csv` | 50 records (A01–A50) | `fa28e813ea54b4e5ea985585f22f084d826241e1182accbc27ace17066125724` |
| `data/brand_facts.csv` | 132 facts (6 models) | `2190d40fad2dd167ef2437956c43d595aaf2554ea673a4789b9343cb7e2936f2` |
| `data/development_gold.csv` | 376 gold claims | `afe84cc78f304afb6a4aaa505dd7504af80fcbf8bc507806aa5ff0aab6bf4850` |

---

## 4. Setup & Environment

### Prerequisites
* Python 3.11+
* PostgreSQL 15+

### Installation
```bash
git clone https://github.com/charancreates/ai-claim-classifier.git
cd ai-claim-classifier

# Create virtual environment
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

### Environment Configuration
Copy `.env.example` to `.env` (note: `.env` is ignored by `.gitignore` and never committed):
```bash
cp .env.example .env
```

Set the database connection string and provider configuration in `.env`:
```env
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/kasparro
LLM_PROVIDER=gemini
LLM_MODEL=gemini-flash-latest
LLM_API_KEY=your_gemini_api_key_here
PROVIDER_TIMEOUT_SECONDS=30
DATA_DIR=data
```

### PostgreSQL Setup
```sql
CREATE DATABASE kasparro;
```
Tables (`evaluation_runs`, `answer_outcomes`, `claim_outcomes`) and relational foreign keys with cascading deletes are automatically initialized on application startup.

---

## 5. Execution & API Examples

### A. Run Controlled Tests
```bash
python -m pytest -v
```
Verified actual test output (17/17 passed in 1.09s) is committed in [`reports/test_output.txt`](file:///c:/Users/charan/Desktop/kasparro%20assessment/reports/test_output.txt).

### B. Start the FastAPI Service
```bash
uvicorn kasparro.main:app --reload --port 8000
```

### C. Example 1: `POST /classify` Request
```bash
curl -X POST "http://127.0.0.1:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "answer_id": "A01",
    "brand_id": "cleartrail",
    "question": "As of 1 October 2026, consider ClearTrail Mini for weekend packing...",
    "ai_answer": "For the current India catalogue, ClearTrail Mini is listed at INR 1,200 less than ClearTrail Plus, comparing the two tax-inclusive prices before shipping. For an authorized India purchase with the original invoice and a manufacturing defect, ClearTrail Mini receives a standard manufacturer warranty lasting 18 months from the purchase date.",
    "agent_version": "v2"
  }'
```

### D. Example 2: Real Extracted Claims Output (from Answer A01)
```json
{
  "answer_id": "A01",
  "claims": [
    {
      "claim_id": "A01:306:460",
      "start_char": 306,
      "end_char": 460,
      "quote": "For the current India catalogue, ClearTrail Mini is listed at INR 1,200 less than ClearTrail Plus, comparing the two tax-inclusive prices before shipping.",
      "label": "supported",
      "fact_ids": ["cleartrail.mini.price_2026", "cleartrail.plus.price_2026"],
      "reason": "ClearTrail Mini list price is INR 1,299 and ClearTrail Plus is INR 2,499, making Mini exactly INR 1,200 less."
    },
    {
      "claim_id": "A01:461:647",
      "start_char": 461,
      "end_char": 647,
      "quote": "For an authorized India purchase with the original invoice and a manufacturing defect, ClearTrail Mini receives a standard manufacturer warranty lasting 18 months from the purchase date.",
      "label": "contradicted",
      "fact_ids": ["cleartrail.mini.warranty_india"],
      "reason": "The standard manufacturer warranty for ClearTrail Mini in India is 12 months, not 18 months."
    }
  ]
}
```

### E. Run the CLI Evaluation
```bash
python run_evaluation.py --version v1 --split development
python run_evaluation.py --version v2 --split development
python run_evaluation.py --version v1 --split check
python run_evaluation.py --version v2 --split check
```

---

## 6. Baseline (V1) vs Candidate (V2): Targeted Improvement & Results

### Failure Diagnosis (V1)
In baseline V1 on the 40 development answers, the LLM frequently produced explanations exceeding two sentences. Under the domain contract (`reason must contain at most two sentences`), these triggered `invalid_output` errors, dropping answers A05, A06, A34, and A35 entirely. Because an error results in 0 predictions while reference claims stay in $G$ as missed, this hurt recall significantly.

### One Targeted Improvement (V2)
V2 retained the exact tool and evaluator rules, modifying only the prompt template to append an explicit reason brevity instruction:
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

---

## 7. Limitations and TODOs

### Not Done Yet (Planned Test Extensions)
* Fake provider tests for an unsupported label, an illegal tool name, an unknown requested brand, and a timeout.
* Separate scorer unit tests for isolated missed claims, extra advice claims, and wrong labels.
* Scorer edge cases explicitly rejecting missing or duplicated answer outcomes.
* API integration tests specifically for HTTP 404, 502, 504 status code assertions, and `GET /evaluation-runs/{id}` with missing IDs or malformed UUIDs.

### Known Limits
* **Sequential Execution**: The runner executes sequentially; a full run across 50 answers takes several minutes.
* **Single Model / Provider**: Configured for Gemini without automated provider fallback if unavailable.
* **Private Check Labels**: Check-split quality is not scored locally; quality metrics remain `null` as required so the reviewer can score with private labels.
* **Zero Final-Check Tuning**: Prompts and rules were evaluated strictly on A01–A40; no tuning was performed on A41–A50.

### Next Improvements
* **Soft Retry on Output Failure**: Implement a single in-turn retry when model output formatting fails, before dropping the whole answer.
* **Concurrent Execution**: Add an asynchronous or thread-pool worker queue for batch evaluation.
* **Refined Clause Disambiguation**: Review the remaining 3 wrong-label and 2 extra claims in the development report to further isolate compound arithmetic statements.

---

## 8. AI Assistance Disclosure & Time Spent

* **Time Spent**: ~2.5 to 3 hours total building the agent, parser, database schema, evaluation harness, API endpoints, tests, and running artifacts.
* **AI Assistance Disclosure**:
  * **Claude**: Used for initial brainstorming, requirement analysis, and rubric alignment.
  * **Google Gemini / Antigravity**: Assisted in coding, test generation, and pair-programming.
