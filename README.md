# Kasparro FSE Intern Pathway Assessment

## Structure

```text
src/kasparro/
  agent.py       controlled provider/tool loop and span validation
  prompts.py     frozen v1 baseline and v2 context-ordering prompts
  parsing.py     UTF-8 CSV loading and strict typed fact parsing
  evaluator.py   exact-span deterministic extraction/joint scorer
  models.py      domain contracts
  database.py    PostgreSQL SQLAlchemy persistence models
  api.py         thin FastAPI endpoints
  main.py        configured application entrypoint
tests/
  test_core.py   controlled parser, span, and arithmetic checks
```

## Setup

Use Python 3.11 or newer:

```powershell
py -m pip install -e ".[test]"
Copy-Item .env.example .env
py -m pytest
```

Set `DATABASE_URL` before starting the API. The application intentionally
does not contain a provider key. A provider adapter must implement
`Provider.complete(prompt, tools, timeout_seconds)` and return an object with
`actions` and `claims`.

```powershell
py -m uvicorn kasparro.main:app --reload
```

The agent never accepts model-supplied offsets. It locates each exact quote in
the original decoded answer using Python Unicode code-point indexing, then
validates bounds, order, overlap, labels, and fact IDs. Tool names and brand
IDs are checked by `FactTool`; failures remain answer-level errors.

The evaluator uses `M/P`, `M/G`, and `2M/(P+G)` for extraction and
`C/P`, `C/G`, and `2C/(P+G)` for joint quality. Check runs return null quality
metrics because their labels are private.
