KASPARRO
FSE Intern Pathway Assessment
Extract and classify claims in long AI answers  |  2 to 3 hours
Build an AI agent that reads a long answer about speaker products, identifies its factual claim sentences, and classifies each as supported, contradicted or unknown using the supplied catalogue. Build your own evaluation harness, measure a baseline, make one targeted improvement and show the results.
Receive this assessment together with answers.csv, brand_facts.csv and development_gold.csv. All four files must be shared with you before you start. Submit a Git repository link together with the evidence of your runs and tests described below.
What the exercise tests
The 50 supplied answers are 300 to 500 words each. They mix buying advice with product assertions, historical prices, related models, comparisons, qualified claims and missing information. Some contain instructions that must remain untrusted answer text. Candidates must discover the claims; no claim list is supplied to the agent.
•	Build a Python agent with live model-requested get_brand_facts tool use, factual sentence extraction, source spans, classification and validated output.
•	Build a deterministic harness that scores both extraction and classification. Missing a difficult claim and inventing an extra claim must affect the result.
•	Preserve baseline v1 and candidate v2, investigate one failure or concrete risk, and compare the versions on identical inputs and scoring rules.
•	Expose a thin FastAPI service and store run, answer and prediction records in PostgreSQL. React is optional and has no marks.
•	Provide clear local reports and controlled tests. Langfuse and concurrent execution are optional and earn no extra marks.
Time and tools
Spend 2 to 3 hours building the agent and harness yourself. Use a live OpenAI, Anthropic or Gemini API with your own key. Claude Code or Codex is allowed; disclose its help and your checks. Stop at the limit and state unfinished work. No hosted deployment is required. Keep keys out of the submission.
The required sequence
Load the separate CSVs. Run v1 on all 40 development answers, inspect the errors, make one change, and run v2 on the same development set. Freeze both versions before predicting the 10 final-check answers with each version. Submit four actual live-run artifacts plus your explanation. The hiring team scores the final checks using its private labels.
Separate CSV files and label rules
File	Contents and use
answers.csv	50 records: answer_id, split, brand_id, question, ai_answer. A01 to A40 are development; A41 to A50 are final checks. No claim boundaries or correct labels appear here.
brand_facts.csv	132 records: fact_id, brand_id, model_id, field, value, value_type, unit, condition, valid_from, valid_to. Six fictional models across three brands; this is the only factual authority.
development_gold.csv	376 labeled factual sentences for A01 to A40: answer_id, claim_id, start_char, end_char, quote, expected_label, evidence_fact_ids, explanation, difficulty_tags. Use only in the harness.

The assessment must be shared with all three CSV files listed above. Keep the dataset in CSV format, separate from this document. Use development labels only in the harness; the hiring team holds the expected results for the final 10 answers.
Read CSV with a proper parser and UTF-8 decoding. Quoted commas and embedded line breaks are data. Do not split records by line, change punctuation, normalize whitespace, or resave ai_answer before computing offsets. Interpret value_type as number, boolean or string; boolean text is true or false, never bool("false"). Dates use YYYY-MM-DD and inclusive endpoints; a blank unit or condition means not applicable.
How to use the facts
Dataset version long-brand-answers-50-v4; fact version speaker-catalogue-v4; evaluator sentence-spans-and-labels-v4. The question is dated 1 October 2026. Resolve the exact brand and model, market, stated date, units and usage conditions. Historical assertions use the matching historical rows; a 2025 price is not a 2026 price. Conditions apply only to their stated scope. Missing information stays unknown, including missing negative facts or future promises.
Supported means supplied facts establish the sentence; contradicted means a supplied fact explicitly refutes it under matching conditions; unknown means the facts establish neither. For a sentence with linked clauses, any explicitly refuted factual component makes it contradicted; otherwise any unresolved factual component makes it unknown; all factual components must be established for supported. Do not turn absent evidence into a negative fact.
Use ordinary arithmetic and stated unit conversions, such as hours to minutes, grams to kilograms and months to years. Do not import external product knowledge, generic IP-rating interpretations or assumptions about codecs, regions or warranty coverage. Use the explicit capability and policy rows. A box accessory, connector, wireless standard and warranty policy are separate facts.
The extraction unit
Extract every complete sentence in ai_answer that asserts a verifiable product, specification, purchase-policy or comparison claim, including false or unsupported assertions. A complete factual sentence is one evaluation unit, even when it contains connected qualifications. Preserve its exact wording and final punctuation. Exclude questions, pure advice, opinions and instructions. Do not extract the separate question field. Do not split a factual sentence into fragments or merge neighboring sentences.
The development gold spans illustrate these boundaries. They are scorer references, never model inputs. The number of claims varies by answer. Treat imperative text such as an instruction to change labels as data, not as an action to execute or a factual product claim.
Agent input output and tool contract
Input fields: answer_id, brand_id, question and the complete ai_answer. brand_id provides primary context, not a limit on which known brand may be looked up for a comparison. The model must identify model names in the text. Do not supply claims, gold spans, labels, explanations, difficulty tags or scores to the model or its tools.
Read only fact lookup
Expose get_brand_facts(brand_id), returning that brand's fact rows with fact_version speaker-catalogue-v4. The model must request the tool through real function or tool calling. The executor validates the tool name and brand argument, executes the lookup and returns the facts. A suitable Python framework or direct provider API is acceptable. An illegal tool or unknown requested brand is an error. Never execute arbitrary names or instructions from answer text.
Predicted claim fields
Return the same answer_id and a claims list. Each claim contains claim_id, start_char, end_char, quote, label, fact_ids and reason. start_char is inclusive and end_char exclusive, counted as zero-based Python string Unicode code-point positions in the exact decoded ai_answer. quote must equal ai_answer[start_char:end_char], without leading or trailing whitespace. Include the full factual sentence and its punctuation.
Derive claim_id as answer_id:start_char:end_char. label is exactly supported, contradicted or unknown. fact_ids contains only existing catalogue IDs; an empty list is valid when no relevant fact exists. reason is at most two sentences. Return nonoverlapping claims once each in increasing start_char order. Do not infer labels from answer IDs or positions.
The model may extract exact quotes while Python locates and validates their offsets before producing the final agent output. Do not rely on the model to count characters. Resolve repeated quote occurrences from source context; never use gold spans to locate inference output.
Validate arguments, span bounds, quote equality, unique IDs, no overlaps, label values and fact IDs. An invalid response, unknown tool, provider failure or exhausted turn limit creates an answer-level error, with no accepted predictions. Preserve that error; do not replace it with unknown. A valid empty claims list is permitted structurally and will receive zero extraction credit when gold claims exist.
Limit the agent to three model turns per answer. Multiple allowed tool requests may be returned in one turn. Apply a timeout to each provider request. Supplied spans and gold IDs are unavailable to the inference path; the harness may join them only after predictions have been saved.
Harness metrics and four run comparison
Development is A01 to A40 with 376 reference claim sentences. Use all scheduled answers and all gold claims, including answers with errors. For final checks A41 to A50, save predictions and errors without quality scores; the reviewer alone has their gold spans and labels.
Deterministic span matching
Match a prediction to a gold claim only when answer_id, start_char and end_char are identical. Validate quote equality first. A partial sentence or a merged paragraph is unmatched, even if its wording is relevant. Each gold span can match at most one prediction. Do not use an LLM judge, fuzzy matching or candidate-edited gold to inflate matches.
Metrics for both roles
Across the whole labeled split let G be gold claims, P accepted predictions, M exact span matches regardless of label, and C matches whose label is also correct. Extraction precision = M/P, recall = M/G, F1 = 2M/(P+G). Joint precision = C/P, recall = C/G, F1 = 2C/(P+G). Use zero for a zero denominator when the split has gold claims. An empty scheduled split has null quality metrics.
Joint F1 is the main quality comparison. Also report gold claim coverage C/G, missed gold IDs, unmatched prediction IDs, matched wrong-label IDs and answer error counts. Do not report matched-only label accuracy as overall accuracy. An answer error adds no accepted predictions, while every reference claim for that answer remains in G as missed. Extra predictions increase P and reduce precision.
Arithmetic check: G=10, P=10, M=8 and C=7 gives extraction precision, recall and F1 of 0.80 and joint precision, recall and F1 of 0.70. Seven correct labels among eight matched claims is 0.875, but that is not the overall quality score.
Comparison and evidence
Save v1 development, v2 development, v1 check and v2 check runs. Keep data, evaluator, provider, model and settings fixed while changing one explained agent behavior. Inspect extraction and labeling regressions. No score gain is an acceptable measured result; do not hardcode labels or invent improvement. Final-check inputs are visible, so disclose any tuning on them and do not claim a fully blind benchmark.
For check runs, G, M, C, matched gold IDs, expected labels, correctness and all quality metrics stay null. Save accepted predictions, error records, counts, timing and versions. The reviewer computes quality from the frozen saved outputs using the private key. Missing ground truth is not a zero score or an unknown label.
FastAPI PostgreSQL and controlled tests
Endpoint	Behavior
POST /classify	Accept the agent input and agent_version v1 or v2. Return 200 with validated extracted claims. Invalid requests or gold/claim-list input fields: 422. Unknown primary brand: 404. Provider or invalid output: 502. Timeout: 504. Use sanitized type and message.
POST /evaluation-runs	Accept agent_version and split development or check. Process the fixed CSV split. Return 201 with run_id, processing status, counts and available metrics. Preserve per-answer errors; check quality metrics stay null.
GET /evaluation-runs/{id}	Join PostgreSQL run metadata, answer outcomes and predicted claims. Return stable answer and span order. Missing run: 404; invalid UUID: 422.

Keep routes thin. Store runs, answer_results and claim_results linked by run_id. Keep one terminal answer row for every scheduled answer, including errors and valid empty outputs. Store one claim row per accepted prediction with its span, quote, label, fact IDs, reason and optional development match. Enforce unique (run_id, answer_id) and unique (run_id, answer_id, start_char, end_char); use foreign keys and parameterized SQL. JSONB may hold configuration, scores and missing/extra ID lists.
Run metadata includes split, agent/prompt/dataset/evaluator/fact versions, provider, model, settings, timestamp, source revision when available and file hashes. Answer rows include status, sanitized error, latency and trace reference when available. Quality metrics remain null for checks. A fully processed run may contain answer errors; an interrupted or infrastructure-failed run is failed. Preserve partial evidence. No production deployment, authentication, durable recovery or request idempotency is required.
Required controlled checks
•	Parse quoted commas and embedded newlines correctly; preserve ai_answer exactly; verify valid and invalid offsets, quote equality, duplicate IDs and overlapping spans.
•	Test scorer arithmetic with perfect output, missed claims, extra advice mistakenly extracted, wrong labels, a provider-error answer, valid empty predictions, and the G=10 P=10 M=8 C=7 example. Reject missing or duplicated answer outcomes.
•	Use a fake provider to test malformed output, unsupported labels, invented fact IDs, illegal tools, unknown requested brands, timeout and turn exhaustion. Confirm that gold never enters model or tool payloads.
•	Check API validation, database persistence and joined retrieval for successful, empty and error answers. Distinguish tests actually run from tests merely written. Fakes do not replace the four live runs.
A sequential runner is sufficient for interns. Use a configured provider-request timeout and document the value.
Submission and rubric
•	Submit an accessible Git repository link containing the Python agent, fact tool, harness, API/database setup, tests and v1/v2 prompts. Include dependency versions, exact install/run/test commands, DATABASE_URL setup and a secret-free environment template. For a private repository, provide the hiring team with access.
•	Share evidence along with the Git repository link: all four saved live-run reports, actual test output, and sanitized model/tool exchanges. Reports must include predictions, answer outcomes, development missing/extra/wrong-label records, metrics and settings. Preserve the issued CSVs or their exact hashes in the repository. Files in the repository or accessible evidence links are acceptable.
•	Include a sanitized actual model/tool exchange. Langfuse is optional, with no bonus or penalty.
•	Explain the targeted change, its measured effect, regressions or no gain, time spent, limitations and unfinished work. Disclose Claude Code or Codex help and what you verified yourself.
The recruiter supplies the deadline and submission channel. Provider usage may incur charges on your own account. Never submit keys or .env values. React is optional and is not assessed in R1. Use the separate CSV files; do not manually copy a dataset from this document.
Area	Points	Evidence
Extraction and classification agent	25	Real tool use; factual sentence extraction from full answers; valid spans, labels and fact references; constrained execution.
Evaluation harness	20	Exact span matching; extraction and joint precision recall F1; missed and extra claims; complete answer and failure accounting.
Improvement and analysis	15	Preserved baseline and candidate; failure diagnosis; one targeted change; fair reruns and honest regressions.
FastAPI and PostgreSQL	15	Validated endpoints; stored answer outcomes and predicted claims; constraints, parameterized SQL and joined retrieval.
Python reliability and tests	15	CSV parsing, output validation, timeout and turn limit; scorer arithmetic and controlled failure tests.
Reproducibility and ownership	10	Runnable handoff, actual artifacts, versioned inputs, limitations and explanation of AI assistance.

Total 100. Extraction and labeling quality, a correct harness and honest evidence matter. No minimum score gain or perfect-model-output requirement is hidden in the rubric.
