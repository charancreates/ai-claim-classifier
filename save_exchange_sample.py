import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
from kasparro.parsing import load_answers, load_facts
from kasparro.prompts import build_prompt
from google import genai
from google.genai import types

def generate_sanitized_exchange():
    answers = load_answers("data/answers.csv")
    facts = load_facts("data/brand_facts.csv")
    ans = answers[0]

    api_key = os.getenv("LLM_API_KEY")
    client = genai.Client(api_key=api_key)
    model = os.getenv("LLM_MODEL", "gemini-flash-lite-latest")

    tool_decl = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="get_brand_facts",
                description="Read-only catalogue fact lookup for a speaker brand.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "brand_id": types.Schema(type="STRING", description="Brand ID to look up facts for")
                    },
                    required=["brand_id"],
                ),
            )
        ]
    )

    prompt = build_prompt(ans, "v1")

    # Turn 1: Model request with tool declaration
    resp1 = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(tools=[tool_decl]),
    )

    tool_calls = []
    if resp1.function_calls:
        for call in resp1.function_calls:
            tool_calls.append({"name": call.name, "args": dict(call.args or {})})

    # Execute tool
    brand_facts = [f.__dict__ for f in facts if f.brand_id == ans.brand_id]

    # Turn 2: Provide facts and request claims
    turn2_prompt = (
        f"{prompt}\n\n"
        f"Catalogue facts returned from get_brand_facts:\n"
        f"{json.dumps(brand_facts, indent=2)}\n\n"
        "Now, extract all complete factual sentences from the target answer and classify each against the catalogue facts above.\n"
        "Output ONLY a valid JSON object matching the claims schema."
    )

    resp2 = client.models.generate_content(
        model=model,
        contents=turn2_prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )

    exchange = {
        "answer_id": ans.answer_id,
        "brand_id": ans.brand_id,
        "question": ans.question,
        "turn_1": {
            "role": "user",
            "prompt_excerpt": prompt[:300] + "... [truncated]",
            "tools_provided": ["get_brand_facts(brand_id: string)"],
            "model_response": {
                "function_calls": tool_calls
            }
        },
        "tool_execution": {
            "tool": "get_brand_facts",
            "arguments": {"brand_id": ans.brand_id},
            "fact_count": len(brand_facts),
            "sample_facts": brand_facts[:3]
        },
        "turn_2": {
            "role": "user",
            "instruction": "Catalogue facts provided. Extract and classify claims.",
            "model_response_claims": json.loads(resp2.text or "{}").get("claims", [])
        }
    }

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "sanitized_exchange.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(exchange, f, indent=2)
    print(f"Saved sanitized model/tool exchange to {out_file}")

if __name__ == "__main__":
    generate_sanitized_exchange()
