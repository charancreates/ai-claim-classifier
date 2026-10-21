from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from .agent import Provider, ProviderError


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    candidate = match.group(1).strip() if match else text
    try:
        data = json.loads(candidate)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    obj_match = re.search(r"\{[\s\S]*\}", text)
    if obj_match:
        try:
            data = json.loads(obj_match.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    raise ProviderError("invalid_output", "Could not parse JSON object from provider output.")


class GeminiProvider:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.model = model or os.getenv("LLM_MODEL", "gemini-flash-lite-latest")
        if not self.api_key:
            raise ValueError("Gemini API key is required.")
        from google import genai
        try:
            self.client = genai.Client(api_key=self.api_key)
        except Exception:
            self.client = genai.Client(vertexai=True, api_key=self.api_key)

    def _generate_with_retry(self, contents: Any, config: Any = None) -> Any:
        last_error = None
        for attempt in range(6):
            try:
                kwargs = {"model": self.model, "contents": contents}
                if config:
                    kwargs["config"] = config
                return self.client.models.generate_content(**kwargs)
            except Exception as e:
                last_error = e
                err_str = str(e)
                if any(code in err_str for code in ["503", "429", "RESOURCE_EXHAUSTED", "UNAVAILABLE"]):
                    delay_match = re.search(r"retryDelay':\s*'(\d+)s'", err_str) or re.search(r"retry in (\d+(?:\.\d+)?)s", err_str)
                    if delay_match:
                        sleep_time = float(delay_match.group(1)) + 1.5
                    else:
                        sleep_time = max(4.0, 2.5 * (attempt + 1))
                    time.sleep(sleep_time)
                    continue
                raise ProviderError("provider_error", f"Gemini request failed: {e}") from e
        raise ProviderError("provider_error", f"Gemini request failed after retries: {last_error}") from last_error

    def complete(self, prompt: str, tools: list[dict[str, Any]], timeout_seconds: float) -> dict[str, Any]:
        from google.genai import types

        # Turn 2+: Catalogue facts have been provided via tool execution
        if "Catalogue facts returned from get_brand_facts" in prompt or ("tool" in prompt and "result" in prompt):
            response = self._generate_with_retry(
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            return _extract_json(response.text or "{}")

        # Turn 1: Primary prompt with tool declaration
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

        response = self._generate_with_retry(
            contents=prompt,
            config=types.GenerateContentConfig(tools=[tool_decl]),
        )

        actions: list[dict[str, Any]] = []
        if response.function_calls:
            for call in response.function_calls:
                actions.append({
                    "tool": call.name,
                    "arguments": dict(call.args or {}),
                    "brand_id": (call.args or {}).get("brand_id", ""),
                })
            return {"actions": actions}

        return _extract_json(response.text or "{}")


class MockProvider:
    """Mock provider for deterministic offline testing."""
    def __init__(self, brand_facts: list[Any] | None = None):
        self.brand_facts = brand_facts or []

    def complete(self, prompt: str, tools: list[dict[str, Any]], timeout_seconds: float) -> dict[str, Any]:
        if "Catalogue facts returned from get_brand_facts" in prompt or ("tool" in prompt and "result" in prompt):
            return {"claims": []}
        brand_match = re.search(r"Primary brand:\s*(\w+)", prompt)
        brand = brand_match.group(1) if brand_match else "cleartrail"
        return {
            "actions": [{"tool": "get_brand_facts", "arguments": {"brand_id": brand}}]
        }


def create_provider(provider_type: str | None = None, api_key: str | None = None, model: str | None = None) -> Provider:
    p_type = (provider_type or os.getenv("LLM_PROVIDER", "gemini")).lower()
    if p_type == "gemini":
        return GeminiProvider(api_key=api_key, model=model or os.getenv("LLM_MODEL", "gemini-flash-lite-latest"))
    if p_type == "mock":
        return MockProvider()
    raise ValueError(f"Unsupported provider: {p_type}. Supported: 'gemini', 'mock'")
