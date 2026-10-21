import os
from pathlib import Path
from dotenv import load_dotenv

from .agent import Agent
from .api import create_app
from .database import make_session_factory
from .parsing import load_facts
from .provider import create_provider

load_dotenv()

data_dir = Path(os.getenv("DATA_DIR", "data"))
facts_file = data_dir / "brand_facts.csv"
facts = load_facts(facts_file) if facts_file.exists() else []

provider = create_provider(
    provider_type=os.getenv("LLM_PROVIDER", "gemini"),
    api_key=os.getenv("LLM_API_KEY"),
    model=os.getenv("LLM_MODEL", "gemini-flash-lite-latest"),
)

app = create_app(
    Agent(
        provider,
        facts,
        version=os.getenv("AGENT_VERSION", "v1"),
        timeout_seconds=float(os.getenv("PROVIDER_TIMEOUT_SECONDS", "30")),
    ),
    make_session_factory() if os.getenv("DATABASE_URL") else None,
)
