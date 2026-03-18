import re
from pathlib import Path
from typing import Dict, Any

# legacy heuristic threshold
# score threshold for advanced model path
# higher => more restrictive
ROUTING_THRESHOLD = 2

# length threshold
LEN_THRESHOLD = 20

# models
models = {
    "small": {
        "name": "gpt-4o-mini",
        "description": "Fast, cost-efficient model for simple retrieval and formatting"
    },
    "advanced": {
        "name": "claude-3.5-sonnet",
        "description": "High-reasoning model for complex analysis, debugging and architecture"
    }
}

# regex patterns
SIGNALS = {
    "complex": re.compile(
        r'\b(analyze|debug|explain|detail|code|architectural|construct|advanced|compare|economic|military|custom|prove|theorem)\b',
        re.IGNORECASE
    ),
    "simple": re.compile(
        r'\b(translate|summarize|capital|what is| recipe|time|simple|overall|hi|reply|one sentence|random)\b',
        re.IGNORECASE
    )
}

# 14.03.2026 - semantic router & api config
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
LOW_CONFIDENCE_THRESHOLD: float = 0.15
BUDGET_CAP_DAILY: float = 10.0
COST_SIMPLE_USD: float = 0.001
COST_ADV_USD: float = 0.03
DB_PATH: Path = Path("data") / "routing_log.db"

if __name__ == "__main__":
    # smoke test to confirm constants and I/O path resolution without net access
    print(f"Config data loaded successfully. DB_PATH will resolve to {DB_PATH.absolute()}.")