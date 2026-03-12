import re

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