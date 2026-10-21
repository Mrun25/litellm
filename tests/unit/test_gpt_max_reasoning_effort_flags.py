import json
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
MAIN_MAP = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_MAP = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"

# Entries called out in https://github.com/BerriAI/litellm/issues/47884:
# the OpenAI/Azure/chatgpt entries must match the Bedrock ones, which already
# set supports_max_reasoning_effort.
FLAGGED_MODELS = (
    "gpt-5.6-sol",
    "gpt-5.6-luna",
    "gpt-5.6-terra",
    "azure/gpt-5.6-sol",
    "azure/gpt-5.6-luna",
    "azure/gpt-5.6-terra",
    "azure/us/gpt-5.6-sol",
    "azure/us/gpt-5.6-luna",
    "azure/us/gpt-5.6-terra",
    "azure/eu/gpt-5.6-sol",
    "azure/eu/gpt-5.6-luna",
    "azure/eu/gpt-5.6-terra",
    "chatgpt/gpt-5.6-sol",
    "chatgpt/gpt-5.6-luna",
    "chatgpt/gpt-5.6-terra",
    "azure/gpt-6-astra",
    "azure/us/gpt-6-astra",
    "azure/eu/gpt-6-astra",
    "azure_ai/gpt-6-astra",
)


def test_max_reasoning_effort_flagged_in_main_and_backup_maps():
    with open(MAIN_MAP) as f:
        main_cost = json.load(f)
    with open(BACKUP_MAP) as f:
        backup_cost = json.load(f)

    for model in FLAGGED_MODELS:
        for name, cost_map in (("main", main_cost), ("backup", backup_cost)):
            entry = cost_map.get(model)
            assert entry is not None, f"{model} missing from {name} model cost map"
            assert entry.get("supports_max_reasoning_effort") is True, (
                f"{model} must set supports_max_reasoning_effort in {name} model cost map"
            )
