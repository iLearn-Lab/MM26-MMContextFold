"""
configs/
Configuration loader & API Keys management
"""

import os
import yaml
from typing import Dict, Any

from .api_keys import (
    SERPER_API_KEY,
    SERPAPI_API_KEY,
    JINA_API_KEY,
    OPENROUTER_API_KEY,
    setup_env,
)


def load_config(config_name: str = "base") -> Dict[str, Any]:
    config_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(config_dir, f"{config_name}.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if "_base_" in config:
        base_name = config.pop("_base_").replace(".yaml", "")
        base_config = load_config(base_name)
        config = _merge_configs(base_config, config)

    return config


def _merge_configs(base: Dict, override: Dict) -> Dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def get_dataset_config(config: Dict, dataset_name: str) -> Dict[str, str]:
    datasets = config.get("datasets", {})
    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return datasets[dataset_name]


def get_model_config(config: Dict, model_name: str) -> Dict[str, str]:
    models = config.get("models", {})
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    return models[model_name]
