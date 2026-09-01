import os
from pathlib import Path
from typing import Any, Dict
import yaml

_CONFIG_CACHE = None

def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    # Try traversing upwards from this file
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "config.yaml").exists() or (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()

def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration YAML file with caching."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and config_path is None:
        return _CONFIG_CACHE

    if config_path is None:
        config_path = str(get_project_root() / "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    _CONFIG_CACHE = config
    return config

def get_path(path_key: str) -> Path:
    """Resolve a relative path key from config relative to the project root."""
    config = load_config()
    relative_path = config.get("paths", {}).get(path_key)
    if relative_path is None:
        raise KeyError(f"Path key '{path_key}' not defined in config.yaml under 'paths'")
    return get_project_root() / relative_path
