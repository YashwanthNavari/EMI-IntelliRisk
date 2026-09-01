import os
import json
import joblib
from pathlib import Path
from typing import Any, Dict, Optional
from ..utils.logger import setup_logger
from ..utils.config import get_project_root, load_config

logger = setup_logger("ModelRegistry")

def save_model_artifact(
    model_pipeline: Any,
    task_type: str,
    model_name: str,
    metadata: Dict[str, Any]
) -> Path:
    """Save trained pipeline and metadata JSON to disk."""
    root = get_project_root()
    save_dir = root / "models" / task_type
    save_dir.mkdir(parents=True, exist_ok=True)

    model_file = save_dir / f"best_{task_type}_model.joblib"
    meta_file = save_dir / f"best_{task_type}_metadata.json"

    # Save model pipeline
    joblib.dump(model_pipeline, str(model_file))

    # Save metadata
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved {task_type} model artifact to {model_file}")
    return model_file

def load_model_artifact(task_type: str) -> Optional[Any]:
    """Load serialized model pipeline from models/ directory."""
    root = get_project_root()
    model_file = root / "models" / task_type / f"best_{task_type}_model.joblib"

    if not model_file.exists():
        logger.warning(f"Model file not found at {model_file}")
        return None

    try:
        pipeline = joblib.load(str(model_file))
        logger.info(f"Loaded {task_type} model from {model_file}")
        return pipeline
    except Exception as e:
        logger.error(f"Failed loading {task_type} model: {e}")
        return None

def export_mlflow_summary(summary_data: Dict[str, Any]) -> Path:
    """Persist all experiment runs summary to models/mlflow_summary.json for Streamlit display."""
    root = get_project_root()
    export_path = root / "models" / "mlflow_summary.json"
    export_path.parent.mkdir(parents=True, exist_ok=True)

    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    logger.info(f"Exported MLflow experiments summary to {export_path}")
    return export_path
