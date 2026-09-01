import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

from ..utils.logger import setup_logger
from ..utils.config import get_project_root, load_config

logger = setup_logger("MLflowTracker")

class MLflowExperimentTracker:
    """Manages MLflow experiment lifecycle, logging, and artifact generation."""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.config = load_config()
        self.root = get_project_root()
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        mlflow_db = (self.root / "database" / "mlflow.db").as_posix()
        self.tracking_uri = f"sqlite:///{mlflow_db}"

        if HAS_MLFLOW:
            try:
                mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(experiment_name)
                logger.info(f"MLflow initialized for experiment '{experiment_name}' at {self.tracking_uri}")
            except Exception as e:
                logger.warning(f"MLflow URI init notice: {e}")
                # Fallback to local experiment tracking
                pass
        else:
            logger.warning("MLflow package not available; running in offline artifact logging mode.")

    def log_run(
        self,
        run_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        artifacts: Optional[Dict[str, Any]] = None,
        model: Optional[Any] = None
    ) -> str:
        """Log parameters, metrics, and figures to MLflow run."""
        run_id = f"run_{run_name}_{int(np.random.randint(1000, 9999))}"

        if HAS_MLFLOW:
            with mlflow.start_run(run_name=run_name) as run:
                run_id = run.info.run_id
                # Log params
                for k, v in params.items():
                    mlflow.log_param(k, str(v))

                # Log metrics
                for k, v in metrics.items():
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        mlflow.log_metric(k, float(v))

                # Log model
                if model is not None:
                    try:
                        mlflow.sklearn.log_model(model, artifact_path="model")
                    except Exception as e:
                        logger.warning(f"Could not log sklearn model directly in MLflow: {e}")

                # Log artifacts
                if artifacts:
                    for art_name, art_obj in artifacts.items():
                        temp_path = self.root / "reports" / f"temp_{art_name}.png"
                        temp_path.parent.mkdir(parents=True, exist_ok=True)
                        if isinstance(art_obj, plt.Figure):
                            art_obj.savefig(str(temp_path), bbox_inches="tight")
                            mlflow.log_artifact(str(temp_path), artifact_path="figures")
                            if temp_path.exists():
                                temp_path.unlink()

        logger.info(f"Logged experiment run '{run_name}' (ID: {run_id})")
        return run_id
