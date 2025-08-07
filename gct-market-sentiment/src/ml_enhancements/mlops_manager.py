"""
MLOps Manager for TraderAI
Handles experiment tracking, model versioning, and performance monitoring
"""

import mlflow
import mlflow.sklearn
import mlflow.pytorch
import mlflow.xgboost
import mlflow.lightgbm
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import json
import logging
from datetime import datetime
from pathlib import Path
import pickle
import shutil

logger = logging.getLogger(__name__)


class MLOpsManager:
    """
    Comprehensive MLOps management for TraderAI models
    """
    
    def __init__(self, 
                 tracking_uri: str = "sqlite:///mlruns.db",
                 experiment_name: str = "TraderAI-Production",
                 artifact_location: Optional[str] = None):
        """
        Initialize MLOps Manager
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of the MLflow experiment
            artifact_location: Optional custom artifact storage location
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        
        # Set up MLflow
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_location
            )
        except:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id
            
        mlflow.set_experiment(experiment_name)
        
        # Initialize client
        self.client = MlflowClient(tracking_uri)
        
        logger.info(f"MLOps Manager initialized with experiment: {experiment_name}")
        
    def start_run(self, 
                  run_name: Optional[str] = None,
                  tags: Optional[Dict[str, str]] = None) -> str:
        """
        Start a new MLflow run
        
        Args:
            run_name: Optional name for the run
            tags: Optional tags to add to the run
            
        Returns:
            Run ID
        """
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        # Start run
        mlflow.start_run(run_name=run_name)
        
        # Add tags
        if tags:
            mlflow.set_tags(tags)
            
        # Add default tags
        mlflow.set_tag("framework", "TraderAI")
        mlflow.set_tag("version", "1.0")
        
        run_id = mlflow.active_run().info.run_id
        logger.info(f"Started MLflow run: {run_id}")
        
        return run_id
        
    def log_model_training(self,
                          model: Any,
                          model_type: str,
                          params: Dict[str, Any],
                          metrics: Dict[str, float],
                          features: List[str],
                          feature_importance: Optional[pd.DataFrame] = None,
                          training_data_stats: Optional[Dict] = None,
                          artifacts: Optional[Dict[str, str]] = None):
        """
        Log comprehensive model training information
        
        Args:
            model: Trained model object
            model_type: Type of model (xgboost, lightgbm, ensemble, etc.)
            params: Model hyperparameters
            metrics: Performance metrics
            features: List of feature names
            feature_importance: Optional feature importance DataFrame
            training_data_stats: Optional statistics about training data
            artifacts: Optional additional artifacts to log
        """
        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("n_features", len(features))
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log feature list
            with open("features.json", "w") as f:
                json.dump(features, f)
            mlflow.log_artifact("features.json")
            
            # Log feature importance
            if feature_importance is not None:
                feature_importance.to_csv("feature_importance.csv")
                mlflow.log_artifact("feature_importance.csv")
                
                # Log feature importance plot
                try:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(10, 6))
                    top_features = feature_importance.head(20)
                    plt.barh(range(len(top_features)), top_features.values)
                    plt.yticks(range(len(top_features)), top_features.index)
                    plt.xlabel("Importance")
                    plt.title("Top 20 Feature Importances")
                    plt.tight_layout()
                    plt.savefig("feature_importance_plot.png")
                    mlflow.log_artifact("feature_importance_plot.png")
                    plt.close()
                except Exception as e:
                    logger.warning(f"Could not create feature importance plot: {e}")
                    
            # Log training data statistics
            if training_data_stats:
                mlflow.log_params({
                    f"train_data_{k}": v 
                    for k, v in training_data_stats.items()
                })
                
            # Log model based on type
            if model_type == "xgboost":
                mlflow.xgboost.log_model(model, "model", 
                                        input_example=pd.DataFrame(columns=features))
            elif model_type == "lightgbm":
                mlflow.lightgbm.log_model(model, "model")
            elif model_type == "sklearn" or model_type == "ensemble":
                mlflow.sklearn.log_model(model, "model", 
                                       input_example=pd.DataFrame(columns=features))
            elif model_type == "pytorch":
                mlflow.pytorch.log_model(model, "model")
            else:
                # Generic model logging
                with open("model.pkl", "wb") as f:
                    pickle.dump(model, f)
                mlflow.log_artifact("model.pkl")
                
            # Log additional artifacts
            if artifacts:
                for name, path in artifacts.items():
                    mlflow.log_artifact(path, name)
                    
            # Log model info
            model_info = {
                "model_type": model_type,
                "n_features": len(features),
                "training_date": datetime.now().isoformat(),
                "metrics": metrics
            }
            with open("model_info.json", "w") as f:
                json.dump(model_info, f)
            mlflow.log_artifact("model_info.json")
            
            logger.info(f"Logged model training for run: {run.info.run_id}")
            
            # Clean up temporary files
            for file in ["features.json", "feature_importance.csv", 
                        "feature_importance_plot.png", "model.pkl", "model_info.json"]:
                if Path(file).exists():
                    Path(file).unlink()
                    
    def log_prediction_results(self,
                              predictions: np.ndarray,
                              actuals: Optional[np.ndarray] = None,
                              timestamps: Optional[pd.DatetimeIndex] = None,
                              metadata: Optional[Dict] = None):
        """Log prediction results for monitoring"""
        with mlflow.start_run():
            # Calculate metrics if actuals provided
            if actuals is not None:
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                
                mae = mean_absolute_error(actuals, predictions)
                mse = mean_squared_error(actuals, predictions)
                rmse = np.sqrt(mse)
                r2 = r2_score(actuals, predictions)
                
                mlflow.log_metrics({
                    "mae": mae,
                    "mse": mse,
                    "rmse": rmse,
                    "r2": r2
                })
                
                # Log prediction accuracy by percentile
                errors = np.abs(predictions - actuals)
                for percentile in [50, 75, 90, 95, 99]:
                    mlflow.log_metric(
                        f"error_p{percentile}", 
                        np.percentile(errors, percentile)
                    )
                    
            # Log prediction statistics
            mlflow.log_metrics({
                "predictions_mean": float(np.mean(predictions)),
                "predictions_std": float(np.std(predictions)),
                "predictions_min": float(np.min(predictions)),
                "predictions_max": float(np.max(predictions)),
                "predictions_count": len(predictions)
            })
            
            # Save predictions to file
            results_df = pd.DataFrame({
                "predictions": predictions,
                "timestamps": timestamps if timestamps is not None else range(len(predictions))
            })
            if actuals is not None:
                results_df["actuals"] = actuals
                results_df["errors"] = predictions - actuals
                
            results_df.to_csv("predictions.csv", index=False)
            mlflow.log_artifact("predictions.csv")
            
            # Log metadata
            if metadata:
                mlflow.log_params(metadata)
                
            Path("predictions.csv").unlink()
            
    def compare_models(self, 
                      run_ids: List[str],
                      metric: str = "rmse") -> pd.DataFrame:
        """
        Compare multiple model runs
        
        Args:
            run_ids: List of MLflow run IDs to compare
            metric: Metric to use for comparison
            
        Returns:
            DataFrame with model comparison
        """
        comparison_data = []
        
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            
            model_data = {
                "run_id": run_id,
                "run_name": run.data.tags.get("mlflow.runName", ""),
                "model_type": run.data.params.get("model_type", ""),
                "n_features": run.data.params.get("n_features", ""),
                metric: run.data.metrics.get(metric, np.nan),
                "mae": run.data.metrics.get("mae", np.nan),
                "r2": run.data.metrics.get("r2", np.nan),
                "training_time": run.data.metrics.get("training_time", np.nan),
            }
            
            comparison_data.append(model_data)
            
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values(metric)
        
        return comparison_df
        
    def register_model(self,
                      run_id: str,
                      model_name: str,
                      tags: Optional[Dict[str, str]] = None) -> str:
        """
        Register a model for production
        
        Args:
            run_id: MLflow run ID
            model_name: Name for the registered model
            tags: Optional tags for the model
            
        Returns:
            Model version
        """
        # Create model URI
        model_uri = f"runs:/{run_id}/model"
        
        # Register model
        model_version = mlflow.register_model(model_uri, model_name)
        
        # Add tags
        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(
                    model_name, 
                    model_version.version,
                    key, 
                    value
                )
                
        logger.info(f"Registered model {model_name} version {model_version.version}")
        
        return model_version.version
        
    def transition_model_stage(self,
                              model_name: str,
                              version: str,
                              stage: str,
                              archive_existing: bool = True):
        """
        Transition model to a new stage
        
        Args:
            model_name: Name of the registered model
            version: Model version
            stage: Target stage (Staging, Production, Archived)
            archive_existing: Whether to archive existing models in target stage
        """
        self.client.transition_model_version_stage(
            model_name,
            version,
            stage,
            archive_existing_versions=archive_existing
        )
        
        logger.info(f"Transitioned {model_name} v{version} to {stage}")
        
    def load_production_model(self, model_name: str) -> Any:
        """Load the current production model"""
        # Get production model
        production_models = self.client.get_latest_versions(
            model_name, 
            stages=["Production"]
        )
        
        if not production_models:
            raise ValueError(f"No production model found for {model_name}")
            
        model_version = production_models[0]
        model_uri = f"models:/{model_name}/{model_version.version}"
        
        # Load model based on flavor
        if "sklearn" in model_version.flavors:
            model = mlflow.sklearn.load_model(model_uri)
        elif "xgboost" in model_version.flavors:
            model = mlflow.xgboost.load_model(model_uri)
        elif "lightgbm" in model_version.flavors:
            model = mlflow.lightgbm.load_model(model_uri)
        elif "pytorch" in model_version.flavors:
            model = mlflow.pytorch.load_model(model_uri)
        else:
            # Try generic loading
            model = mlflow.pyfunc.load_model(model_uri)
            
        logger.info(f"Loaded production model {model_name} v{model_version.version}")
        
        return model
        
    def log_data_drift(self,
                      reference_data: pd.DataFrame,
                      current_data: pd.DataFrame,
                      feature_columns: List[str],
                      drift_metrics: Dict[str, float]):
        """Log data drift metrics"""
        with mlflow.start_run():
            # Log drift metrics
            mlflow.log_metrics({
                f"drift_{k}": v for k, v in drift_metrics.items()
            })
            
            # Log data statistics
            for col in feature_columns:
                if col in reference_data.columns and col in current_data.columns:
                    mlflow.log_metrics({
                        f"{col}_ref_mean": reference_data[col].mean(),
                        f"{col}_ref_std": reference_data[col].std(),
                        f"{col}_curr_mean": current_data[col].mean(),
                        f"{col}_curr_std": current_data[col].std(),
                    })
                    
            # Create drift report
            drift_report = {
                "timestamp": datetime.now().isoformat(),
                "n_features": len(feature_columns),
                "reference_size": len(reference_data),
                "current_size": len(current_data),
                "drift_metrics": drift_metrics
            }
            
            with open("drift_report.json", "w") as f:
                json.dump(drift_report, f)
            mlflow.log_artifact("drift_report.json")
            
            Path("drift_report.json").unlink()
            
    def cleanup_old_runs(self, 
                        keep_last_n: int = 10,
                        keep_production: bool = True):
        """Clean up old experiment runs"""
        # Get all runs
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=["start_time DESC"]
        )
        
        # Identify runs to delete
        runs_to_delete = []
        production_run_ids = set()
        
        if keep_production:
            # Get all production model run IDs
            models = self.client.list_registered_models()
            for model in models:
                versions = self.client.get_latest_versions(
                    model.name, 
                    stages=["Production", "Staging"]
                )
                for version in versions:
                    production_run_ids.add(version.run_id)
                    
        # Select runs to delete
        for i, run in enumerate(runs):
            if i >= keep_last_n and run.info.run_id not in production_run_ids:
                runs_to_delete.append(run.info.run_id)
                
        # Delete runs
        for run_id in runs_to_delete:
            self.client.delete_run(run_id)
            logger.info(f"Deleted run: {run_id}")
            
        logger.info(f"Cleaned up {len(runs_to_delete)} old runs")