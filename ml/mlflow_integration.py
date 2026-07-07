# mlflow_integration.py
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import os
import json
import base64
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import pickle

class MLflowManager:
    """Manages MLflow experiment tracking for the data science agent"""  

    def __init__(self):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("Data_Science_Agent")
        self.experiment = mlflow.get_experiment_by_name("Data_Science_Agent")
    
    def log_regression_run(self, 
                          model, 
                          model_name: str, 
                          task_info: Dict[str, Any],
                          results: Dict[str, Any],
                          visualizations: List[Dict],
                          dataset_info: Dict[str, Any]) -> str:
        """Log regression experiment to MLflow"""
        
        run_name = f"regression_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_param("task_type", "regression")
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("target_column", task_info.get('target_column', 'unknown'))
            mlflow.log_param("model_name", model_name)
            
            # Log metrics
            mlflow.log_metric("r2_score", results.get('r2', 0))
            mlflow.log_metric("rmse", results.get('rmse', 0))
            mlflow.log_metric("mse", results.get('mse', 0))
            mlflow.log_metric("mae", results.get('mae', 0))
            
            # Log dataset info
            mlflow.log_param("dataset_rows", dataset_info.get('total_rows', 0))
            mlflow.log_param("dataset_columns", len(dataset_info.get('columns', [])))
            mlflow.log_param("numeric_columns", len(dataset_info.get('numeric_columns', [])))
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Log visualizations as artifacts
            for i, viz in enumerate(visualizations[:5]):  # Limit to 5 visualizations
                if viz.get('type') == 'image' and viz.get('content'):
                    try:
                        # Decode base64 image and save as file
                        img_data = base64.b64decode(viz['content'])
                        viz_path = f"visualizations/{viz.get('name', f'viz_{i}')}.png"
                        os.makedirs(os.path.dirname(viz_path), exist_ok=True)
                        with open(viz_path, "wb") as f:
                            f.write(img_data)
                        mlflow.log_artifact(viz_path)
                    except Exception as e:
                        print(f"Failed to log visualization: {e}")
            
            # Save results as JSON
            results_path = "data/results/results.json"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            mlflow.log_artifact(results_path)
            
            # Log tags
            mlflow.set_tag("task", "regression")
            mlflow.set_tag("automl", "true")
            mlflow.set_tag("agent_version", "1.0")
            
            run_id = mlflow.active_run().info.run_id
            print(f"✅ Regression run logged to MLflow: {run_id}")
            
            return run_id
    
    def log_classification_run(self, 
                              model, 
                              model_name: str, 
                              task_info: Dict[str, Any],
                              results: Dict[str, Any],
                              visualizations: List[Dict],
                              dataset_info: Dict[str, Any]) -> str:
        """Log classification experiment to MLflow"""
        
        run_name = f"classification_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_param("task_type", "classification")
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("target_column", task_info.get('target_column', 'unknown'))
            mlflow.log_param("model_name", model_name)
            
            # Log metrics
            mlflow.log_metric("accuracy", results.get('accuracy', 0))
            mlflow.log_metric("f1_score", results.get('f1_score', 0))
            mlflow.log_metric("precision", results.get('precision', 0))
            mlflow.log_metric("recall", results.get('recall', 0))
            
            # Log dataset info
            mlflow.log_param("dataset_rows", dataset_info.get('total_rows', 0))
            mlflow.log_param("dataset_columns", len(dataset_info.get('columns', [])))
            mlflow.log_param("numeric_columns", len(dataset_info.get('numeric_columns', [])))
            mlflow.log_param("categorical_columns", len(dataset_info.get('categorical_columns', [])))
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Log visualizations as artifacts
            for i, viz in enumerate(visualizations[:5]):
                if viz.get('type') == 'image' and viz.get('content'):
                    try:
                        img_data = base64.b64decode(viz['content'])
                        viz_path = f"visualizations/{viz.get('name', f'viz_{i}')}.png"
                        os.makedirs(os.path.dirname(viz_path), exist_ok=True)
                        with open(viz_path, "wb") as f:
                            f.write(img_data)
                        mlflow.log_artifact(viz_path)
                    except Exception as e:
                        print(f"Failed to log visualization: {e}")
            
            # Save results as JSON
            results_path = "results.json"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            mlflow.log_artifact(results_path)
            
            # Log tags
            mlflow.set_tag("task", "classification")
            mlflow.set_tag("automl", "true")
            mlflow.set_tag("agent_version", "1.0")
            
            run_id = mlflow.active_run().info.run_id
            print(f"✅ Classification run logged to MLflow: {run_id}")
            
            return run_id
    
    def log_clustering_run(self, 
                          model, 
                          model_name: str, 
                          task_info: Dict[str, Any],
                          results: Dict[str, Any],
                          visualizations: List[Dict],
                          dataset_info: Dict[str, Any]) -> str:
        """Log clustering experiment to MLflow"""
        
        run_name = f"clustering_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_param("task_type", "clustering")
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("clustering_method", model_name)
            mlflow.log_param("n_clusters", results.get('n_clusters', 0))
            
            # Log metrics
            evaluation = results.get('evaluation', {})
            mlflow.log_metric("silhouette_score", evaluation.get('silhouette_score', -1))
            mlflow.log_metric("davies_bouldin_score", evaluation.get('davies_bouldin_score', float('inf')))
            mlflow.log_metric("calinski_harabasz_score", evaluation.get('calinski_harabasz_score', 0))
            
            # Log dataset info
            mlflow.log_param("dataset_rows", dataset_info.get('total_rows', 0))
            mlflow.log_param("dataset_columns", len(dataset_info.get('columns', [])))
            mlflow.log_param("numeric_columns", len(dataset_info.get('numeric_columns', [])))
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Log visualizations as artifacts
            for i, viz in enumerate(visualizations[:5]):
                if viz.get('type') == 'image' and viz.get('content'):
                    try:
                        img_data = base64.b64decode(viz['content'])
                        viz_path = f"visualizations/{viz.get('name', f'viz_{i}')}.png"
                        os.makedirs(os.path.dirname(viz_path), exist_ok=True)
                        with open(viz_path, "wb") as f:
                            f.write(img_data)
                        mlflow.log_artifact(viz_path)
                    except Exception as e:
                        print(f"Failed to log visualization: {e}")
            
            # Save results as JSON
            results_path = "results.json"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            mlflow.log_artifact(results_path)
            
            # Log tags
            mlflow.set_tag("task", "clustering")
            mlflow.set_tag("automl", "true")
            mlflow.set_tag("agent_version", "1.0")
            
            run_id = mlflow.active_run().info.run_id
            print(f"✅ Clustering run logged to MLflow: {run_id}")
            
            return run_id
    
    def log_multi_model_comparison(self,
                                  task_type: str,
                                  comparison_results: Dict[str, Any],
                                  best_model,
                                  best_model_name: str,
                                  visualizations: List[Dict],
                                  dataset_info: Dict[str, Any]) -> str:
        """Log multi-model comparison to MLflow"""
        
        run_name = f"comparison_{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_param("task_type", task_type)
            mlflow.log_param("comparison_type", "multi_model")
            mlflow.log_param("best_model", best_model_name)
            mlflow.log_param("total_models", len(comparison_results.get('all_models', {})))
            
            # Log comparison metrics
            all_models = comparison_results.get('all_models', {})
            for model_name, metrics in all_models.items():
                if task_type == 'regression':
                    mlflow.log_metric(f"{model_name}_r2", metrics.get('r2', 0))
                    mlflow.log_metric(f"{model_name}_rmse", metrics.get('rmse', 0))
                else:  # classification
                    mlflow.log_metric(f"{model_name}_accuracy", metrics.get('accuracy', 0))
                    mlflow.log_metric(f"{model_name}_f1", metrics.get('f1_score', 0))
            
            # Log best model metrics
            mlflow.log_metric("best_score", comparison_results.get('best_score', 0))
            
            # Log dataset info
            mlflow.log_param("dataset_rows", dataset_info.get('total_rows', 0))
            mlflow.log_param("dataset_columns", len(dataset_info.get('columns', [])))
            
            # Log best model
            mlflow.sklearn.log_model(best_model, "best_model")
            
            # Log comparison visualizations
            for i, viz in enumerate(visualizations[:3]):
                if viz.get('type') == 'image' and viz.get('content'):
                    try:
                        img_data = base64.b64decode(viz['content'])
                        viz_path = f"comparison_viz/{viz.get('name', f'comparison_viz_{i}')}.png"
                        os.makedirs(os.path.dirname(viz_path), exist_ok=True)
                        with open(viz_path, "wb") as f:
                            f.write(img_data)
                        mlflow.log_artifact(viz_path)
                    except Exception as e:
                        print(f"Failed to log comparison visualization: {e}")
            
            # Save comparison results
            comparison_path = "data/results/comparison_results.json"
            with open(comparison_path, "w") as f:
                json.dump(comparison_results, f, indent=2, default=str)
            mlflow.log_artifact(comparison_path)
            
            # Log tags
            mlflow.set_tag("task", f"{task_type}_comparison")
            mlflow.set_tag("automl", "true")
            mlflow.set_tag("multi_model", "true")
            mlflow.set_tag("agent_version", "1.0")
            
            run_id = mlflow.active_run().info.run_id
            print(f"✅ Multi-model comparison logged to MLflow: {run_id}")
            
            return run_id
    
    def list_runs(self, experiment_name: str = None) -> List[Dict]:
        """List all MLflow runs"""
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        
        runs = mlflow.search_runs()
        return runs.to_dict('records')
    
    def get_run_details(self, run_id: str) -> Dict[str, Any]:
        """Get details of a specific run"""
        run = mlflow.get_run(run_id)
        return {
            'run_id': run.info.run_id,
            'experiment_id': run.info.experiment_id,
            'status': run.info.status,
            'start_time': run.info.start_time,
            'end_time': run.info.end_time,
            'params': run.data.params,
            'metrics': run.data.metrics,
            'tags': run.data.tags
        }

# Create a singleton instance
mlflow_manager = MLflowManager()