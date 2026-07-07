import os
import json
import io
import base64
import traceback
from io import BytesIO
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
import numpy as np

# Enhanced ML Libraries
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    AdaBoostRegressor, AdaBoostClassifier,
    BaggingRegressor, BaggingClassifier,
    VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor,
    ExtraTreesRegressor, ExtraTreesClassifier
)
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, classification_report, silhouette_score,
    confusion_matrix, precision_score, recall_score, f1_score,
    davies_bouldin_score, calinski_harabasz_score
)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_regression, f_classif
from sklearn.naive_bayes import GaussianNB

# Visualization (server-safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Utils (STRUCTURE FIX)
from utils.model_persistence import ModelPersistenceManager

# MLflow (STRUCTURE FIX)
try:
    from ml.mlflow_integration import mlflow_manager
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False



class EnhancedDataScienceAgent:
    def __init__(self, df: pd.DataFrame, azure_client=None):
         
        df.columns = [str(col).strip('"').strip("'").strip() for col in df.columns]
        self.df = df.copy()
        row_count = len(df)
        
        if row_count > 500000:
            sample_size = 50000
            print(f" Very large dataset ({row_count} rows). Using {sample_size} rows for faster processing.")
            self.df = df.sample(n=sample_size, random_state=42)
        elif row_count > 100000:
            sample_size = 30000
            print(f"⚠️ Large dataset ({row_count} rows). Using {sample_size} rows for faster processing.")
            self.df = df.sample(n=sample_size, random_state=42)
        elif row_count > 50000:
            sample_size = 20000
            print(f"📊 Medium-large dataset ({row_count} rows). Using {sample_size} rows.")
            self.df = df.sample(n=sample_size, random_state=42)
        elif row_count > 10000:
            sample_size = 10000
            print(f"📊 Medium dataset ({row_count} rows). Using {sample_size} rows.")
            self.df = df.sample(n=sample_size, random_state=42)
        else:
            print(f"✅ Small dataset ({row_count} rows). Using full data.")
            self.df = df.copy()
        
        # Store sampling info
        self.sampling_info = {
            'original_rows': row_count,
            'used_rows': len(self.df),
            'sampling_applied': row_count > 10000
        }
        self.azure_client = azure_client
        self.detect_column_types()
        self.models = {}  # Now will store multiple models
        self.results = {}
        self.visualizations = []
        self.label_encoders = {}
        self.scaler = None
        self.X_columns = None
        self.y_column = None
        self.model_manager = ModelPersistenceManager()  # Add model persistence manager
        self.dataset_hash = self.model_manager.calculate_dataset_hash(df)
        
    def _get_model_filename(self, model_name: str, task_type: str) -> str:
        """Generate consistent filename for model"""
        if hasattr(self, 'filename') and self.filename:
            base_name = os.path.splitext(self.filename)[0]
            # Remove special characters from model name
            clean_model_name = model_name.replace('_', '-')
            return f"{task_type}_{clean_model_name}_{base_name}.pkl"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{task_type}_{model_name}_{timestamp}.pkl"    
        
        
    def train_and_compare_regression_models(self, target_column: str) -> Dict[str, Any]:
        """Train multiple regression models and compare them"""
        print(f"\n🧮 Training multiple regression models for target: {target_column}")
        
        df_processed = self.preprocess_data(target_column)
        X = df_processed.drop(columns=[target_column])
        y = df_processed[target_column]
        self.X_columns = X.columns.tolist()
        self.y_column = target_column
        
        # Split data
        test_size = 0.2 if len(df_processed) > 10 else 0.5
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models to train
        regression_models = {
            'linear_regression': LinearRegression(),
            # 'decision_tree_regressor': DecisionTreeRegressor(random_state=42),
            'random_forest_regressor': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting_regressor': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'ridge_regression': Ridge(alpha=1.0),
            # 'lasso_regression': Lasso(alpha=0.1),
            'svr': SVR(kernel='rbf')
        }
        
        # Train and evaluate each model
        model_results = {}
        best_score = -np.inf
        best_model_name = None
        
        #  FIX: Define base_name and unique_id HERE (outside any condition)
        if hasattr(self, 'filename') and self.filename:
            # Remove extension and clean filename for folder name
            base_name = os.path.splitext(self.filename)[0]
            # Remove special characters if any
            base_name = base_name.replace(' ', '_').replace('-', '_').replace('.', '_').replace('"', '').replace("'", "")
            # Add timestamp for uniqueness
            unique_id = f"{base_name}"
        else:
            # Fallback to target column if filename not available
            base_name = target_column.replace(' ', '_').replace('"', '').replace("'", "")
            unique_id = f"{base_name}"
        
        # Main folder name based on unique_id
        # main_folder = f"regression_{unique_id}"
              
        if hasattr(self, 'filename') and self.filename:
            base_filename = os.path.splitext(self.filename)[0]
            clean_filename = base_filename.replace(' ', '_').replace('-', '_').replace('.', '')
            clean_filename = os.path.basename(clean_filename)
            main_folder = f"regression_{clean_filename}"  # ✅ regression_1_Car_Price_train
            print(f"📁 Saving models in folder: {main_folder} (from filename: {self.filename})")
        else:
            # Fallback to target column if filename not available
            clean_target = target_column.replace(' ', '_').replace('-', '_').replace('.', '')
            main_folder = f"regression_{clean_target}"
            print(f"📁 Using target column for folder: {main_folder}")
        # ✅ List of models to save (5 models)
        models_to_save = [
            'linear_regression',
            'random_forest_regressor', 
            'gradient_boosting_regressor',
            'ridge_regression'
        ]
        
        # ✅ List to store saved model IDs
        saved_model_ids = []
        
        for model_name, model in regression_models.items():
            print(f"  Training {model_name}...")
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                model_results[model_name] = {
                    'model': model,
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'r2': float(r2),
                    'mae': float(mae),
                    'predictions': y_pred.tolist()[:50],
                    'actuals': y_test.tolist()[:50]
                }
                
                print(f"    R²: {r2:.4f}, RMSE: {rmse:.4f}")
                
                # Update best model
                if r2 > best_score:
                    best_score = r2
                    best_model_name = model_name
                
                # ✅ Save ONLY the 5 specified models
                if model_name in models_to_save:
                    try:
                        # Create metadata for this model
                        metadata = {
                            'task_type': 'regression',
                            'target_column': target_column,
                            'model_name': model_name,
                            'metrics': {
                                'r2': float(r2),
                                'rmse': float(rmse),
                                'mse': float(mse),
                                'mae': float(mae)
                            },
                            'feature_count': len(self.X_columns),
                            'feature_columns': self.X_columns,
                            'dataset_hash': self.dataset_hash,
                            'timestamp': datetime.now().isoformat(),
                            'filename': self.filename if hasattr(self, 'filename') else None,
                            'unique_folder': main_folder
                        }
                        
                        # ✅ Create model ID with unique folder structure
                        model_id = f"{main_folder}/regression_{model_name}_{base_name}"
                        
                        # Save the model
                        saved_id = self.model_manager.save_model(
                            model_id=model_id,
                            model=model,
                            scaler=self.scaler,
                            label_encoders=self.label_encoders,
                            feature_columns=self.X_columns,
                            metadata=metadata
                        )
                        
                        saved_model_ids.append(saved_id)
                        print(f"    ✅ Saved: {model_id}")
                        
                    except Exception as save_error:
                        print(f"    ⚠️ Could not save {model_name}: {save_error}")
                
            except Exception as e:
                print(f"    Error training {model_name}: {e}")
                continue
        
        # Create ensemble model
        if len(model_results) >= 3:
            # Get top 3 models by R² score
            sorted_models = sorted(model_results.items(), 
                                key=lambda x: x[1]['r2'], 
                                reverse=True)[:3]
            
            top_models = [(name, data['model']) for name, data in sorted_models]
            ensemble_name = "ensemble_average"
            
            # Create VotingRegressor
            from sklearn.ensemble import VotingRegressor
            ensemble_model = VotingRegressor(estimators=top_models)
            ensemble_model.fit(X_train_scaled, y_train)
            y_pred_ensemble = ensemble_model.predict(X_test_scaled)
            
            # Calculate metrics
            mse_ensemble = mean_squared_error(y_test, y_pred_ensemble)
            rmse_ensemble = np.sqrt(mse_ensemble)
            r2_ensemble = r2_score(y_test, y_pred_ensemble)
            mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)
            
            model_results[ensemble_name] = {
                'model': ensemble_model,
                'mse': float(mse_ensemble),
                'rmse': float(rmse_ensemble),
                'r2': float(r2_ensemble),
                'mae': float(mae_ensemble),
                'predictions': y_pred_ensemble.tolist()[:50],
                'actuals': y_test.tolist()[:50],
                'component_models': [name for name, _ in top_models]
            }
            
            print(f"  Ensemble (average of top 3) - R²: {r2_ensemble:.4f}, RMSE: {rmse_ensemble:.4f}")
            
            # Update best model if ensemble is better
            if r2_ensemble > best_score:
                best_score = r2_ensemble
                best_model_name = ensemble_name
            
            # # ✅ Save ensemble model
            # try:
            #     metadata = {
            #         'task_type': 'regression',
            #         'target_column': target_column,
            #         'model_name': ensemble_name,
            #         'metrics': {
            #             'r2': float(r2_ensemble),
            #             'rmse': float(rmse_ensemble),
            #             'mse': float(mse_ensemble),
            #             'mae': float(mae_ensemble)
            #         },
            #         'feature_count': len(self.X_columns),
            #         'ensemble': True,
            #         'component_models': [name for name, _ in top_models]
            #     }
                
            #     model_id = f"{main_folder}/regression_{ensemble_name}_{base_name}"
                
            #     saved_id = self.model_manager.save_model(
            #         model_id=model_id,
            #         model=ensemble_model,
            #         scaler=self.scaler,
            #         label_encoders=self.label_encoders,
            #         feature_columns=self.X_columns,
            #         metadata=metadata
            #     )
            #     saved_model_ids.append(saved_id)
            #     print(f"    ✅ Saved ensemble: {model_id}")
                
            # except Exception as e:
            #     print(f"    ⚠️ Could not save ensemble: {e}")
        
        # Store all models
        self.models['regression_models'] = model_results
        self.results['regression_comparison'] = {
            'all_models': {name: {k: v for k, v in data.items() if k != 'model'} 
                        for name, data in model_results.items()},
            'best_model': best_model_name,
            'best_score': best_score,
            'test_size': len(X_test),
            'train_size': len(X_train),
            'saved_model_ids': saved_model_ids,
            'total_saved': len(saved_model_ids),
            'main_folder': main_folder,
            'unique_id': unique_id,
            'base_name': base_name
        }
        
        # ✅ Save the best model
        best_model_data = model_results[best_model_name]
        metadata = {
            'task_type': 'regression',
            'target_column': target_column,
            'best_model_name': best_model_name,
            'metrics': {
                'r2': best_model_data['r2'],
                'rmse': best_model_data['rmse'],
                'mse': best_model_data['mse'],
                'mae': best_model_data['mae']
            },
            'feature_count': len(self.X_columns),
            'ensemble': 'ensemble' in best_model_name,
            'component_models': best_model_data.get('component_models', [])
        }
        
        model_id = f"{main_folder}/regression_best_{base_name}"
        
        model_id = self.model_manager.save_model(
            model_id=model_id,
            model=best_model_data['model'],
            scaler=self.scaler,
            label_encoders=self.label_encoders,
            feature_columns=self.X_columns,
            metadata=metadata
        )
        self.results['regression_comparison']['best_model_id'] = model_id
        
        # MLflow Integration
        if MLFLOW_AVAILABLE:
            try:
                os.makedirs("data/results", exist_ok=True)
                mlflow_run_id = mlflow_manager.log_regression_run(
                    model=best_model_data['model'],
                    model_name=best_model_name,
                    task_info={'target_column': target_column},
                    results=best_model_data,
                    visualizations=self.visualizations,
                    dataset_info={
                        'shape': self.df.shape,
                        'columns': self.df.columns.tolist(),
                        'numeric_columns': self.numeric_cols,
                        'categorical_columns': self.categorical_cols,
                        'total_rows': len(self.df)
                    }
                )
                
                mlflow_manager.log_multi_model_comparison(
                    task_type='regression',
                    comparison_results=self.results['regression_comparison'],
                    best_model=best_model_data['model'],
                    best_model_name=best_model_name,
                    visualizations=self.visualizations,
                    dataset_info={
                        'shape': self.df.shape,
                        'columns': self.df.columns.tolist(),
                        'total_rows': len(self.df)
                    }
                )
                
                print(f"📊 MLflow run logged: {mlflow_run_id}")
                self.results['regression_comparison']['mlflow_run_id'] = mlflow_run_id
                metadata['mlflow_run_id'] = mlflow_run_id
                
            except Exception as mlflow_error:
                print(f"⚠️ MLflow logging failed: {mlflow_error}")
        else:
            print("ℹ️ MLflow not available, skipping experiment tracking")
        
        # Create comparison visualization
        self.create_model_comparison_visualization(model_results, 'regression')
        
        return self.results['regression_comparison']
        
    def train_and_compare_classification_models(self, target_column: str) -> Dict[str, Any]:
        """Train multiple classification models and compare them"""
        print(f"\n🧮 Training multiple classification models for target: {target_column}")
        
        df_processed = self.preprocess_data(target_column)
        
        # Encode target
        if df_processed[target_column].dtype == 'object' or df_processed[target_column].nunique() <= 10:
            le_target = LabelEncoder()
            y = le_target.fit_transform(df_processed[target_column])
            self.label_encoders[target_column] = le_target
            class_names = le_target.classes_.tolist()
        else:
            y = df_processed[target_column]
            class_names = [str(c) for c in np.unique(y)]
        
        X = df_processed.drop(columns=[target_column])
        self.X_columns = X.columns.tolist()
        self.y_column = target_column
        
        # Split with stratification
        test_size = 0.2 if len(df_processed) > 10 else 0.5
        stratify = y if len(np.unique(y)) > 1 and min(np.bincount(y)) >= 2 else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models to train
        classification_models = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            # 'decision_tree_classifier': DecisionTreeClassifier(random_state=42),
            'random_forest_classifier': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting_classifier': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'svc': SVC(kernel='rbf', probability=True, random_state=42),
            'knn_classifier': KNeighborsClassifier(n_neighbors=5)
            # 'naive_bayes': GaussianNB()
        }
        
        # ========== FIX: Define base_name and unique_id HERE (outside any condition) ==========
        if hasattr(self, 'filename') and self.filename:
            # Remove extension and clean filename for folder name
            base_name = os.path.splitext(self.filename)[0]
            # Remove special characters if any
            base_name = base_name.replace(' ', '_').replace('-', '_').replace('.', '_').replace('"', '').replace("'", "")
            unique_id = f"{base_name}"
        else:
            # Fallback to target column if filename not available
            base_name = target_column.replace(' ', '_').replace('"', '').replace("'", "")
            unique_id = f"{base_name}"
        
        # Main folder name based on unique_id
        main_folder = f"classification_{unique_id}"
        
        print(f"📁 Saving models in unique folder: {main_folder}")
        print(f"📁 This ensures no conflicts even with same target column!")
        
        # List of models to save (5 models)
        models_to_save = [
            'logistic_regression',
            'random_forest_classifier', 
            'gradient_boosting_classifier',
            'knn_classifier'
        ]
        
        # Train and evaluate each model
        model_results = {}
        best_score = -np.inf
        best_model_name = None
        saved_model_ids = []  # List to store saved model IDs
        
        for model_name, model in classification_models.items():
            print(f"  Training {model_name}...")
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
                
                model_results[model_name] = {
                    'model': model,
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'cv_mean': float(cv_scores.mean()),
                    'cv_std': float(cv_scores.std()),
                    'predictions': y_pred.tolist()[:50],
                    'actuals': y_test.tolist()[:50]
                }
                
                print(f"    Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
                
                # Update best model (using F1 score as primary metric)
                if f1 > best_score:
                    best_score = f1
                    best_model_name = model_name
                
                # ========== Save ONLY the 5 specified models ==========
                if model_name in models_to_save:
                    try:
                        # Create metadata for this model
                        metadata = {
                            'task_type': 'classification',
                            'target_column': target_column,
                            'model_name': model_name,
                            'metrics': {
                                'accuracy': float(accuracy),
                                'f1_score': float(f1),
                                'precision': float(precision),
                                'recall': float(recall),
                                'cv_mean': float(cv_scores.mean())
                            },
                            'class_names': class_names,
                            'feature_count': len(self.X_columns),
                            'feature_columns': self.X_columns,
                            'dataset_hash': self.dataset_hash,
                            'timestamp': datetime.now().isoformat(),
                            'filename': self.filename if hasattr(self, 'filename') else None,
                            'unique_folder': main_folder
                        }
                        
                        # Create model ID with unique folder structure
                        model_id = f"{main_folder}/classification_{model_name}_{base_name}"
                        
                        # Save the model
                        saved_id = self.model_manager.save_model(
                            model_id=model_id,
                            model=model,
                            scaler=self.scaler,
                            label_encoders=self.label_encoders,
                            feature_columns=self.X_columns,
                            metadata=metadata
                        )
                        
                        saved_model_ids.append(saved_id)
                        print(f"    ✅ Saved: {model_id}")
                        
                    except Exception as save_error:
                        print(f"    ⚠️ Could not save {model_name}: {save_error}")
                
            except Exception as e:
                print(f"    Error training {model_name}: {e}")
                continue
        
        # ========== Create ensemble model (voting classifier) ==========
        if len(model_results) >= 3:
            # Get top 3 models by F1 score
            sorted_models = sorted(model_results.items(), 
                                key=lambda x: x[1]['f1_score'], 
                                reverse=True)[:3]
            
            top_models = [(name, data['model']) for name, data in sorted_models]
            ensemble_name = "ensemble_voting"
            
            # Create VotingClassifier
            from sklearn.ensemble import VotingClassifier
            ensemble_model = VotingClassifier(estimators=top_models, voting='soft')
            ensemble_model.fit(X_train_scaled, y_train)
            y_pred_ensemble = ensemble_model.predict(X_test_scaled)
            
            # Calculate metrics for ensemble
            accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
            precision_ensemble = precision_score(y_test, y_pred_ensemble, average='weighted', zero_division=0)
            recall_ensemble = recall_score(y_test, y_pred_ensemble, average='weighted', zero_division=0)
            f1_ensemble = f1_score(y_test, y_pred_ensemble, average='weighted', zero_division=0)
            
            model_results[ensemble_name] = {
                'model': ensemble_model,
                'accuracy': float(accuracy_ensemble),
                'precision': float(precision_ensemble),
                'recall': float(recall_ensemble),
                'f1_score': float(f1_ensemble),
                'predictions': y_pred_ensemble.tolist()[:50],
                'actuals': y_test.tolist()[:50],
                'component_models': [name for name, _ in top_models]
            }
            
            print(f"  Ensemble (voting of top 3) - Accuracy: {accuracy_ensemble:.4f}, F1: {f1_ensemble:.4f}")
            
            # Update best model if ensemble is better
            if f1_ensemble > best_score:
                best_score = f1_ensemble
                best_model_name = ensemble_name
            
            # ========== Save ensemble model ==========
            # try:
            #     metadata = {
            #         'task_type': 'classification',
            #         'target_column': target_column,
            #         'model_name': ensemble_name,
            #         'metrics': {
            #             'accuracy': float(accuracy_ensemble),
            #             'f1_score': float(f1_ensemble),
            #             'precision': float(precision_ensemble),
            #             'recall': float(recall_ensemble)
            #         },
            #         'class_names': class_names,
            #         'feature_count': len(self.X_columns),
            #         'ensemble': True,
            #         'component_models': [name for name, _ in top_models]
            #     }
                
            #     model_id = f"{main_folder}/classification_{ensemble_name}_{base_name}"
                
            #     saved_id = self.model_manager.save_model(
            #         model_id=model_id,
            #         model=ensemble_model,
            #         scaler=self.scaler,
            #         label_encoders=self.label_encoders,
            #         feature_columns=self.X_columns,
            #         metadata=metadata
            #     )
            #     saved_model_ids.append(saved_id)
            #     print(f"    ✅ Saved ensemble: {model_id}")
                
            # except Exception as e:
            #     print(f"    ⚠️ Could not save ensemble: {e}")
        
        # ========== Store all models ==========
        self.models['classification_models'] = model_results
        self.results['classification_comparison'] = {
            'all_models': {name: {k: v for k, v in data.items() if k != 'model'} 
                        for name, data in model_results.items()},
            'best_model': best_model_name,
            'best_score': best_score,
            'test_size': len(X_test),
            'train_size': len(X_train),
            'class_names': class_names,
            'saved_model_ids': saved_model_ids,
            'total_saved': len(saved_model_ids),
            'main_folder': main_folder,
            'unique_id': unique_id,
            'base_name': base_name
        }
        
        # ========== Save the best model ==========
        best_model_data = model_results[best_model_name]
        metadata = {
            'task_type': 'classification',
            'target_column': target_column,
            'best_model_name': best_model_name,
            'metrics': {
                'accuracy': best_model_data['accuracy'],
                'f1_score': best_model_data['f1_score'],
                'precision': best_model_data['precision'],
                'recall': best_model_data['recall']
            },
            'class_names': class_names,
            'feature_count': len(self.X_columns),
            'ensemble': 'ensemble' in best_model_name,
            'component_models': best_model_data.get('component_models', [])
        }
        
        model_id = f"{main_folder}/classification_best_{base_name}"
        
        model_id = self.model_manager.save_model(
            model_id=model_id,
            model=best_model_data['model'],
            scaler=self.scaler,
            label_encoders=self.label_encoders,
            feature_columns=self.X_columns,
            metadata=metadata
        )
        self.results['classification_comparison']['best_model_id'] = model_id
        
        # ========== MLflow Integration ==========
        try:
            from mlflow_integration import mlflow_manager
            
            # Log classification run to MLflow
            mlflow_run_id = mlflow_manager.log_classification_run(
                model=best_model_data['model'],
                model_name=best_model_name,
                task_info={'target_column': target_column},
                results=best_model_data,
                visualizations=self.visualizations,
                dataset_info={
                    'shape': self.df.shape,
                    'columns': self.df.columns.tolist(),
                    'numeric_columns': self.numeric_cols,
                    'categorical_columns': self.categorical_cols,
                    'total_rows': len(self.df)
                }
            )
            
            # Also log the full comparison
            mlflow_manager.log_multi_model_comparison(
                task_type='classification',
                comparison_results=self.results['classification_comparison'],
                best_model=best_model_data['model'],
                best_model_name=best_model_name,
                visualizations=self.visualizations,
                dataset_info={
                    'shape': self.df.shape,
                    'columns': self.df.columns.tolist(),
                    'total_rows': len(self.df)
                }
            )
            
            print(f"📊 MLflow run logged: {mlflow_run_id}")
            
            # Store MLflow run ID in results
            self.results['classification_comparison']['mlflow_run_id'] = mlflow_run_id
            metadata['mlflow_run_id'] = mlflow_run_id
            
        except ImportError as e:
            print(f"⚠️ MLflow not available: {e}")
        except Exception as mlflow_error:
            print(f"⚠️ MLflow logging failed: {mlflow_error}")
        
        # ========== Create comparison visualization ==========
        self.create_model_comparison_visualization(model_results, 'classification')
        
        return self.results['classification_comparison']



    def create_model_comparison_visualization(self, model_results: Dict, task_type: str):
        """Create visualization comparing all trained models"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Extract model names and metrics
            model_names = list(model_results.keys())
            
            if task_type == 'regression':
                # R² scores
                r2_scores = [model_results[name]['r2'] for name in model_names]
                
                axes[0, 0].barh(range(len(model_names)), r2_scores, color='skyblue')
                axes[0, 0].set_yticks(range(len(model_names)))
                axes[0, 0].set_yticklabels(model_names)
                axes[0, 0].set_xlabel('R² Score')
                axes[0, 0].set_title('Model Comparison - R² Scores')
                axes[0, 0].set_xlim([0, 1])
                
                # RMSE scores
                rmse_scores = [model_results[name]['rmse'] for name in model_names]
                
                axes[0, 1].barh(range(len(model_names)), rmse_scores, color='lightcoral')
                axes[0, 1].set_yticks(range(len(model_names)))
                axes[0, 1].set_yticklabels(model_names)
                axes[0, 1].set_xlabel('RMSE')
                axes[0, 1].set_title('Model Comparison - RMSE')
                
                # Find best model
                best_idx = np.argmax(r2_scores)
                
            else:  # classification
                # Accuracy scores
                accuracy_scores = [model_results[name]['accuracy'] for name in model_names]
                
                axes[0, 0].barh(range(len(model_names)), accuracy_scores, color='lightgreen')
                axes[0, 0].set_yticks(range(len(model_names)))
                axes[0, 0].set_yticklabels(model_names)
                axes[0, 0].set_xlabel('Accuracy')
                axes[0, 0].set_title('Model Comparison - Accuracy')
                axes[0, 0].set_xlim([0, 1])
                
                # F1 scores
                f1_scores = [model_results[name]['f1_score'] for name in model_names]
                
                axes[0, 1].barh(range(len(model_names)), f1_scores, color='gold')
                axes[0, 1].set_yticks(range(len(model_names)))
                axes[0, 1].set_yticklabels(model_names)
                axes[0, 1].set_xlabel('F1 Score')
                axes[0, 1].set_title('Model Comparison - F1 Score')
                axes[0, 1].set_xlim([0, 1])
                
                # Find best model
                best_idx = np.argmax(f1_scores)
            
            # Highlight best model
            for ax in [axes[0, 0], axes[0, 1]]:
                bars = ax.patches
                bars[best_idx].set_color('red')
                bars[best_idx].set_alpha(0.8)
            
            # Metric comparison table
            axes[1, 0].axis('off')
            
            if task_type == 'regression':
                table_data = []
                for name in model_names:
                    table_data.append([
                        name,
                        f"{model_results[name]['r2']:.4f}",
                        f"{model_results[name]['rmse']:.4f}",
                        f"{model_results[name]['mae']:.4f}"
                    ])
                
                table = axes[1, 0].table(
                    cellText=table_data,
                    colLabels=['Model', 'R²', 'RMSE', 'MAE'],
                    cellLoc='center',
                    loc='center'
                )
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 1.5)
                
            else:  # classification
                table_data = []
                for name in model_names:
                    table_data.append([
                        name,
                        f"{model_results[name]['accuracy']:.4f}",
                        f"{model_results[name]['f1_score']:.4f}",
                        f"{model_results[name]['precision']:.4f}",
                        f"{model_results[name]['recall']:.4f}"
                    ])
                
                table = axes[1, 0].table(
                    cellText=table_data,
                    colLabels=['Model', 'Accuracy', 'F1', 'Precision', 'Recall'],
                    cellLoc='center',
                    loc='center'
                )
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 1.5)
            
            # Performance summary
            axes[1, 1].axis('off')
            best_model_name = model_names[best_idx]
            if task_type == 'regression':
                summary_text = f"Best Model: {best_model_name}\n\n"
                summary_text += f"R²: {model_results[best_model_name]['r2']:.4f}\n"
                summary_text += f"RMSE: {model_results[best_model_name]['rmse']:.4f}\n"
                summary_text += f"MAE: {model_results[best_model_name]['mae']:.4f}\n\n"
                summary_text += f"Total Models: {len(model_names)}"
            else:
                summary_text = f"Best Model: {best_model_name}\n\n"
                summary_text += f"Accuracy: {model_results[best_model_name]['accuracy']:.4f}\n"
                summary_text += f"F1 Score: {model_results[best_model_name]['f1_score']:.4f}\n"
                summary_text += f"Precision: {model_results[best_model_name]['precision']:.4f}\n"
                summary_text += f"Recall: {model_results[best_model_name]['recall']:.4f}\n\n"
                summary_text += f"Total Models: {len(model_names)}"
            
            axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes,
                          fontsize=11, verticalalignment='center',
                          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
            
            plt.suptitle(f'Multi-Model Comparison - {task_type.title()}', fontsize=16, y=1.02)
            plt.tight_layout()
            
            # Convert to base64
            plot_base64 = self._plot_to_base64(fig, is_matplotlib=True)
            if plot_base64:
                self.visualizations.append({
                    'type': 'image',
                    'name': f'model_comparison_{task_type}',
                    'title': f'Model Comparison - {task_type.title()}',
                    'content': plot_base64,
                    'description': f'Comparison of all {task_type} models with performance metrics'
                })
            
            plt.close(fig)
            
        except Exception as e:
            print(f"Error creating model comparison visualization: {e}")
    
    def load_saved_model(self, task_type: str, target_column: str = None):
        """Load a previously saved model - ONLY if same dataset"""
        
        # Get current filename if available
        current_filename = getattr(self, 'filename', None)
        
        # List all saved models
        all_models = self.model_manager.list_all_models()
        
        matching_models = []
        for model in all_models:
            model_meta = model.get('metadata', {})
            model_filename = model_meta.get('filename')
            
            # Check conditions
            if model_meta.get('task_type') != task_type:
                continue
                
            if target_column and model_meta.get('target_column') != target_column:
                continue
                
            # CRITICAL: Only load if same dataset
            # First try: match by filename
            if current_filename and model_filename:
                if os.path.basename(model_filename) == os.path.basename(current_filename):
                    matching_models.append(model)
            # Second try: match by dataset hash
            elif model_meta.get('dataset_hash') == self.dataset_hash:
                matching_models.append(model)
        
        if matching_models:
            # Get most recent model
            latest_model = max(matching_models, 
                            key=lambda x: x.get('metadata', {}).get('timestamp', ''))
            
            model_id = latest_model['model_id']
            print(f"📂 Loading saved model: {model_id}")
            
            loaded_data = self.model_manager.load_model(model_id)
            
            self.models['loaded_model'] = loaded_data['model']
            self.scaler = loaded_data.get('scaler')
            self.label_encoders = loaded_data.get('label_encoders', {})
            
            # For clustering, use numeric columns; for others, use feature_columns
            if task_type == 'clustering':
                self.X_columns = self.numeric_cols
            else:
                self.X_columns = loaded_data.get('feature_columns')
            
            return {
                'success': True,
                'model_id': model_id,
                'metadata': loaded_data.get('metadata', {}),
                'message': 'Model loaded from storage'
            }
        else:
            return {
                'success': False,
                'message': 'No saved model found for this dataset, need to train new model'
            }


    def predict_with_saved_model(self, new_data: pd.DataFrame, task_type: str, 
                                target_column: str = None) -> Dict[str, Any]:
        """Make predictions using a saved model"""
        # Check if model exists
        load_result = self.load_saved_model(task_type, target_column)
        
        if not load_result['success']:
            return {
                'success': False,
                'error': 'No saved model found',
                'message': 'Please train a model first'
            }
        
        try:
            model = self.models['loaded_model']
            metadata = load_result['metadata']
            
            # Prepare new data
            if self.X_columns:
                # Ensure all required columns are present
                missing_cols = set(self.X_columns) - set(new_data.columns)
                if missing_cols:
                    # Add missing columns with default values
                    for col in missing_cols:
                        new_data[col] = 0
                
                X_new = new_data[self.X_columns]
            else:
                X_new = new_data
            
            # Apply preprocessing
            if self.scaler:
                X_new_scaled = self.scaler.transform(X_new)
            else:
                X_new_scaled = X_new
            
            # Make predictions
            predictions = model.predict(X_new_scaled)
            
            # Decode predictions if classification
            if task_type == 'classification' and self.label_encoders.get(target_column):
                le = self.label_encoders[target_column]
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X_new_scaled)
                    result = {
                        'predictions': le.inverse_transform(predictions).tolist(),
                        'probabilities': probabilities.tolist(),
                        'class_names': le.classes_.tolist()
                    }
                else:
                    result = {
                        'predictions': le.inverse_transform(predictions).tolist(),
                        'class_names': le.classes_.tolist()
                    }
            else:
                result = {
                    'predictions': predictions.tolist()
                }
            
            result.update({
                'success': True,
                'model_id': load_result['model_id'],
                'model_type': metadata.get('best_model_name', 'Unknown'),
                'task_type': task_type,
                'prediction_count': len(predictions),
                'used_saved_model': True
            })
            
            return result
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Error making predictions with saved model'
            }

    def detect_column_types(self):
        """Detect numeric and categorical columns"""
        self.numeric_cols = []
        self.categorical_cols = []
        
        for col in self.df.columns:
            try:
                pd.to_numeric(self.df[col])
                self.numeric_cols.append(col)
            except:
                self.categorical_cols.append(col)
        
        # Move low-cardinality numeric columns to categorical
        for col in self.numeric_cols.copy():
            if self.df[col].nunique() <= 10 and self.df[col].nunique() < len(self.df) * 0.3:
                self.numeric_cols.remove(col)
                self.categorical_cols.append(col)
    
    def detect_task_type(self, prompt: str) -> Dict[str, Any]:
        """AI-powered task detection - NO FALLBACK"""
        print(f"   Prompt: '{prompt}'")
        
        if not self.azure_client:
            raise Exception("OpenAI client not available. Cannot perform AI task detection.")
        
        try:
            # Pehle column types analyze karo
            categorical_cols = self.categorical_cols.copy()
            numeric_cols = self.numeric_cols.copy()
            
            # Low cardinality numeric columns ko categorical treat karo
            for col in numeric_cols.copy():
                if self.df[col].nunique() <= 10:
                    categorical_cols.append(col)
                    numeric_cols.remove(col)
            
            messages = [
                {
                    "role": "system", 
                    "content": f"""You are a data science task detector. Analyze the user prompt and dataset columns.

    Dataset Info:
    - All columns: {list(self.df.columns)}
    - Numeric columns (suitable for regression): {numeric_cols}
    - Categorical columns (suitable for classification): {categorical_cols}

    Guidelines:
    1. For classification, target must have <= 20 unique values
    2. For regression, target must be numeric with > 20 unique values
    3. If user says "classify" but target has many unique values, suggest regression instead

    Return JSON format with:
    - task_type (regression/classification/clustering/visualization/exploratory/feature_selection)
    - target_column (if applicable)
    - explanation
    - warning (if target might be unsuitable)"""
                    },
                    {
                        "role": "user", 
                        "content": f"User prompt: {prompt}"
                    }
                ]
                
            response = self.azure_client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4"),
                    messages=messages,
                    response_format={"type": "json_object"}
                )
                
            result = json.loads(response.choices[0].message.content)
                
                # Validate classification target
            if result.get('task_type') == 'classification' and result.get('target_column'):
                    target = result['target_column']
                    if target in self.df.columns:
                        unique_count = self.df[target].nunique()
                        if unique_count > 20:
                            result['warning'] = f"Target '{target}' has {unique_count} unique values. Consider regression instead."
                            # Auto-correct to regression if too many classes
                            if unique_count > 50:
                                result['task_type'] = 'regression'
                                result['explanation'] += f" (auto-changed from classification to regression due to {unique_count} classes)"
                
            return result
                
        except Exception as e:
                raise Exception(f"AI task detection failed: {str(e)}")
            
    def preprocess_data(self, target_column: Optional[str] = None):
        """Enhanced preprocessing with feature engineering"""
        if target_column:
            target_column = target_column.strip('"').strip("'").strip()

        df_processed = self.df.copy()

        df_processed.columns = [str(col).strip('"').strip("'").strip() for col in df_processed.columns]
        
        # Handle missing values with advanced strategies
        for col in self.numeric_cols:
            if df_processed[col].isnull().any():
                # Use different strategies based on data distribution
                if df_processed[col].skew() > 1:
                    df_processed[col].fillna(df_processed[col].median(), inplace=True)
                else:
                    df_processed[col].fillna(df_processed[col].mean(), inplace=True)
        
        for col in self.categorical_cols:
            if df_processed[col].isnull().any():
                mode_val = df_processed[col].mode()
                if not mode_val.empty:
                    df_processed[col].fillna(mode_val[0], inplace=True)
                else:
                    df_processed[col].fillna("Unknown", inplace=True)
        
        # Encode categorical variables
        self.label_encoders = {}
        for col in self.categorical_cols:
            if col != target_column:
                try:
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                    self.label_encoders[col] = le
                except:
                    unique_vals = df_processed[col].unique()
                    mapping = {val: idx for idx, val in enumerate(unique_vals)}
                    df_processed[col] = df_processed[col].map(mapping)
        
        # Feature engineering: add polynomial features for numeric columns
        if len(self.numeric_cols) > 1 and target_column not in self.numeric_cols:
            # Create interaction terms for top correlated features
            corr_matrix = df_processed[self.numeric_cols].corr()
            for i in range(len(self.numeric_cols)):
                for j in range(i+1, len(self.numeric_cols)):
                    col1, col2 = self.numeric_cols[i], self.numeric_cols[j]
                    if abs(corr_matrix.loc[col1, col2]) > 0.5:
                        df_processed[f'{col1}_x_{col2}'] = df_processed[col1] * df_processed[col2]
        
        return df_processed
    
    def perform_regression(self, target_column: str, hyper_tuning: bool = False, 
                      ensemble_method: Optional[str] = None,
                      model_type: Optional[str] = None) -> Dict[str, Any]:
        """Enhanced regression with multiple models and persistence"""
        try:
            print(f"\n🧮 Performing ENHANCED REGRESSION with target: {target_column}")
            print(f"  Data shape: {self.df.shape}")
            
            # ✅ FIX: Check if we have a saved model for THIS dataset
            saved_model_check = None
            try:
                # Get current filename
                current_filename = getattr(self, 'filename', None)
                
                # List all models and find matching ones
                all_models = self.model_manager.list_all_models()
                
                matching_models = []
                for model in all_models:
                    model_meta = model.get('metadata', {})
                    
                    # Check if same task and target
                    if model_meta.get('task_type') != 'regression':
                        continue
                    if model_meta.get('target_column') != target_column:
                        continue
                    
                    # ✅ CRITICAL: Check if same dataset (using hash)
                    if model_meta.get('dataset_hash') == self.dataset_hash:
                        matching_models.append(model)
                
                if matching_models:
                    # Get latest model
                    latest_model = max(matching_models, 
                                    key=lambda x: x.get('metadata', {}).get('timestamp', ''))
                    
                    model_id = latest_model['model_id']
                    print(f"📂 Loading saved model: {model_id}")
                    
                    loaded_data = self.model_manager.load_model(model_id)
                    
                    self.models['loaded_model'] = loaded_data['model']
                    self.scaler = loaded_data.get('scaler')
                    self.label_encoders = loaded_data.get('label_encoders', {})
                    self.X_columns = loaded_data.get('feature_columns')
                    
                    saved_model_check = {
                        'success': True,
                        'model_id': model_id,
                        'metadata': loaded_data.get('metadata', {})
                    }
                else:
                    saved_model_check = {'success': False}
                    
            except Exception as e:
                print(f"  ⚠️ Could not check saved models: {e}")
                saved_model_check = {'success': False}
            
            # If saved model found for THIS dataset, use it
            if saved_model_check and saved_model_check.get('success'):
                print(f"  ✅ Using saved model for this dataset")
                
                # Create a test/train split for evaluation
                df_processed = self.preprocess_data(target_column)
                X = df_processed.drop(columns=[target_column])
                y = df_processed[target_column]
                
                test_size = 0.2 if len(df_processed) > 10 else 0.5
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                if self.scaler:
                    X_test_scaled = self.scaler.transform(X_test)
                else:
                    X_test_scaled = X_test
                
                # Evaluate saved model
                model = self.models['loaded_model']
                y_pred = model.predict(X_test_scaled)
                
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                self.results['regression'] = {
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'r2': float(r2),
                    'mae': float(mae),
                    'model_type': saved_model_check['metadata'].get('best_model_name', 'Loaded Model'),
                    'test_size': len(X_test),
                    'train_size': len(X_train),
                    'predictions': y_pred.tolist()[:50],
                    'actuals': y_test.tolist()[:50],
                    'using_saved_model': True,
                    'model_id': saved_model_check['model_id']
                }
                
                # You need to implement this method or comment it out
                # self.create_enhanced_regression_visualizations(
                #     y_test, y_pred, X, target_column, model
                # )
                
                return self.results['regression']
            
            # No saved model found, train new models
            print("  No saved model found for this dataset, training new models...")
            return self.train_and_compare_regression_models(target_column)
            
        except Exception as e:
            print(f"❌ Regression error: {e}")
            traceback.print_exc()
            raise
        

    def perform_classification(
        self,
        target_column: str,
        hyper_tuning: bool = False,
        ensemble_method: Optional[str] = None,
        model_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enhanced classification with automatic binning for numeric targets.
        This guarantees classification even for continuous data.
        """
        try:
            print(f"\n🧮 Performing ENHANCED CLASSIFICATION with target: {target_column}")

            # ======================================================
            # ✅ FORCE CLASSIFICATION: auto-bin numeric targets
            # ======================================================
            y_raw = self.df[target_column]

            if y_raw.dtype != "object" and y_raw.nunique() > 10:
                print("⚠️ Numeric target detected → converting to classification bins")

                try:
                    self.df[target_column] = pd.qcut(
                        y_raw,
                        q=3,  # number of classes (change if needed)
                        labels=["low", "medium", "high"],
                        duplicates="drop"
                    )
                except Exception:
                    # fallback if qcut fails
                    self.df[target_column] = pd.cut(
                        y_raw,
                        bins=3,
                        labels=["low", "medium", "high"]
                    )

            # ======================================================
            # 🔍 Check for saved classification model
            # ======================================================
            saved_model_check = self.load_saved_model("classification", target_column)

            # ======================================================
            #  USE SAVED MODEL IF AVAILABLE
            # ======================================================
            if saved_model_check["success"]:
                print(f"  ✅] Using saved model: {saved_model_check.get('model_id')}")

                df_processed = self.preprocess_data(target_column)

                # Encode target
                le_target = LabelEncoder()
                y = le_target.fit_transform(df_processed[target_column])
                class_names = le_target.classes_.tolist()

                X = df_processed.drop(columns=[target_column])

                test_size = 0.2 if len(df_processed) > 10 else 0.5

                # Safe stratification
                stratify = (
                    y
                    if len(np.unique(y)) > 1 and min(np.bincount(y)) >= 2
                    else None
                )

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=42,
                    stratify=stratify
                )

                if self.scaler:
                    X_test_scaled = self.scaler.transform(X_test)
                else:
                    X_test_scaled = X_test

                model = self.models["loaded_model"]
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = (
                    model.predict_proba(X_test_scaled)
                    if hasattr(model, "predict_proba")
                    else None
                )

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(
                    y_test, y_pred, average="weighted", zero_division=0
                )
                recall = recall_score(
                    y_test, y_pred, average="weighted", zero_division=0
                )
                f1 = f1_score(
                    y_test, y_pred, average="weighted", zero_division=0
                )

                self.results["classification"] = {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                    "model_type": saved_model_check["metadata"].get(
                        "best_model_name", "Loaded Model"
                    ),
                    "class_names": class_names,
                    "test_size": len(X_test),
                    "predictions": y_pred.tolist()[:50],
                    "actuals": y_test.tolist()[:50],
                    "using_saved_model": True,
                    "model_id": saved_model_check["model_id"],
                }

                self.create_enhanced_classification_visualizations(
                    y_test, y_pred, y_pred_proba, target_column, class_names
                )

                return self.results["classification"]

            # ======================================================
            # 🚀 TRAIN NEW CLASSIFICATION MODELS
            # ======================================================
            print("  No saved model found, training multiple models...")

            results = self.train_and_compare_classification_models(target_column)

            if results is None:
                raise ValueError(
                    "All classification models failed even after binning. "
                    "Try reducing number of bins or check data quality."
                )

            return results

        except Exception as e:
            print(f"❌ Classification error: {e}")
            traceback.print_exc()
            raise
            
    def train_and_compare_clustering_models(self, n_clusters: Optional[int] = None) -> Dict[str, Any]:
        """Train multiple clustering models and compare them (like regression/classification)"""
        print(f"\n🧮 Training multiple clustering models for comparison")
        
        df_processed = self.preprocess_data()
        
        if not self.numeric_cols:
            raise ValueError("No numeric columns available for clustering")
        
        X = df_processed[self.numeric_cols]
        self.X_columns = self.numeric_cols
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Determine optimal number of clusters if not specified
        if n_clusters is None:
            n_clusters = self._find_optimal_clusters(X_scaled)
            print(f"  Auto-detected optimal clusters: {n_clusters}")
        
        # ========== FIX: Define base_name and unique_id HERE ==========
        if hasattr(self, 'filename') and self.filename:
            # Remove extension and clean filename for folder name
            base_name = os.path.splitext(self.filename)[0]
            # Remove special characters if any
            base_name = base_name.replace(' ', '_').replace('-', '_').replace('.', '_').replace('"', '').replace("'", "")
            unique_id = f"{base_name}"
        else:
            # Fallback to clusters count if filename not available
            base_name = f"clustering_{n_clusters}clusters"
            unique_id = f"clustering_{n_clusters}clusters"
        
        # Main folder name based on unique_id
        main_folder = f"clustering_{unique_id}"
        
        print(f"📁 Saving models in unique folder: {main_folder}")
        print(f"📁 This ensures no conflicts even with same clustering configuration!")
        
        # Define clustering algorithms to compare (4 algorithms)
        clustering_algorithms = {
            'kmeans': KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
            'agglomerative': AgglomerativeClustering(n_clusters=n_clusters),
            # 'dbscan': DBSCAN(eps=0.5, min_samples=5),
            'gaussian_mixture': GaussianMixture(n_components=n_clusters, random_state=42)
        }
        
        # List of algorithms to save (all 4)
        algorithms_to_save = ['kmeans', 'agglomerative', 'dbscan', 'gaussian_mixture']
        
        # Train and evaluate each algorithm
        model_results = {}
        best_score = -np.inf
        best_model_name = None
        best_model = None
        best_clusters = None
        saved_model_ids = []  # List to store saved model IDs
        
        for model_name, model in clustering_algorithms.items():
            print(f"  Training {model_name}...")
            try:
                # Fit and predict clusters
                if hasattr(model, 'fit_predict'):
                    clusters = model.fit_predict(X_scaled)
                elif hasattr(model, 'predict'):
                    clusters = model.fit(X_scaled).predict(X_scaled)
                else:
                    clusters = model.fit(X_scaled).labels_
                
                # Calculate evaluation metrics
                unique_clusters = np.unique(clusters)
                n_unique = len(unique_clusters)
                
                evaluation = {}
                if n_unique > 1:
                    # Silhouette score (higher is better, range -1 to 1)
                    try:
                        sil_score = silhouette_score(X_scaled, clusters)
                        evaluation['silhouette'] = float(sil_score)
                    except:
                        evaluation['silhouette'] = -1
                    
                    # Davies-Bouldin score (lower is better)
                    try:
                        db_score = davies_bouldin_score(X_scaled, clusters)
                        evaluation['davies_bouldin'] = float(db_score)
                    except:
                        evaluation['davies_bouldin'] = float('inf')
                    
                    # Calinski-Harabasz score (higher is better)
                    try:
                        ch_score = calinski_harabasz_score(X_scaled, clusters)
                        evaluation['calinski_harabasz'] = float(ch_score)
                    except:
                        evaluation['calinski_harabasz'] = 0
                
                model_results[model_name] = {
                    'model': model,
                    'clusters': clusters.tolist(),
                    'n_clusters_actual': int(n_unique),
                    'evaluation': evaluation,
                    'cluster_sizes': pd.Series(clusters).value_counts().sort_index().to_dict()
                }
                
                # Use silhouette score as primary metric for comparison
                current_score = evaluation.get('silhouette', -1)
                print(f"    Silhouette: {current_score:.4f}, Actual clusters: {n_unique}")
                
                # Update best model
                if current_score > best_score:
                    best_score = current_score
                    best_model_name = model_name
                    best_model = model
                    best_clusters = clusters
                
                # ========== Save ALL algorithms ==========
                if model_name in algorithms_to_save:
                    try:
                        # Create metadata for this model
                        metadata = {
                            'task_type': 'clustering',
                            'algorithm': model_name,
                            'n_clusters_target': n_clusters,
                            'n_clusters_actual': int(n_unique),
                            'metrics': evaluation,
                            'feature_count': len(self.numeric_cols),
                            'numeric_columns': self.numeric_cols,
                            'dataset_hash': self.dataset_hash,
                            'timestamp': datetime.now().isoformat(),
                            'filename': self.filename if hasattr(self, 'filename') else None,
                            'unique_folder': main_folder
                        }
                        
                        # Create model ID with unique folder structure
                        model_id = f"{main_folder}/clustering_{model_name}_{base_name}"
                        
                        # Save the model
                        saved_id = self.model_manager.save_model(
                            model_id=model_id,
                            model=model,
                            scaler=self.scaler,
                            label_encoders=self.label_encoders,
                            feature_columns=self.numeric_cols,
                            metadata=metadata
                        )
                        
                        saved_model_ids.append(saved_id)
                        print(f"    ✅ Saved: {model_id}")
                        
                    except Exception as save_error:
                        print(f"    ⚠️ Could not save {model_name}: {save_error}")
                    
            except Exception as e:
                print(f"    Error training {model_name}: {e}")
                continue
        
        # Store all results
        self.models['clustering_models'] = model_results
        self.results['clustering_comparison'] = {
            'all_models': {name: {k: v for k, v in data.items() if k != 'model'} 
                        for name, data in model_results.items()},
            'best_model': best_model_name,
            'best_score': float(best_score),
            'n_clusters_target': n_clusters,
            'total_samples': len(X_scaled),
            'saved_model_ids': saved_model_ids,
            'total_saved': len(saved_model_ids),
            'main_folder': main_folder,
            'unique_id': unique_id,
            'base_name': base_name
        }
        
        # ========== Save the best model ==========
        if best_model_name and best_model:
            best_model_data = model_results[best_model_name]
            metadata = {
                'task_type': 'clustering',
                'algorithm': best_model_name,
                'n_clusters_target': n_clusters,
                'n_clusters_actual': best_model_data['n_clusters_actual'],
                'metrics': best_model_data['evaluation'],
                'feature_count': len(self.numeric_cols),
                'numeric_columns': self.numeric_cols,
                'is_best_model': True
            }
            
            model_id = f"{main_folder}/clustering_best_{base_name}"
            
            model_id = self.model_manager.save_model(
                model_id=model_id,
                model=best_model,
                scaler=self.scaler,
                label_encoders=self.label_encoders,
                feature_columns=self.numeric_cols,
                metadata=metadata
            )
            self.results['clustering_comparison']['best_model_id'] = model_id
        
        # ========== MLflow Integration (if available) ==========
        if MLFLOW_AVAILABLE:
            try:
                os.makedirs("data/results", exist_ok=True)
                # Log clustering run to MLflow
                mlflow_run_id = mlflow_manager.log_clustering_run(
                    model=best_model,  # 'best_model' ऐवजी 'model' वापरा
                    model_name=best_model_name,
                    n_clusters=n_clusters,
                    results=self.results['clustering_comparison'],
                    visualizations=self.visualizations,
                    dataset_info={
                        'shape': self.df.shape,
                        'columns': self.df.columns.tolist(),
                        'numeric_columns': self.numeric_cols,
                        'categorical_columns': self.categorical_cols,
                        'total_rows': len(self.df)
                    }
                )
                
                # Log multi-model comparison
                mlflow_manager.log_multi_model_comparison(
                    task_type='clustering',
                    comparison_results=self.results['clustering_comparison'],
                    best_model=best_model,
                    best_model_name=best_model_name,
                    visualizations=self.visualizations,
                    dataset_info={
                        'shape': self.df.shape,
                        'columns': self.df.columns.tolist(),
                        'total_rows': len(self.df)
                    }
                )
                
                print(f"📊 MLflow run logged: {mlflow_run_id}")
                self.results['clustering_comparison']['mlflow_run_id'] = mlflow_run_id
                
            except Exception as mlflow_error:
                print(f"⚠️ MLflow logging failed: {mlflow_error}")
        else:
            print("ℹ️ MLflow not available, skipping experiment tracking")
        
        # Create comparison visualization
        self._create_clustering_comparison_visualization(model_results, n_clusters)
        
        return self.results['clustering_comparison']

    def _create_clustering_comparison_visualization(self, model_results: Dict, n_clusters: int):
        """Create visualization comparing all clustering algorithms"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Extract algorithm names and metrics
            algo_names = list(model_results.keys())
            
            # Panel 1: Silhouette Scores
            sil_scores = [model_results[name]['evaluation'].get('silhouette', -1) for name in algo_names]
            colors = ['#2E86AB' if s == max(sil_scores) else '#A23B72' for s in sil_scores]
            
            axes[0, 0].barh(range(len(algo_names)), sil_scores, color=colors)
            axes[0, 0].set_yticks(range(len(algo_names)))
            axes[0, 0].set_yticklabels(algo_names)
            axes[0, 0].set_xlabel('Silhouette Score')
            axes[0, 0].set_title('Clustering Algorithms Comparison - Silhouette Score')
            axes[0, 0].set_xlim([-1, 1])
            
            # Panel 2: Actual vs Target Clusters
            actual_clusters = [model_results[name]['n_clusters_actual'] for name in algo_names]
            x = range(len(algo_names))
            width = 0.35
            
            axes[0, 1].bar(x, [n_clusters] * len(algo_names), width, label='Target', color='lightgray')
            axes[0, 1].bar([i + width for i in x], actual_clusters, width, label='Actual', color='#F18F01')
            axes[0, 1].set_xticks([i + width/2 for i in x])
            axes[0, 1].set_xticklabels(algo_names, rotation=45, ha='right')
            axes[0, 1].set_ylabel('Number of Clusters')
            axes[0, 1].set_title('Target vs Actual Clusters')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3, axis='y')
            
            # Panel 3: Davies-Bouldin Scores (lower is better)
            db_scores = [model_results[name]['evaluation'].get('davies_bouldin', float('inf')) for name in algo_names]
            db_scores_norm = [min(1, 1/(1+s)) if s != float('inf') else 0 for s in db_scores]
            
            axes[1, 0].barh(range(len(algo_names)), db_scores_norm, color='lightcoral')
            axes[1, 0].set_yticks(range(len(algo_names)))
            axes[1, 0].set_yticklabels(algo_names)
            axes[1, 0].set_xlabel('Davies-Bouldin Score (normalized)')
            axes[1, 0].set_title('Davies-Bouldin Score (lower is better)')
            axes[1, 0].set_xlim([0, 1])
            
            # Panel 4: Summary Table
            axes[1, 1].axis('off')
            
            table_data = []
            for name in algo_names:
                data = model_results[name]
                table_data.append([
                    name,
                    f"{data['n_clusters_actual']}",
                    f"{data['evaluation'].get('silhouette', -1):.3f}",
                    f"{data['evaluation'].get('davies_bouldin', float('inf')):.2f}",
                    f"{data['evaluation'].get('calinski_harabasz', 0):.0f}"
                ])
            
            table = axes[1, 1].table(
                cellText=table_data,
                colLabels=['Algorithm', 'Clusters', 'Silhouette', 'D-B', 'C-H'],
                cellLoc='center',
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            
            # Highlight best row
            best_idx = list(model_results.keys()).index(self.results['clustering_comparison']['best_model'])
            for j in range(5):
                table[(best_idx+1, j)].set_facecolor('#90EE90')
            
            plt.suptitle(f'Clustering Algorithms Comparison (Target: {n_clusters} clusters)', 
                        fontsize=16, y=1.02)
            plt.tight_layout()
            
            # Convert to base64
            plot_base64 = self._plot_to_base64(fig, is_matplotlib=True)
            if plot_base64:
                self.visualizations.append({
                    'type': 'image',
                    'name': 'clustering_comparison',
                    'title': f'Clustering Comparison (k={n_clusters})',
                    'content': plot_base64,
                    'description': f'Comparison of all clustering algorithms with {n_clusters} target clusters'
                })
            
            plt.close(fig)
            
        except Exception as e:
            print(f"Error creating clustering comparison visualization: {e}")
        
    def perform_clustering(self, method: Optional[str] = None, n_clusters: Optional[int] = None) -> Dict[str, Any]:
        """Enhanced clustering with multiple algorithms and persistence"""
        try:
            print("\n🧮 Performing ENHANCED CLUSTERING")
            
            # Check if we have numeric columns
            if not self.numeric_cols:
                raise ValueError("No numeric columns available for clustering")
            
            print(f"  Using numeric columns: {self.numeric_cols[:5]}...")
            
            # Check if we have a saved model for THIS dataset
            saved_model_check = None
            try:
                saved_model_check = self.load_saved_model('clustering')
                if saved_model_check and saved_model_check.get('success'):
                    print(f"  ✅ Using saved clustering model: {saved_model_check.get('model_id')}")
                    
                    df_processed = self.preprocess_data()
                    X = df_processed[self.numeric_cols]
                    
                    # Scale features using saved scaler
                    if self.scaler:
                        X_scaled = self.scaler.transform(X)
                    else:
                        self.scaler = StandardScaler()
                        X_scaled = self.scaler.fit_transform(X)
                    
                    # Load saved model
                    model = self.models.get('loaded_model')
                    if not model:
                        raise ValueError("No model loaded from saved data")
                    
                    # Predict clusters
                    if hasattr(model, 'predict'):
                        clusters = model.predict(X_scaled)
                    elif hasattr(model, 'fit_predict'):
                        clusters = model.fit_predict(X_scaled)
                    else:
                        clusters = model.labels_
                    
                    # Calculate evaluation metrics
                    evaluation = self._evaluate_clustering(X_scaled, clusters, len(np.unique(clusters)))
                    
                    self.results['clustering'] = {
                        'n_clusters': int(len(np.unique(clusters))),
                        'method': saved_model_check.get('metadata', {}).get('algorithm', method or 'kmeans'),
                        'clusters': clusters.tolist(),
                        'cluster_sizes': pd.Series(clusters).value_counts().sort_index().to_dict(),
                        'evaluation': evaluation,
                        'using_saved_model': True,
                        'model_id': saved_model_check.get('model_id')
                    }
                    
                    # Create visualization
                    self._create_clustering_visualization(X_scaled, clusters, method or 'kmeans', evaluation)
                    
                    return self.results['clustering']
                else:
                    print("  No saved clustering model found, training new model...")
            except Exception as e:
                print(f"  ⚠️ Could not load saved model: {e}")
                print("  Training new clustering model...")
            
            # ========== TRAIN NEW CLUSTERING MODEL ==========
            print("  Training new clustering model...")
            
            # If method is None, use kmeans as default
            if method is None:
                method = 'kmeans'
            
            # Train and compare multiple clustering models
            results = self.train_and_compare_clustering_models(n_clusters)
            
            # Store the best model results
            best_model_name = results.get('best_model')
            best_model_data = results.get('all_models', {}).get(best_model_name, {})
            
            self.results['clustering'] = {
                'n_clusters': results.get('n_clusters_target', n_clusters),
                'method': best_model_name,
                'cluster_sizes': best_model_data.get('cluster_sizes', {}),
                'evaluation': best_model_data.get('evaluation', {}),
                'using_saved_model': False,
                'model_id': results.get('best_model_id')
            }
            
            return self.results['clustering']
            
        except Exception as e:
            print(f"❌ Clustering error: {e}")
            traceback.print_exc()
            raise
    # def _create_clustering_visualization(self, X_scaled, clusters, method_name, evaluation):
    #     """Helper method to create clustering visualization"""
    #     try:
    #         print(f"  Creating clustering visualization...")
            
    #         # Create figure with 2x2 subplots like regression/classification
    #         fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
    #         unique_clusters = np.unique(clusters)
    #         n_clusters = len(unique_clusters)
    #         colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))
            
    #         # ===== PANEL 1: PCA Scatter Plot (Top Left) =====
    #         if X_scaled.shape[1] >= 2:
    #             # Use PCA for 2D visualization
    #             pca = PCA(n_components=2)
    #             X_pca = pca.fit_transform(X_scaled)
                
    #             for i, cluster in enumerate(unique_clusters):
    #                 mask = clusters == cluster
    #                 axes[0, 0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
    #                                 color=colors[i], alpha=0.6, 
    #                                 label=f'Cluster {cluster}', s=50)
                
    #             axes[0, 0].set_xlabel('Principal Component 1')
    #             axes[0, 0].set_ylabel('Principal Component 2')
    #             axes[0, 0].set_title(f'Cluster Visualization ({method_name})')
    #             axes[0, 0].legend(loc='best', fontsize=8)
    #             axes[0, 0].grid(True, alpha=0.3)
                
    #             # Add variance explained
    #             var_explained = pca.explained_variance_ratio_
    #             axes[0, 0].annotate(f'PC1: {var_explained[0]:.1%}\nPC2: {var_explained[1]:.1%}',
    #                             xy=(0.02, 0.98), xycoords='axes fraction',
    #                             fontsize=9, ha='left', va='top',
    #                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    #         else:
    #             # Handle 1D case
    #             X_1d = X_scaled.flatten()
    #             for i, cluster in enumerate(unique_clusters):
    #                 mask = clusters == cluster
    #                 axes[0, 0].hist(X_1d[mask], bins=15, alpha=0.5, 
    #                             color=colors[i], label=f'Cluster {cluster}',
    #                             edgecolor='black', linewidth=0.5)
    #             axes[0, 0].set_xlabel('Feature Value')
    #             axes[0, 0].set_ylabel('Frequency')
    #             axes[0, 0].set_title(f'Cluster Distribution ({method_name})')
    #             axes[0, 0].legend()
    #             axes[0, 0].grid(True, alpha=0.3)
            
    #         # ===== PANEL 2: Cluster Sizes (Top Right) =====
    #         cluster_sizes = pd.Series(clusters).value_counts().sort_index()
    #         cluster_list = list(cluster_sizes.index)
    #         size_list = list(cluster_sizes.values)
            
    #         bars = axes[0, 1].bar(range(len(cluster_list)), size_list, color=colors)
    #         axes[0, 1].set_xlabel('Cluster')
    #         axes[0, 1].set_ylabel('Number of Points')
    #         axes[0, 1].set_title('Cluster Size Distribution')
    #         axes[0, 1].set_xticks(range(len(cluster_list)))
    #         axes[0, 1].set_xticklabels([f'Cluster {c}' for c in cluster_list])
    #         axes[0, 1].grid(True, alpha=0.3, axis='y')
            
    #         # Add value labels
    #         for bar, size in zip(bars, size_list):
    #             height = bar.get_height()
    #             axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
    #                         f'{int(size)}', ha='center', va='bottom', fontsize=10)
            
    #         # ===== PANEL 3: Metrics Comparison (Bottom Left) =====
    #         metrics = []
    #         values = []
            
    #         if 'silhouette_score' in evaluation:
    #             metrics.append('Silhouette')
    #             values.append(max(0, min(1, evaluation['silhouette_score'])))
            
    #         if 'calinski_harabasz_score' in evaluation:
    #             metrics.append('C-H')
    #             ch_score = evaluation['calinski_harabasz_score']
    #             norm_ch = min(1.0, ch_score / 1000) if ch_score > 0 else 0
    #             values.append(norm_ch)
            
    #         if 'davies_bouldin_score' in evaluation:
    #             metrics.append('D-B (inv)')
    #             db_score = evaluation['davies_bouldin_score']
    #             if db_score != float('inf') and db_score > 0:
    #                 norm_db = max(0, min(1, 1.0 / (1.0 + db_score)))
    #             else:
    #                 norm_db = 0
    #             values.append(norm_db)
            
    #         if metrics:
    #             metric_colors = ['#2E86AB', '#A23B72', '#F18F01']
    #             bars = axes[1, 0].bar(range(len(metrics)), values, color=metric_colors[:len(metrics)])
    #             axes[1, 0].set_xlabel('Metrics')
    #             axes[1, 0].set_ylabel('Score (normalized)')
    #             axes[1, 0].set_title('Clustering Quality Metrics')
    #             axes[1, 0].set_xticks(range(len(metrics)))
    #             axes[1, 0].set_xticklabels(metrics, rotation=45, ha='right')
    #             axes[1, 0].set_ylim([0, 1])
    #             axes[1, 0].grid(True, alpha=0.3, axis='y')
                
    #             # Add value labels
    #             for bar, val in zip(bars, values):
    #                 height = bar.get_height()
    #                 axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
    #                             f'{val:.3f}', ha='center', va='bottom', fontsize=9)
            
    #         # ===== PANEL 4: Summary Table (Bottom Right) =====
    #         axes[1, 1].axis('off')
            
    #         # Create summary table
    #         table_data = [
    #             ['Metric', 'Value'],
    #             ['Clusters', str(n_clusters)],
    #             ['Method', method_name],
    #             ['Total Points', str(len(clusters))]
    #         ]
            
    #         if 'silhouette_score' in evaluation:
    #             table_data.append(['Silhouette', f"{evaluation['silhouette_score']:.4f}"])
    #         if 'calinski_harabasz_score' in evaluation:
    #             table_data.append(['C-H Score', f"{evaluation['calinski_harabasz_score']:.2f}"])
    #         if 'davies_bouldin_score' in evaluation and evaluation['davies_bouldin_score'] != float('inf'):
    #             table_data.append(['D-B Score', f"{evaluation['davies_bouldin_score']:.4f}"])
            
    #         # Create the table
    #         table = axes[1, 1].table(cellText=table_data[1:], colLabels=table_data[0],
    #                                 cellLoc='center', loc='center')
    #         table.auto_set_font_size(False)
    #         table.set_fontsize(10)
    #         table.scale(1, 2)
            
    #         # Style the table
    #         for i in range(len(table_data)):
    #             for j in range(2):
    #                 if i == 0:
    #                     table[(i, j)].set_facecolor('#4472C4')
    #                     table[(i, j)].set_text_props(weight='bold', color='white')
    #                 elif i % 2 == 1:
    #                     table[(i-1, j)].set_facecolor('#D9E1F2')
            
    #         plt.suptitle(f'Clustering Analysis - {method_name} (k={n_clusters})', 
    #                     fontsize=16, y=1.02)
    #         plt.tight_layout()
            
    #         # ===== CONVERT TO BASE64 AND ADD TO VISUALIZATIONS =====
    #         plot_base64 = self._plot_to_base64(fig, is_matplotlib=True)
    #         if plot_base64:
    #             self.visualizations.append({
    #                 'type': 'image',
    #                 'name': f'clustering_analysis_{method_name}',
    #                 'title': f'Clustering Analysis - {method_name}',
    #                 'content': plot_base64,
    #                 'description': f'Comprehensive clustering analysis with {n_clusters} clusters'
    #             })
    #             print(f"  ✅ Added clustering visualization")
    #         else:
    #             print(f"  ⚠️ Failed to convert clustering visualization to base64")
            
    #         plt.close(fig)
            
    #     except Exception as e:
    #         print(f"Error creating clustering visualization: {e}")
    #         traceback.print_exc()            
                 
  
    def _find_optimal_clusters(self, X, method: Optional[str] = None) -> int:
        """Find optimal number of clusters using multiple methods"""
        silhouette_scores = []
        davies_bouldin_scores = []
        calinski_harabasz_scores = []
        
        max_clusters = min(15, len(X) - 1)
        
        for k in range(2, max_clusters + 1):
            try:
                if method == 'kmeans' or method is None:
                    model = KMeans(n_clusters=k, random_state=42, n_init=10)
                elif method == 'hierarchical':
                    model = AgglomerativeClustering(n_clusters=k)
                elif method == 'gaussian':
                    model = GaussianMixture(n_components=k, random_state=42)
                else:
                    model = KMeans(n_clusters=k, random_state=42, n_init=10)
                
                labels = model.fit_predict(X)
                
                if len(np.unique(labels)) > 1:
                    silhouette_scores.append(silhouette_score(X, labels))
                    davies_bouldin_scores.append(davies_bouldin_score(X, labels))
                    calinski_harabasz_scores.append(calinski_harabasz_score(X, labels))
                else:
                    silhouette_scores.append(-1)
                    davies_bouldin_scores.append(float('inf'))
                    calinski_harabasz_scores.append(0)
            except:
                silhouette_scores.append(-1)
                davies_bouldin_scores.append(float('inf'))
                calinski_harabasz_scores.append(0)
        
        # Combine scores (normalize first)
        silhouette_norm = (silhouette_scores - np.min(silhouette_scores)) / (np.max(silhouette_scores) - np.min(silhouette_scores))
        db_norm = 1 - ((davies_bouldin_scores - np.min(davies_bouldin_scores)) / (np.max(davies_bouldin_scores) - np.min(davies_bouldin_scores)))
        ch_norm = (calinski_harabasz_scores - np.min(calinski_harabasz_scores)) / (np.max(calinski_harabasz_scores) - np.min(calinski_harabasz_scores))
        
        combined_scores = silhouette_norm + db_norm + ch_norm
        optimal_k = np.argmax(combined_scores) + 2
        
        return optimal_k
    
    def _apply_clustering_algorithm(self, X, method: Optional[str], n_clusters: int):
        """Apply specific clustering algorithm"""
        if method == 'dbscan':
            model = DBSCAN(eps=0.5, min_samples=5)
            clusters = model.fit_predict(X)
        elif method == 'hierarchical':
            model = AgglomerativeClustering(n_clusters=n_clusters)
            clusters = model.fit_predict(X)
        elif method == 'gaussian':
            model = GaussianMixture(n_components=n_clusters, random_state=42)
            clusters = model.fit_predict(X)
        else:  # kmeans default
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = model.fit_predict(X)
        
        return clusters, model
    
    def _evaluate_clustering(self, X, clusters, n_clusters) -> Dict:
        """Evaluate clustering with multiple metrics"""
        evaluation = {}
        
        if len(np.unique(clusters)) > 1:
            try:
                evaluation['silhouette_score'] = float(silhouette_score(X, clusters))
            except:
                evaluation['silhouette_score'] = -1
            
            try:
                evaluation['davies_bouldin_score'] = float(davies_bouldin_score(X, clusters))
            except:
                evaluation['davies_bouldin_score'] = float('inf')
            
            try:
                evaluation['calinski_harabasz_score'] = float(calinski_harabasz_score(X, clusters))
            except:
                evaluation['calinski_harabasz_score'] = 0
        
        return evaluation
    
    def _get_feature_importance(self, model, feature_names):
        """Extract feature importance from model"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            return dict(zip(feature_names, importances))
        elif hasattr(model, 'coef_'):
            if len(model.coef_.shape) > 1:
                importances = np.mean(np.abs(model.coef_), axis=0)
            else:
                importances = np.abs(model.coef_)
            return dict(zip(feature_names, importances))
        return None
    
  
    def _plot_to_base64(self, plt_fig=None, is_matplotlib=True):
        """Convert matplotlib or plotly figure to base64"""
        try:
            if is_matplotlib:
                # For matplotlib
                buf = io.BytesIO()
                if plt_fig:
                    plt_fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                else:
                    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                    plt.close()
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                return img_base64
            else:
                # For plotly - try different methods
                buf = io.BytesIO()
                
                try:
                    # Method 1: Use plotly's built-in export
                    plt_fig.write_image(buf, format='png', width=1200, height=800, scale=1)
                except Exception as e:
                    print(f"Method 1 failed: {e}")
                    # Method 2: Try to_plotly_json and render
                    try:
                        import plotly.io as pio
                        img_bytes = pio.to_image(plt_fig, format='png', width=1200, height=800, scale=1)
                        buf.write(img_bytes)
                    except Exception as e2:
                        print(f"Method 2 failed: {e2}")
                        # Method 3: Use matplotlib backend as fallback
                        try:
                            fig = plt.figure(figsize=(12, 8))
                            ax = fig.add_subplot(111)
                            # Convert plotly to matplotlib if possible
                            if hasattr(plt_fig, 'to_plotly_json'):
                                data = plt_fig.to_plotly_json()
                                # Simple fallback - create a placeholder
                                ax.text(0.5, 0.5, 'Plotly visualization\nNot available in this mode', 
                                       ha='center', va='center', transform=ax.transAxes)
                            else:
                                ax.text(0.5, 0.5, 'Visualization', 
                                       ha='center', va='center', transform=ax.transAxes)
                            ax.axis('off')
                            fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                            plt.close(fig)
                        except:
                            # Final fallback
                            return None
                
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                return img_base64
        except Exception as e:
            print(f"Error converting plot to base64: {e}")
            return None
    
    