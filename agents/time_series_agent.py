import os
import io
import base64
import pickle
import json
import threading
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import torch
from transformers import TimesFmModelForPrediction
# MLflow Integration
import mlflow
import mlflow.sklearn
from ml.mlflow_integration import mlflow_manager

# ========== MACHINE LEARNING ALGORITHMS ==========
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb

class TimeSeriesAgent:
    def __init__(self, azure_client=None):
        # TimesFM model (fast forecasting)
        self.timesfm_model = None
        
        # ML models (background training)
        self.ml_models = {}
        self.best_ml_model = None
        self.best_ml_model_name = None
        self.best_ml_score = -np.inf
        self.scaler = StandardScaler()
        
        self.visualizations = []
        self.models_dir = "saved_models"
        self.azure_client = azure_client
        self.ml_training_complete = False
        self.ml_results = None
        
        self._init_timesfm()
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure saved_models directory exists"""
        os.makedirs(self.models_dir, exist_ok=True)
        print(f"✅ Models directory: {self.models_dir}")
    
    def _init_timesfm(self):
        """Initialize TimesFM 2.5 Model for fast forecasting"""
        try:
            print("\n🚀 Loading TimesFM 2.5 Model for fast forecasting...")
            self.timesfm_model = TimesFmModelForPrediction.from_pretrained(
                "google/timesfm-2.5-200m-transformers",
                attn_implementation="sdpa"
            )
            self.timesfm_model = self.timesfm_model.to(torch.float32).eval()
            print("✅ TimesFM model loaded successfully!")
        except Exception as e:
            print(f"❌ Could not load TimesFM model: {e}")
            self.timesfm_model = None
    
    def _get_model_folder(self, filename: str) -> str:
        """Generate unique folder name for this file"""
        base_name = os.path.splitext(filename)[0]
        base_name = base_name.replace(' ', '_').replace('-', '_').replace('.', '')
        
        folder_name = f"forecast_{base_name}"
        folder_path = os.path.join(self.models_dir, folder_name)
        
        os.makedirs(folder_path, exist_ok=True)
        return folder_path
    
    def _save_ml_model(self, folder_path: str, metadata: Dict):
        """Save trained ML model as .pkl (background task)"""
        try:
            if self.best_ml_model is None:
                print("⚠️ No ML model to save")
                return False
            
            # Save model
            model_path = os.path.join(folder_path, 'ml_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(self.best_ml_model, f)
            
            # Save scaler
            scaler_path = os.path.join(folder_path, 'scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save metadata
            metadata_path = os.path.join(folder_path, 'ml_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump({
                    'model_type': 'ensemble_ml',
                    'best_model': self.best_ml_model_name,
                    'best_score': float(self.best_ml_score),
                    'all_models': self.ml_models,
                    'date_column': metadata.get('date_column'),
                    'value_column': metadata.get('value_column'),
                    'saved_at': datetime.now().isoformat()
                }, f, indent=2)
            
            print(f"✅ ML Model saved as .pkl: {model_path}")
            return True
            
        except Exception as e:
            print(f"⚠️ Could not save ML model: {e}")
            return False
    
    def auto_detect_date_column(self, df: pd.DataFrame) -> str:
        """Auto-detect date column"""
        print("\n🔍 Auto-detecting DATE column...")
        
        for col in df.columns:
            try:
                pd.to_datetime(df[col].dropna().head(5), errors='coerce')
                print(f"  ✅ Found date column: '{col}'")
                return col
            except:
                continue
        
        print(f"  ⚠️ Using first column: '{df.columns[0]}'")
        return df.columns[0]
    
    def auto_detect_value_column(self, df: pd.DataFrame) -> str:
        """Auto-detect value column"""
        print("\n🔍 Auto-detecting VALUE column...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            print(f"  ✅ Using numeric column: '{numeric_cols[0]}'")
            return numeric_cols[0]
        
        print(f"  ⚠️ Using second column: '{df.columns[1] if len(df.columns) > 1 else df.columns[0]}'")
        return df.columns[1] if len(df.columns) > 1 else df.columns[0]
    
    def prepare_time_series_data(self, df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
        """Convert data to time series format"""
        print("\n" + "="*60)
        print("📊 TIME SERIES PREPARATION")
        print("="*60)
        
        print(f"\n✅ Using:")
        print(f"   📅 Date: '{date_col}'")
        print(f"   💰 Value: '{value_col}'")
        
        df = df.copy()
        df['_parsed_date'] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=['_parsed_date'])
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df = df.dropna(subset=[value_col])
        df = df.sort_values('_parsed_date')
        
        ts_df = df.groupby('_parsed_date')[value_col].sum().reset_index()
        ts_df.columns = ['date', f'{value_col}_total']
        ts_df = ts_df.sort_values('date')
        
        print(f"\n✅ Time series created: {len(ts_df)} days")
        print(f"   Date range: {ts_df['date'].min().date()} to {ts_df['date'].max().date()}")
        
        return ts_df
    
    def create_features(self, ts_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        """Create features for ML models"""
        df = ts_df.copy()
        values = df[f'{value_col}_total'].values
        
        # Date features
        df['dayofweek'] = df['date'].dt.dayofweek
        df['dayofmonth'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['weekend'] = (df['dayofweek'] >= 5).astype(int)
        
        # Lag features
        for lag in [1, 2, 3, 7, 14]:
            if lag < len(values):
                df[f'lag_{lag}'] = values[:-lag].tolist() + [np.nan] * lag
        
        # Rolling statistics
        for window in [3, 7, 14]:
            if window < len(values):
                df[f'rolling_mean_{window}'] = pd.Series(values).rolling(window).mean().values
                df[f'rolling_std_{window}'] = pd.Series(values).rolling(window).std().values
        
        return df.dropna()
    
    def train_ml_models_background(self, X_train, y_train, X_test, y_test, folder_path, metadata):
        """Train ML models in background thread with MLflow tracking"""
        try:
            print("\n🔄 [BACKGROUND] Training ML models...")
            
            # Start MLflow run for background training
            run_name = f"timeseries_ml_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            with mlflow.start_run(run_name=run_name):
                # Log parameters
                mlflow.log_param("task_type", "timeseries_forecast")
                mlflow.log_param("date_column", metadata.get('date_column', 'unknown'))
                mlflow.log_param("value_column", metadata.get('value_column', 'unknown'))
                mlflow.log_param("horizon", metadata.get('horizon', 30))
                mlflow.log_param("filename", metadata.get('filename', 'unknown'))
                mlflow.log_param("training_mode", "background")
                
                models = {
                    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
                    'Linear Regression': LinearRegression(),
                    'Ridge': Ridge(alpha=1.0),
                    'Lasso': Lasso(alpha=0.1),
                    'Decision Tree': DecisionTreeRegressor(random_state=42),
                    'KNN': KNeighborsRegressor(n_neighbors=5)
                }
                
                self.ml_models = {}
                self.best_ml_score = -np.inf
                
                for name, model in models.items():
                    try:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        self.ml_models[name] = {
                            'rmse': float(rmse),
                            'mae': float(mae),
                            'r2': float(r2)
                        }
                        
                        # Log individual model metrics
                        mlflow.log_metric(f"{name}_rmse", rmse)
                        mlflow.log_metric(f"{name}_mae", mae)
                        mlflow.log_metric(f"{name}_r2", r2)
                        
                        print(f"   [BG] {name}: R² = {r2:.4f}, RMSE = {rmse:.2f}")
                        
                        if r2 > self.best_ml_score:
                            self.best_ml_score = r2
                            self.best_ml_model = model
                            self.best_ml_model_name = name
                            
                    except Exception as e:
                        print(f"   [BG] {name} failed: {e}")
                        mlflow.log_param(f"{name}_error", str(e))
                
                # Log best model info
                mlflow.log_param("best_model", self.best_ml_model_name)
                mlflow.log_metric("best_r2_score", self.best_ml_score)
                
                # Log dataset info
                mlflow.log_param("train_samples", len(X_train))
                mlflow.log_param("test_samples", len(X_test))
                mlflow.log_param("features", X_train.shape[1])
                
                # Save and log best model
                if self.best_ml_model:
                    mlflow.sklearn.log_model(self.best_ml_model, "best_model")
                    
                    # Save locally as .pkl
                    self._save_ml_model(folder_path, metadata)
                    
                    # Log model info as artifact
                    model_info = {
                        'best_model': self.best_ml_model_name,
                        'best_score': float(self.best_ml_score),
                        'all_models': self.ml_models,
                        'date_column': metadata.get('date_column'),
                        'value_column': metadata.get('value_column'),
                        'saved_at': datetime.now().isoformat()
                    }
                    
                    info_path = os.path.join(folder_path, 'mlflow_info.json')
                    with open(info_path, 'w') as f:
                        json.dump(model_info, f, indent=2)
                    mlflow.log_artifact(info_path)
                
                # Log tags
                mlflow.set_tag("agent", "time_series")
                mlflow.set_tag("forecast_type", "ml_ensemble")
                mlflow.set_tag("background_training", "true")
                
                run_id = mlflow.active_run().info.run_id
                print(f"📊 MLflow run ID: {run_id}")
                
                self.ml_training_complete = True
                
        except Exception as e:
            print(f"\n❌ [BACKGROUND] ML training error: {e}")
            with mlflow.start_run(run_name="timeseries_ml_error"):
                mlflow.log_param("error", str(e))
                mlflow.set_tag("status", "failed")


    def detect_seasonality(self, series: np.array, dates: pd.Series) -> Dict[str, Any]:
        """Detect patterns in time series data"""
        results = {
            'has_weekly_pattern': False,
            'has_monthly_pattern': False,
            'weekly_pattern': {},
            'monthly_pattern': {},
            'peak_day': None,
            'peak_month': None,
            'weekend_effect': False,
            'trend': 'stable'
        }
        
        if len(series) < 14:
            return results
        
        temp_df = pd.DataFrame({'date': dates, 'value': series})
        
        # Weekly pattern
        temp_df['dayofweek'] = temp_df['date'].dt.day_name()
        weekly = temp_df.groupby('dayofweek')['value'].mean()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly = weekly.reindex(day_order)
        
        results['weekly_pattern'] = {k: float(v) if pd.notna(v) else 0 for k, v in weekly.items()}
        
        # Check weekend effect
        weekday_avg = weekly[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']].mean()
        weekend_avg = weekly[['Saturday', 'Sunday']].mean()
        
        if pd.notna(weekday_avg) and pd.notna(weekend_avg):
            if weekend_avg > weekday_avg * 1.1:
                results['has_weekly_pattern'] = True
                results['weekend_effect'] = True
                results['peak_day'] = weekly.idxmax() if pd.notna(weekly.max()) else None
        
        # Monthly pattern
        if len(series) >= 60:
            temp_df['month'] = temp_df['date'].dt.month
            monthly = temp_df.groupby('month')['value'].mean()
            results['monthly_pattern'] = {int(k): float(v) if pd.notna(v) else 0 for k, v in monthly.items()}
            
            if 12 in monthly.index and pd.notna(monthly[12]):
                if monthly[12] > monthly.mean() * 1.2:
                    results['has_monthly_pattern'] = True
                    results['peak_month'] = 12
        
        # Detect trend
        if len(series) > 1:
            first_half = series[:len(series)//2]
            second_half = series[len(series)//2:]
            
            if np.mean(second_half) > np.mean(first_half) * 1.1:
                results['trend'] = 'increasing'
            elif np.mean(second_half) < np.mean(first_half) * 0.9:
                results['trend'] = 'decreasing'
        
        return results
    
    def forecast(self, df: pd.DataFrame, filename: str = "uploaded_file", 
             date_col: str = None, value_col: str = None, horizon: int = 30) -> Dict[str, Any]:
        """
        PARALLEL PROCESSING:
        - Task 1: TimesFM fast forecast (immediate) with MLflow
        - Task 2: Train ML models in background (saves .pkl)
        """
        try:
            print("\n" + "="*60)
            print("🤖 PARALLEL FORECASTING with MLflow")
            print("="*60)
            
            # Start MLflow run for TimesFM forecast
            run_name = f"timesfm_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            with mlflow.start_run(run_name=run_name):
                # Log parameters
                mlflow.log_param("task_type", "timesfm_forecast")
                mlflow.log_param("filename", filename)
                mlflow.log_param("horizon", horizon)
                mlflow.log_param("model", "google/timesfm-2.5-200m-transformers")
                
                # Auto-detect columns
                if date_col is None or date_col not in df.columns:
                    date_col = self.auto_detect_date_column(df)
                    mlflow.log_param("auto_detected_date", date_col)
                
                if value_col is None or value_col not in df.columns:
                    value_col = self.auto_detect_value_column(df)
                    mlflow.log_param("auto_detected_value", value_col)
                
                # Log dataset info
                mlflow.log_param("dataset_rows", len(df))
                mlflow.log_param("dataset_columns", len(df.columns))
                
                # Get model folder for this file
                model_folder = self._get_model_folder(filename)
                
                # ===== TASK 1: TIMESFM FAST FORECAST =====
                print(f"\n🚀 TASK 1: Generating TimesFM forecast...")
                
                # Prepare time series data
                ts_df = self.prepare_time_series_data(df, date_col, value_col)
                series = ts_df[f'{value_col}_total'].values
                dates = ts_df['date']
                
                if len(series) < 14:
                    mlflow.log_param("error", "insufficient_data")
                    mlflow.log_param("data_points", len(series))
                    return {
                        'success': False,
                        'error': 'Insufficient data',
                        'message': f'Need at least 14 data points, but only have {len(series)}'
                    }
                
                # Log time series info
                mlflow.log_param("time_series_days", len(ts_df))
                mlflow.log_param("date_range_start", ts_df['date'].min().strftime('%Y-%m-%d'))
                mlflow.log_param("date_range_end", ts_df['date'].max().strftime('%Y-%m-%d'))
                
                # Detect seasonality
                seasonality = self.detect_seasonality(series, dates)
                mlflow.log_metric("has_weekly_pattern", int(seasonality.get('has_weekly_pattern', False)))
                mlflow.log_metric("has_monthly_pattern", int(seasonality.get('has_monthly_pattern', False)))
                mlflow.log_metric("weekend_effect", int(seasonality.get('weekend_effect', False)))
                mlflow.log_param("trend", seasonality.get('trend', 'stable'))
                
                # Calculate statistics for normalization
                mean_val = np.mean(series)
                std_val = np.std(series)
                mlflow.log_metric("historical_mean", mean_val)
                mlflow.log_metric("historical_std", std_val)
                
                # Generate forecast with TimesFM
                print(f"   TimesFM forecasting for {horizon} days...")
                mlflow.log_param("forecast_horizon", horizon)
                
                # Normalize
                if std_val > 0:
                    normalized_series = (series - mean_val) / std_val
                else:
                    normalized_series = series - mean_val
                
                input_tensor = torch.tensor(normalized_series, dtype=torch.float32)
                
                with torch.no_grad():
                    outputs = self.timesfm_model(
                        past_values=[input_tensor],
                        forecast_context_len=1024  
                    )
                
                if hasattr(outputs, 'mean_predictions'):
                    normalized_forecast = outputs.mean_predictions[0].cpu().numpy()[:horizon]
                else:
                    normalized_forecast = outputs[0, :horizon].cpu().numpy()
                
                # Denormalize
                if std_val > 0:
                    forecast_values = (normalized_forecast * std_val) + mean_val
                else:
                    forecast_values = normalized_forecast + mean_val
                
                # Create forecast dates
                last_date = ts_df['date'].iloc[-1]
                forecast_dates = [last_date + timedelta(days=i+1) for i in range(horizon)]
                
                # ===== FIX: Create forecast_data BEFORE using it =====
                forecast_data = [
                    {'date': d.strftime('%d-%m-%Y'), 'value': float(forecast_values[i])}
                    for i, d in enumerate(forecast_dates)
                ]
                
                # Statistics
                valid_series = series[~np.isnan(series)]
                mean_val_stat = float(np.mean(valid_series)) if len(valid_series) > 0 else 0
                total_val = float(np.sum(valid_series)) if len(valid_series) > 0 else 0
                
                # Create visualization (now forecast_data is defined)
                viz_base64 = self._create_forecast_visualization(
                    ts_df, forecast_data, 'date', value_col, seasonality
                )
                
                # Log forecast metrics
                mlflow.log_metric("forecast_mean", float(np.mean(forecast_values)))
                mlflow.log_metric("forecast_min", float(np.min(forecast_values)))
                mlflow.log_metric("forecast_max", float(np.max(forecast_values)))
                
                # Save and log visualization
                if viz_base64:
                    viz_path = f"forecast_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    with open(viz_path, "wb") as f:
                        f.write(base64.b64decode(viz_base64))
                    mlflow.log_artifact(viz_path)
                    os.remove(viz_path)
                
                # Log tags
                mlflow.set_tag("forecast_type", "timesfm")
                mlflow.set_tag("status", "success")
                
                run_id = mlflow.active_run().info.run_id
                print(f"📊 TimesFM MLflow run: {run_id}")
                
                # Prepare TimesFM result
                timesfm_result = {
                    'success': True,
                    'model_folder': model_folder,
                    'mlflow_run_id': run_id,
                    'auto_detected': {
                        'date_column': date_col,
                        'value_column': value_col,
                    },
                    'historical': {
                        'dates': ts_df['date'].dt.strftime('%d-%m-%Y').tolist(),
                        'values': [float(v) for v in series],
                    },
                    'forecast': forecast_data,
                    'seasonality': seasonality,
                    'statistics': {
                        'mean': mean_val_stat,
                        'total': total_val,
                        'trend': seasonality.get('trend', 'stable'),
                        'date_range': {
                            'start': ts_df['date'].min().strftime('%d-%m-%Y'),
                            'end': ts_df['date'].max().strftime('%d-%m-%Y')
                        },
                        'historical_min': float(np.min(series)),
                        'historical_max': float(np.max(series)),
                        'forecast_min': float(np.min(forecast_values)),
                        'forecast_max': float(np.max(forecast_values))
                    },
                    'visualization': viz_base64,
                    'model_used': 'google/timesfm-2.5-200m-transformers'
                }
                
                # ===== TASK 2: START ML TRAINING IN BACKGROUND =====
                if len(series) >= 30:
                    try:
                        print(f"\n🔄 TASK 2: Starting ML model training in background...")
                        
                        # Create features for ML
                        feature_df = self.create_features(ts_df, value_col)
                        
                        if len(feature_df) >= 30:
                            X = feature_df.drop(columns=['date', f'{value_col}_total']).values
                            y = feature_df[f'{value_col}_total'].values
                            
                            X_scaled = self.scaler.fit_transform(X)
                            split_idx = int(len(X_scaled) * 0.8)
                            
                            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
                            y_train, y_test = y[:split_idx], y[split_idx:]
                            
                            metadata = {
                                'date_column': date_col,
                                'value_column': value_col,
                                'horizon': horizon,
                                'filename': filename
                            }
                            
                            # Start background thread
                            thread = threading.Thread(
                                target=self.train_ml_models_background,
                                args=(X_train, y_train, X_test, y_test, model_folder, metadata)
                            )
                            thread.daemon = True
                            thread.start()
                            
                            timesfm_result['ml_training_started'] = True
                            timesfm_result['ml_training_message'] = "ML model training in background. Model will be saved as .pkl"
                        else:
                            timesfm_result['ml_training_started'] = False
                            timesfm_result['ml_training_message'] = "Insufficient data for ML training (need 30+ samples)"
                            
                    except Exception as e:
                        print(f"⚠️ Could not start ML training: {e}")
                        timesfm_result['ml_training_started'] = False
                        timesfm_result['ml_training_error'] = str(e)
                else:
                    timesfm_result['ml_training_started'] = False
                    timesfm_result['ml_training_message'] = "Insufficient data for ML training (need 30+ samples)"
                
                return timesfm_result
                
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Try to log error to MLflow if possible
            try:
                with mlflow.start_run(run_name="timesfm_forecast_error"):
                    mlflow.log_param("error", str(e))
                    mlflow.set_tag("status", "failed")
            except:
                pass
            
            return {
                'success': False,
                'error': str(e),
                'message': 'Forecasting failed'
            }


    def _create_forecast_visualization(self, ts_df, forecast_data, date_col, value_col, seasonality):
        """Create 4-panel visualization"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # PLOT 1: Historical + Forecast
            ax1 = axes[0, 0]
            hist_dates = ts_df['date']
            hist_values = ts_df[f'{value_col}_total']
            
            ax1.plot(hist_dates, hist_values, 'b-', linewidth=2, label='Historical', alpha=0.7)
            
            f_dates = [datetime.strptime(f['date'], '%d-%m-%Y') for f in forecast_data]
            f_values = [f['value'] for f in forecast_data]
            
            ax1.plot(f_dates, f_values, 'r--', linewidth=2, label='TimesFM Forecast', alpha=0.7)
            ax1.axvline(x=hist_dates.iloc[-1], color='gray', linestyle=':', alpha=0.5)
            
            ax1.set_xlabel('Date')
            ax1.set_ylabel(value_col)
            ax1.set_title('TimesFM Fast Forecast')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # PLOT 2: Weekly Pattern
            ax2 = axes[0, 1]
            if seasonality['weekly_pattern'] and any(seasonality['weekly_pattern'].values()):
                days = list(seasonality['weekly_pattern'].keys())
                values = list(seasonality['weekly_pattern'].values())
                
                colors = ['orange' if d in ['Saturday', 'Sunday'] else 'steelblue' for d in days]
                ax2.bar(days, values, color=colors, alpha=0.7)
                ax2.set_xlabel('Day of Week')
                ax2.set_ylabel(f'Average {value_col}')
                ax2.set_title('Weekly Pattern')
                ax2.tick_params(axis='x', rotation=45)
            else:
                ax2.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
                ax2.set_title('Weekly Pattern')
            
            # PLOT 3: Monthly Pattern
            ax3 = axes[1, 0]
            if seasonality['monthly_pattern'] and any(seasonality['monthly_pattern'].values()):
                months = list(seasonality['monthly_pattern'].keys())
                values = list(seasonality['monthly_pattern'].values())
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                plot_months = [month_names[int(m)-1] for m in months]
                ax3.plot(plot_months, values, 'go-', linewidth=2, markersize=8)
                ax3.set_xlabel('Month')
                ax3.set_ylabel(f'Average {value_col}')
                ax3.set_title('Monthly Pattern')
                ax3.tick_params(axis='x', rotation=45)
            else:
                ax3.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
                ax3.set_title('Monthly Pattern')
            
            # PLOT 4: Info
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            info_text = f"TimesFM 2.5\n\n"
            info_text += f"Total Days: {len(ts_df)}\n"
            info_text += f"Trend: {seasonality.get('trend', 'stable').upper()}\n"
            info_text += f"Weekend Effect: {'Yes' if seasonality.get('weekend_effect') else 'No'}\n\n"
            info_text += f"⚡ Fast Forecast: TimesFM\n"
            info_text += f"🔄 Background: ML Training → .pkl"
            
            ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow'))
            
            plt.suptitle('Parallel Processing: TimesFM + ML Training', fontsize=14, y=1.02)
            plt.tight_layout()
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            return img_base64
            
        except Exception as e:
            print(f"Visualization error: {e}")
            return None