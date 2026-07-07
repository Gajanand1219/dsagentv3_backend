# utils/model_persistence.py - Add this at the beginning of the file

import os
import json
import joblib
import hashlib
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List

class ModelPersistenceManager:
    def __init__(self, models_dir: str = "saved_models"):
        """
        Initialize model persistence manager
        """
        self.models_dir = models_dir
        
        # ========== FIX: Create directory if it doesn't exist ==========
        os.makedirs(self.models_dir, exist_ok=True)
        print(f"✅ Models directory: {self.models_dir}")
        
        self.metadata_file = os.path.join(self.models_dir, "metadata.json")
        self._init_metadata()
    
    def _init_metadata(self):
        """Initialize metadata file if it doesn't exist"""
        if not os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'w') as f:
                json.dump({}, f)
    
    def calculate_dataset_hash(self, df: pd.DataFrame) -> str:
        """Calculate hash of dataset for identification"""
        df_str = df.to_string()
        return hashlib.md5(df_str.encode()).hexdigest()[:16]
    
    def generate_model_id(self, dataset_hash: str, task_type: str, target: str = None) -> str:
        """Generate unique model ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if target:
            return f"{dataset_hash}_{task_type}_{target}_{timestamp}"
        return f"{dataset_hash}_{task_type}_{timestamp}"
    
    def save_model(self, model_id: str, model, scaler=None, 
                  label_encoders=None, feature_columns=None, 
                  metadata: Dict = None) -> str:
        """Save model and associated artifacts"""
        try:
            # Create model directory
            model_dir = os.path.join(self.models_dir, model_id)
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model
            model_path = os.path.join(model_dir, "model.pkl")
            joblib.dump(model, model_path)
            
            # Save scaler if exists
            if scaler:
                scaler_path = os.path.join(model_dir, "scaler.pkl")
                joblib.dump(scaler, scaler_path)
            
            # Save label encoders if exists
            if label_encoders:
                encoders_path = os.path.join(model_dir, "label_encoders.pkl")
                joblib.dump(label_encoders, encoders_path)
            
            # Save feature columns
            if feature_columns:
                cols_path = os.path.join(model_dir, "feature_columns.json")
                with open(cols_path, 'w') as f:
                    json.dump(feature_columns, f)
            
            # Save metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                'model_id': model_id,
                'saved_at': datetime.now().isoformat(),
                'has_scaler': scaler is not None,
                'has_encoders': label_encoders is not None,
                'feature_count': len(feature_columns) if feature_columns else 0
            })
            
            meta_path = os.path.join(model_dir, "metadata.json")
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Update main metadata registry
            self._update_metadata_registry(model_id, metadata)
            
            print(f"✅ Model saved: {model_id}")
            return model_id
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            raise
    
    def _update_metadata_registry(self, model_id: str, metadata: Dict):
        """Update the main metadata registry"""
        try:
            with open(self.metadata_file, 'r') as f:
                all_metadata = json.load(f)
        except:
            all_metadata = {}
        
        all_metadata[model_id] = metadata
        
        with open(self.metadata_file, 'w') as f:
            json.dump(all_metadata, f, indent=2, default=str)
    
    def load_model(self, model_id: str) -> Dict[str, Any]:
        """Load model and associated artifacts - Supports both ML and Forecast models"""
        model_dir = os.path.join(self.models_dir, model_id)
        
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model {model_id} not found")
        
        loaded_data = {}
        
        # ========== NEW: Check for forecast model first ==========
        # Forecast models save ML model as ml_model.pkl
        forecast_model_path = os.path.join(model_dir, "ml_model.pkl")
        if os.path.exists(forecast_model_path):
            print(f"📂 Loading forecast model from: {forecast_model_path}")
            loaded_data['model'] = joblib.load(forecast_model_path)
            
            # Load scaler for forecast model
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            if os.path.exists(scaler_path):
                loaded_data['scaler'] = joblib.load(scaler_path)
            
            # Load metadata for forecast model
            meta_path = os.path.join(model_dir, "ml_metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    loaded_data['metadata'] = json.load(f)
            else:
                loaded_data['metadata'] = {'model_type': 'forecast'}
            
            # For forecast models, feature columns are just date and value
            loaded_data['feature_columns'] = ['date', 'value']
            loaded_data['label_encoders'] = {}
            
            print(f"✅ Forecast model loaded successfully")
            return loaded_data
        
        # ========== Regular ML model loading ==========
        # Load model
        model_path = os.path.join(model_dir, "model.pkl")
        if os.path.exists(model_path):
            loaded_data['model'] = joblib.load(model_path)
        else:
            raise FileNotFoundError(f"Model file not found in {model_dir}")
        
        # Load scaler
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            loaded_data['scaler'] = joblib.load(scaler_path)
        
        # Load label encoders
        encoders_path = os.path.join(model_dir, "label_encoders.pkl")
        if os.path.exists(encoders_path):
            loaded_data['label_encoders'] = joblib.load(encoders_path)
        else:
            loaded_data['label_encoders'] = {}
        
        # Load feature columns
        cols_path = os.path.join(model_dir, "feature_columns.json")
        if os.path.exists(cols_path):
            with open(cols_path, 'r') as f:
                loaded_data['feature_columns'] = json.load(f)
        else:
            loaded_data['feature_columns'] = []
        
        # Load metadata
        meta_path = os.path.join(model_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                loaded_data['metadata'] = json.load(f)
        else:
            loaded_data['metadata'] = {}
        
        return loaded_data
    
    def list_all_models(self) -> List[Dict]:
        """List all saved models with metadata"""
        models = []
        
        # ========== FIX: Check if directory exists ==========
        if not os.path.exists(self.models_dir):
            print(f"⚠️ Models directory '{self.models_dir}' does not exist, creating...")
            os.makedirs(self.models_dir, exist_ok=True)
            return []
        
        try:
            # Load main metadata registry
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    all_metadata = json.load(f)
            else:
                all_metadata = {}
            
            # Get all model directories
            for item in os.listdir(self.models_dir):
                item_path = os.path.join(self.models_dir, item)
                if os.path.isdir(item_path):
                    # Check if it's a forecast model (has ml_model.pkl)
                    forecast_model_path = os.path.join(item_path, "ml_model.pkl")
                    regular_model_path = os.path.join(item_path, "model.pkl")
                    
                    if os.path.exists(forecast_model_path) or os.path.exists(regular_model_path):
                        # Get metadata from registry or directory
                        if item in all_metadata:
                            metadata = all_metadata[item]
                        else:
                            # Try to load from directory
                            meta_path = os.path.join(item_path, "metadata.json")
                            if os.path.exists(meta_path):
                                with open(meta_path, 'r') as f:
                                    metadata = json.load(f)
                            else:
                                # Try forecast metadata
                                forecast_meta_path = os.path.join(item_path, "ml_metadata.json")
                                if os.path.exists(forecast_meta_path):
                                    with open(forecast_meta_path, 'r') as f:
                                        metadata = json.load(f)
                                else:
                                    metadata = {}
                        
                        # Determine model type
                        model_type = "forecast" if os.path.exists(forecast_model_path) else "ml"
                        
                        models.append({
                            'model_id': item,
                            'path': item_path,
                            'model_type': model_type,
                            'metadata': metadata,
                            'saved_at': metadata.get('saved_at', 'Unknown')
                        })
            
            # Sort by saved_at (newest first)
            models.sort(key=lambda x: x.get('saved_at', ''), reverse=True)
            
        except Exception as e:
            print(f"⚠️ Error listing models: {e}")
        
        return models
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a saved model"""
        import shutil
        model_dir = os.path.join(self.models_dir, model_id)
        
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
            
            # Remove from metadata registry
            try:
                with open(self.metadata_file, 'r') as f:
                    all_metadata = json.load(f)
                
                if model_id in all_metadata:
                    del all_metadata[model_id]
                
                with open(self.metadata_file, 'w') as f:
                    json.dump(all_metadata, f, indent=2, default=str)
            except:
                pass
            
            return True
        
        return False
    
    def check_model_exists(self, dataset_hash: str, task_type: str, target: str = None) -> Optional[str]:
        """Check if a model exists for given dataset and task"""
        try:
            with open(self.metadata_file, 'r') as f:
                all_metadata = json.load(f)
            
            for model_id, metadata in all_metadata.items():
                if (metadata.get('dataset_hash') == dataset_hash and 
                    metadata.get('task_type') == task_type):
                    if target is None or metadata.get('target_column') == target:
                        return model_id
            
            return None
            
        except:
            return None