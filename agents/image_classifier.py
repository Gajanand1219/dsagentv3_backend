import os
import json
import base64
import traceback
from io import BytesIO
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

# Image Processing Libraries
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available for image classification")

try:
    from PIL import Image
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV/PIL not available for image processing")

# Visualization (server-safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns




class ImageClassificationAgent:
    """AI-Powered Image Classification Agent - Supports All Tasks from Screenshot"""
    
    def __init__(self, azure_client=None):
        self.azure_client = azure_client
        self.model = None
        self.results = {}
        self.visualizations = []
        self.task_models = {}  # Store different models for different tasks
        
        # Initialize models for different tasks
        self._init_models()
        
    def _init_models(self):
        """Initialize or load models for different tasks"""
        try:
            if TF_AVAILABLE:
                # Load ImageNet model for 1000 category classification
                self.imagenet_model = tf.keras.applications.MobileNetV2(weights='imagenet')
                
                # MNIST model for 0-9 classification
                self._init_mnist_model()
                
                # Binary model for sign detection (placeholder - would need training data)
                self._init_binary_model()
                
        except Exception as e:
            print(f"Warning: Could not initialize all models: {e}")
    
    def _init_mnist_model(self):
        """Initialize or load MNIST model for 0-9 digit classification"""
        try:
            # Check if model exists locally
            mnist_model_path = "mnist_digit_classifier.h5"
            
            if os.path.exists(mnist_model_path):
                self.mnist_model = tf.keras.models.load_model(mnist_model_path)
                print(" Loaded existing MNIST model")
            else:
                print(" Note: MNIST model not found. Use train_mnist_model() to create one.")
                self.mnist_model = None
                
        except Exception as e:
            print(f"Warning: Could not initialize MNIST model: {e}")
            self.mnist_model = None
    
    def _init_binary_model(self):
        """Initialize binary classification model for sign detection"""
        # This is a placeholder - actual model would need training data
        self.binary_model = None
        print(" Note: Binary sign detection model requires training data")
    
    def detect_image_task_type(self, prompt: str) -> Dict[str, Any]:
        """Enhanced AI-powered image task detection with screenshot tasks"""
        print(f"   Image Prompt: '{prompt}'")
        
        if not self.azure_client:
            raise Exception("OpenAI client not available for image task detection")
        
        try:
            messages = [
                {
                    "role": "system", 
                    "content": """You are an image analysis task detector. Based on common tasks like:
                    1. 0-9 digit recognition (MNIST)
                    2. Dog breed classification 
                    3. Sign present or not (binary classification)
                    4. 1000 category classification (ImageNet)
                    
                    Analyze the user prompt and determine:
                    1. Task type (digit_classification, dog_breed, sign_detection, imagenet_classification)
                    2. Suggested model (MNIST, MobileNetV2, ResNet50, CustomCNN)
                    3. Number of classes expected
                    4. Special requirements
                    
                    Return JSON format with task_type, model_suggestion, expected_classes, explanation"""
                },
                {
                    "role": "user", 
                    "content": f"Image analysis prompt: {prompt}"
                }
            ]
            
            response = self.azure_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4"),
                messages=messages,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            raise Exception(f"AI image task detection failed: {str(e)}")
    
    def load_image(self, image_path: str, grayscale: bool = False) -> np.ndarray:
        """Load and preprocess image with grayscale option"""
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV/PIL not installed. Install with: pip install opencv-python pillow")
        
        try:
            # Try OpenCV first
            if grayscale:
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(image_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if img is None:
                # Try PIL
                img_pil = Image.open(image_path)
                if grayscale:
                    img_pil = img_pil.convert('L')
                img = np.array(img_pil)
            
            return img
        except Exception as e:
            raise Exception(f"Error loading image {image_path}: {str(e)}")
    
    def classify_digits_0_9(self, image_path: str) -> Dict[str, Any]:
        """Classify digits 0-9 (MNIST-like task)"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not installed. Install with: pip install tensorflow")
        
        if self.mnist_model is None:
            return self._classify_digits_with_pretrained(image_path)
        
        try:
            print(f"\n Classifying digits 0-9 in image")
            
            # Load and preprocess image for MNIST
            img = self.load_image(image_path, grayscale=True)
            
            # Resize to 28x28 (MNIST standard)
            img_resized = cv2.resize(img, (28, 28))
            
            # Normalize
            img_normalized = img_resized.astype('float32') / 255.0
            
            # Reshape for model (1, 28, 28, 1)
            img_input = img_normalized.reshape(1, 28, 28, 1)
            
            # Make prediction
            predictions = self.mnist_model.predict(img_input, verbose=0)
            predicted_digit = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_digit])
            
            # Get top 3 predictions
            top_3_indices = np.argsort(predictions[0])[-3:][::-1]
            
            results = []
            for idx in top_3_indices:
                results.append({
                    "digit": int(idx),
                    "confidence": float(predictions[0][idx]),
                    "percentage": float(predictions[0][idx] * 100)
                })
            
            # Store results
            self.results['digit_classification'] = {
                'model': 'MNIST',
                'predicted_digit': int(predicted_digit),
                'confidence': confidence,
                'all_predictions': results,
                'image_path': image_path
            }
            
            # Create visualization
            self.create_digit_classification_visualization(img_resized, results)
            
            return self.results['digit_classification']
            
        except Exception as e:
            print(f" Digit classification error: {e}")
            traceback.print_exc()
            return self._classify_digits_with_pretrained(image_path)
    
    def _classify_digits_with_pretrained(self, image_path: str) -> Dict[str, Any]:
        """Fallback: Use pre-trained model to try to identify digits"""
        print(" Using fallback method for digit classification")
        
        # Use the 1000-category model as fallback
        result = self.classify_1000_categories(image_path, model_name="MobileNetV2")
        
        # Filter results for digit-related classes
        digit_classes = []
        for pred in result['all_predictions']:
            class_name = pred['class'].lower()
            if any(digit in class_name for digit in ['zero', 'one', 'two', 'three', 'four', 
                                                    'five', 'six', 'seven', 'eight', 'nine', 'digit']):
                digit_classes.append(pred)
        
        if digit_classes:
            result['digit_predictions'] = digit_classes
        
        return result
    
    def detect_sign_present(self, image_path: str) -> Dict[str, Any]:
        """Detect if a sign is present in the image (binary classification)"""
        print(f"\n Detecting sign presence in image")
        
        try:
            # Method 1: Use pre-trained model and look for sign-related classes
            imagenet_result = self.classify_1000_categories(image_path, model_name="MobileNetV2")
            
            # Check for sign-related classes
            sign_keywords = ['sign', 'board', 'billboard', 'poster', 'placard', 'traffic', 
                           'street sign', 'advertisement', 'notice', 'indication']
            
            sign_present = False
            sign_confidence = 0.0
            sign_classes = []
            
            for pred in imagenet_result['all_predictions'][:10]:  # Check top 10 predictions
                class_name = pred['class'].lower()
                if any(keyword in class_name for keyword in sign_keywords):
                    sign_present = True
                    sign_confidence = max(sign_confidence, pred['confidence'])
                    sign_classes.append({
                        'class': pred['class'],
                        'confidence': pred['confidence']
                    })
            
            # Method 2: Simple edge detection heuristic
            img = self.load_image(image_path, grayscale=True)
            
            # Edge detection
            edges = cv2.Canny(img, 50, 150)
            edge_density = np.sum(edges > 0) / (img.shape[0] * img.shape[1])
            
            # If many edges and rectangular shapes, might be a sign
            heuristic_sign = edge_density > 0.1
            
            # Combine both methods
            final_detection = sign_present or heuristic_sign
            confidence = sign_confidence if sign_present else (edge_density * 0.5)
            
            result = {
                'sign_present': bool(final_detection),
                'confidence': float(confidence),
                'method': 'pre-trained + heuristic',
                'sign_classes_found': sign_classes,
                'edge_density': float(edge_density),
                'heuristic_detection': bool(heuristic_sign),
                'image_path': image_path
            }
            
            self.results['sign_detection'] = result
            
            # Create visualization
            self.create_sign_detection_visualization(img, edges, result)
            
            return result
            
        except Exception as e:
            print(f" Sign detection error: {e}")
            traceback.print_exc()
            return {
                'sign_present': False,
                'confidence': 0.0,
                'error': str(e),
                'image_path': image_path
            }
    
    def classify_1000_categories(self, image_path: str, model_name: str = "MobileNetV2") -> Dict[str, Any]:
        """Classify image into 1000 ImageNet categories"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not installed. Install with: pip install tensorflow")
        
        try:
            print(f"\n Classifying into 1000 categories with {model_name}")
            
            # Load image
            img = self.load_image(image_path)
            if img is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Load pre-trained model
            if model_name == "MobileNetV2":
                model = tf.keras.applications.MobileNetV2(weights='imagenet')
                preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
                decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

            elif model_name == "ResNet50":
                model = tf.keras.applications.ResNet50(weights='imagenet')
                preprocess_input = tf.keras.applications.resnet50.preprocess_input
                decode_predictions = tf.keras.applications.resnet50.decode_predictions

            elif model_name == "VGG16":
                model = tf.keras.applications.VGG16(weights='imagenet')
                preprocess_input = tf.keras.applications.vgg16.preprocess_input
                decode_predictions = tf.keras.applications.vgg16.decode_predictions

            elif model_name == "EfficientNetB0":
                model = tf.keras.applications.EfficientNetB0(weights='imagenet')
                preprocess_input = tf.keras.applications.efficientnet.preprocess_input
                decode_predictions = tf.keras.applications.efficientnet.decode_predictions
                
            else:
                print(f" Model {model_name} not found, using MobileNetV2")
                model = tf.keras.applications.MobileNetV2(weights='imagenet')
                preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
                decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions
            
            # Preprocess image
            img_resized = cv2.resize(img, (224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_array = tf.expand_dims(img_array, 0)
            img_array = preprocess_input(img_array)
            
            # Make prediction
            predictions = model.predict(img_array, verbose=0)
            
            # Decode predictions
            decoded_predictions = decode_predictions(predictions, top=10)[0]
            
            # Format results
            results = []
            for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
                results.append({
                    "rank": i + 1,
                    "class": label,
                    "confidence": float(score),
                    "percentage": float(score * 100)
                })
            
            # Check if it's a dog (for 🐶 task)
            is_dog = any('dog' in pred['class'].lower() for pred in results[:5])
            dog_predictions = [pred for pred in results if 'dog' in pred['class'].lower()]
            
            # Store results
            self.results['1000_category_classification'] = {
                'model': model_name,
                'top_prediction': results[0],
                'all_predictions': results,
                'is_dog': is_dog,
                'dog_predictions': dog_predictions[:3] if dog_predictions else [],
                'image_path': image_path,
                'image_shape': img.shape
            }
            
            # Create visualization
            self.create_1000category_visualization(img, results, model_name, is_dog)
            
            return self.results['1000_category_classification']
            
        except Exception as e:
            print(f" 1000-category classification error: {e}")
            traceback.print_exc()
            raise
    
    def train_mnist_model(self, save_path: str = "mnist_digit_classifier.h5"):
        """Train a simple MNIST model for digit classification"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required for training")
        
        print("\n Training MNIST model for 0-9 digit classification...")
        
        try:
            # Load MNIST dataset
            mnist = tf.keras.datasets.mnist
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            
            # Normalize
            x_train, x_test = x_train / 255.0, x_test / 255.0
            
            # Build model
            model = tf.keras.models.Sequential([
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10, activation='softmax')
            ])
            
            model.compile(optimizer='adam',
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
            
            # Train
            model.fit(x_train, y_train, epochs=5, validation_split=0.1, verbose=1)
            
            # Evaluate
            test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
            print(f" Test accuracy: {test_acc:.4f}")
            
            # Save model
            model.save(save_path)
            print(f" Model saved to {save_path}")
            
            self.mnist_model = model
            return model
            
        except Exception as e:
            print(f" Training error: {e}")
            traceback.print_exc()
            raise
    
    def create_digit_classification_visualization(self, digit_image: np.ndarray, predictions: List[Dict]):
        """Create visualization for digit classification"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Display digit image
            axes[0].imshow(digit_image, cmap='gray')
            axes[0].axis('off')
            axes[0].set_title('Digit Image (28x28)', fontsize=14)
            
            # Create bar chart for predictions
            digits = [str(pred['digit']) for pred in predictions]
            confidences = [pred['percentage'] for pred in predictions]
            colors = plt.cm.Set3(np.linspace(0, 1, len(predictions)))
            
            bars = axes[1].barh(range(len(predictions)), confidences, color=colors)
            axes[1].set_yticks(range(len(predictions)))
            axes[1].set_yticklabels(digits)
            axes[1].invert_yaxis()
            axes[1].set_xlabel('Confidence (%)', fontsize=12)
            axes[1].set_title('Top Digit Predictions', fontsize=14)
            axes[1].set_xlim([0, 100])
            axes[1].grid(True, alpha=0.3, axis='x')
            
            # Add confidence values
            for i, (bar, conf) in enumerate(zip(bars, confidences)):
                axes[1].text(conf + 1, bar.get_y() + bar.get_height()/2,
                           f'{conf:.1f}%', va='center', fontsize=10)
            
            plt.suptitle('Digit (0-9) Classification Results', fontsize=16, y=1.02)
            plt.tight_layout()
            
            # Convert to base64
            plot_base64 = self._plot_to_base64(fig, is_matplotlib=True)
            if plot_base64:
                self.visualizations.append({
                    'type': 'image',
                    'name': 'digit_classification_results',
                    'title': 'Digit Classification (0-9)',
                    'content': plot_base64,
                    'description': f'Digit classification showing top {len(predictions)} predictions'
                })
            
            plt.close(fig)
            
        except Exception as e:
            print(f"Error creating digit visualization: {e}")
    
    def create_sign_detection_visualization(self, original_img: np.ndarray, edges: np.ndarray, result: Dict):
        """Create visualization for sign detection"""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(original_img, cmap='gray')
            axes[0].axis('off')
            axes[0].set_title('Original Image', fontsize=12)
            
            # Edge detection
            axes[1].imshow(edges, cmap='gray')
            axes[1].axis('off')
            axes[1].set_title(f'Edge Detection (density: {result["edge_density"]:.3f})', fontsize=12)
            
            # Result info
            axes[2].axis('off')
            sign_status = "PRESENT ✓" if result['sign_present'] else "NOT PRESENT ✗"
            color = 'green' if result['sign_present'] else 'red'
            
            info_text = f"Sign Detection Result\n\n"
            info_text += f"Status: {sign_status}\n"
            info_text += f"Confidence: {result['confidence']:.2f}\n"
            info_text += f"Method: {result['method']}\n\n"
            
            if result['sign_classes_found']:
                info_text += "Detected sign types:\n"
                for sign in result['sign_classes_found'][:3]:
                    info_text += f"- {sign['class']} ({sign['confidence']:.2f})\n"
            
            axes[2].text(0.1, 0.9, info_text, transform=axes[2].transAxes,
                        fontsize=11, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
            
            plt.suptitle('Sign Presence Detection', fontsize=16, y=1.05)
            plt.tight_layout()
            
            # Convert to base64
            plot_base64 = self._plot_to_base64(fig, is_matplotlib=True)
            if plot_base64:
                self.visualizations.append({
                    'type': 'image',
                    'name': 'sign_detection_results',
                    'title': 'Sign Detection (Present/Not)',
                    'content': plot_base64,
                    'description': 'Sign presence detection with edge analysis'
                })
            
            plt.close(fig)
            
        except Exception as e:
            print(f"Error creating sign detection visualization: {e}")
    
    def create_1000category_visualization(self, image: np.ndarray, predictions: List[Dict], model_name: str, is_dog: bool):
        """Create visualization for 1000-category classification"""
        try:
            fig = plt.figure(figsize=(15, 8))
            gs = fig.add_gridspec(2, 2, height_ratios=[3, 2])
            
            # Original image
            ax1 = fig.add_subplot(gs[0, :])
            ax1.imshow(image)
            ax1.axis('off')
            
            # Add dog indicator if applicable
            title = f'Input Image - {model_name}'
            if is_dog:
                title += ' 🐶 DOG DETECTED'
                ax1.set_title(title, fontsize=14, color='green', fontweight='bold')
            else:
                ax1.set_title(title, fontsize=14)
            
            # Top predictions bar chart
            ax2 = fig.add_subplot(gs[1, 0])
            top_n = min(8, len(predictions))
            classes = [pred['class'].replace('_', ' ').title() for pred in predictions[:top_n]]
            confidences = [pred['percentage'] for pred in predictions[:top_n]]
            
            # Color code: green for dogs, blue for others
            colors = []
            for pred in predictions[:top_n]:
                if 'dog' in pred['class'].lower():
                    colors.append('lightgreen')
                else:
                    colors.append('lightblue')
            
            bars = ax2.barh(range(top_n), confidences, color=colors)
            ax2.set_yticks(range(top_n))
            ax2.set_yticklabels(classes, fontsize=9)
            ax2.invert_yaxis()
            ax2.set_xlabel('Confidence (%)', fontsize=10)
            ax2.set_title(f'Top {top_n} Predictions', fontsize=12)
            ax2.set_xlim([0, 100])
            ax2.grid(True, alpha=0.3, axis='x')
            
            # Add confidence values
            for i, (bar, conf) in enumerate(zip(bars, confidences)):
                ax2.text(conf + 1, bar.get_y() + bar.get_height()/2,
                        f'{conf:.1f}%', va='center', fontsize=8)
            
            # Stats panel
            ax3 = fig.add_subplot(gs[1, 1])
            ax3.axis('off')
            
            stats_text = f"1000-Category Classification\n"
            stats_text += f"Model: {model_name}\n"
            stats_text += f"Top Prediction: {predictions[0]['class']}\n"
            stats_text += f"Confidence: {predictions[0]['percentage']:.1f}%\n\n"
            
            stats_text += f"Image Shape: {image.shape}\n"
            
            if is_dog:
                stats_text += "\n🐶 Dog-related predictions:\n"
                dog_preds = [p for p in predictions if 'dog' in p['class'].lower()]
                for i, dog in enumerate(dog_preds[:3]):
                    stats_text += f"{i+1}. {dog['class']}: {dog['percentage']:.1f}%\n"
            
            ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
            
            plt.suptitle('1000-Category ImageNet Classification', fontsize=18, y=1.02)
            plt.tight_layout()
            
            # Convert to base64
            plot_base64 = self._plot_to_base64(fig, is_matplotlib=True)
            if plot_base64:
                self.visualizations.append({
                    'type': 'image',
                    'name': '1000category_classification_results',
                    'title': f'1000 Categories - {model_name}',
                    'content': plot_base64,
                    'description': f'Image classification into 1000 ImageNet categories'
                })
            
            plt.close(fig)
            
        except Exception as e:
            print(f"Error creating 1000-category visualization: {e}")
    
    def run_all_tasks(self, image_path: str) -> Dict[str, Any]:
        """Run all screenshot tasks on a single image"""
        print(f"\n{'='*60}")
        print(f"RUNNING ALL SCREENSHOT TASKS ON: {image_path}")
        print(f"{'='*60}")
        
        all_results = {}
        
        try:
            # Task 1: 0-9 Digit Classification
            print(f"\n1️⃣  TASK 1: 0-9 Digit Classification")
            all_results['digit_classification'] = self.classify_digits_0_9(image_path)
            
            # Task 2: Sign Present or Not
            print(f"\n2️⃣  TASK 2: Sign Present or Not")
            all_results['sign_detection'] = self.detect_sign_present(image_path)
            
            # Task 3/4: 1000 Category Classification (includes dog detection)
            print(f"\n3️⃣  TASK 3/4: 1000 Category Classification")
            all_results['1000_category'] = self.classify_1000_categories(image_path)
            
            # Summary
            print(f"\n{'='*60}")
            print("SUMMARY:")
            print(f"  Digit Classification: {all_results['digit_classification'].get('predicted_digit', 'N/A')}")
            print(f"  Sign Present: {all_results['sign_detection'].get('sign_present', False)}")
            print(f"  Top 1000-Category: {all_results['1000_category'].get('top_prediction', {}).get('class', 'N/A')}")
            print(f"  Dog Detected: {all_results['1000_category'].get('is_dog', False)}")
            print(f"{'='*60}")
            
            return all_results
            
        except Exception as e:
            print(f"Error running all tasks: {e}")
            traceback.print_exc()
            return {"error": str(e)}
    
    def classify_with_pretrained(self, image_path: str, model_name: str = "MobileNetV2") -> Dict[str, Any]:
        """Original method for backward compatibility"""
        return self.classify_1000_categories(image_path, model_name)
    
    def _plot_to_base64(self, plt_fig=None, is_matplotlib=True):
        """Convert matplotlib figure to base64"""
        try:
            if is_matplotlib:
                buf = BytesIO()
                if plt_fig:
                    plt_fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                else:
                    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                    plt.close()
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                return img_base64
            else:
                return None
        except Exception as e:
            print(f"Error converting plot to base64: {e}")
            return None
    
    def get_image_statistics(self, image_path: str) -> Dict[str, Any]:
        """Get basic image statistics"""
        try:
            img = self.load_image(image_path)
            
            stats = {
                'path': image_path,
                'shape': img.shape,
                'height': img.shape[0],
                'width': img.shape[1],
                'channels': img.shape[2] if len(img.shape) > 2 else 1,
                'dtype': str(img.dtype),
                'min_value': float(img.min()),
                'max_value': float(img.max()),
                'mean_value': float(img.mean()),
                'std_value': float(img.std()),
                'size_kb': os.path.getsize(image_path) / 1024 if os.path.exists(image_path) else None
            }
            
            # Create histogram visualization
            if len(img.shape) == 3:  # Color image
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                colors = ['Red', 'Green', 'Blue']
                for i in range(3):
                    axes[i].hist(img[:, :, i].ravel(), bins=50, color=colors[i].lower(), alpha=0.7)
                    axes[i].set_title(f'{colors[i]} Channel Histogram', fontsize=12)
                    axes[i].set_xlabel('Pixel Value')
                    axes[i].set_ylabel('Frequency')
                    axes[i].grid(True, alpha=0.3)
            else:  # Grayscale image
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(img.ravel(), bins=50, color='gray', alpha=0.7)
                ax.set_title('Grayscale Histogram', fontsize=14)
                ax.set_xlabel('Pixel Value')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
            
            plt.suptitle('Image Statistics', fontsize=16, y=1.05)
            plt.tight_layout()
            
            # Convert to base64
            plot_base64 = self._plot_to_base64(fig, is_matplotlib=True)
            if plot_base64:
                self.visualizations.append({
                    'type': 'image',
                    'name': 'image_statistics',
                    'title': 'Image Statistics',
                    'content': plot_base64,
                    'description': 'Image histogram and basic statistics'
                })
            
            plt.close(fig)
            
            return stats
            
        except Exception as e:
            raise Exception(f"Error getting image statistics: {str(e)}")



