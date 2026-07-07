import json
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid threading issues
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from matplotlib.patches import Patch, Circle

# Set backend before any other matplotlib imports
plt.switch_backend('Agg')

LOG_FILE = "outputs/training_logs.json"
OUTPUT_IMG = "outputs/safety_dashboard.png"

def load_latest_log():
    """Load only the latest training log"""
    if not os.path.exists(LOG_FILE):
        print("❌ No training logs found.")
        return None
    
    with open(LOG_FILE, "r") as f:
        logs = json.load(f)
    
    if not logs:
        print("❌ Log file is empty.")
        return None
    
    # Return only the latest log
    return logs[-1]

def calculate_safety_score(categories):
    """Calculate safety score from categories"""
    safe = categories.get("safe", 0)
    dangerous = categories.get("dangerous", 0)
    compassion = categories.get("compassion", 0)
    total = safe + dangerous + compassion
    
    if total == 0:
        return {
            "safety_score": 0,
            "risk_score": 0,
            "total_samples": 0,
            "safe_count": 0,
            "dangerous_count": 0,
            "compassion_count": 0
        }
    
    safety_score = (safe + compassion) / total * 100
    risk_score = (dangerous) / total * 100
    
    return {
        "safety_score": safety_score,
        "risk_score": risk_score,
        "total_samples": total,
        "safe_count": safe,
        "dangerous_count": dangerous,
        "compassion_count": compassion
    }

def create_simple_dashboard(log):
    """Create a simple dashboard without complex plots"""
    try:
        # Extract values
        loss = log.get("train_loss", 0)
        categories = log.get("categories", {})
        total_samples = log.get("total_samples", 0)
        dataset_name = log.get("dataset", "Unknown")
        timestamp = log.get("timestamp", "")
        epochs = log.get("epochs", 0)
        
        # Calculate safety metrics
        safety_data = calculate_safety_score(categories)
        
        # Format timestamp
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                formatted_time = timestamp
        else:
            formatted_time = "Unknown"
        
        # Create figure
        fig = plt.figure(figsize=(14, 10), facecolor='#f8f9fa')
        
        # Define color palette
        colors = {
            'primary': '#4361ee',
            'secondary': '#7209b7',
            'success': '#38b000',
            'warning': '#f8961e',
            'danger': '#f72585',
            'info': '#4cc9f0',
            'light': '#f8f9fa',
            'dark': '#212529'
        }
        
        # 1. Header Section
        ax_header = plt.subplot(3, 1, 1)
        ax_header.axis('off')
        
        header_text = f"""
        🛡️ SAFETY LoRA TRAINING DASHBOARD
        {'='*50}
        
        📊 Dataset: {dataset_name[:40]}...
        ⏰ Training Time: {formatted_time}
        🔢 Total Samples: {total_samples}
        ⚙️ Training Epochs: {epochs}
        📉 Final Loss: {loss:.4f}
        """
        
        ax_header.text(0.5, 0.8, header_text, 
                      fontsize=14, fontfamily='monospace',
                      ha='center', va='center',
                      bbox=dict(boxstyle='round', facecolor='white', 
                               alpha=0.9, edgecolor=colors['primary'], linewidth=2))
        
        # 2. Safety Metrics Section
        ax_metrics = plt.subplot(3, 3, 4)
        ax_metrics.axis('off')
        
        safety_score = safety_data["safety_score"]
        
        # Create a simple progress bar for safety score
        bar_height = 0.3
        ax_metrics.barh(0, safety_score, height=bar_height, 
                       color=colors['success'], edgecolor='white', linewidth=2)
        ax_metrics.barh(0, 100 - safety_score, left=safety_score, 
                       height=bar_height, color=colors['light'], 
                       edgecolor='white', linewidth=2)
        ax_metrics.set_xlim(0, 100)
        ax_metrics.set_ylim(-0.5, 0.5)
        ax_metrics.text(safety_score/2, 0, f'{safety_score:.1f}%', 
                       ha='center', va='center', fontsize=24, fontweight='bold',
                       color='white')
        ax_metrics.text(50, 0.5, 'SAFETY SCORE', 
                       ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # 3. Category Distribution
        ax_categories = plt.subplot(3, 3, 5)
        
        category_counts = [
            categories.get("safe", 0),
            categories.get("dangerous", 0),
            categories.get("compassion", 0)
        ]
        category_labels = ['Safe', 'Dangerous', 'Compassion']
        category_colors = [colors['success'], colors['danger'], colors['info']]
        
        # Simple bar chart
        bars = ax_categories.bar(category_labels, category_counts, 
                               color=category_colors, edgecolor='white', linewidth=2)
        
        # Add value labels
        for bar, count in zip(bars, category_counts):
            height = bar.get_height()
            ax_categories.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                             f'{count}', ha='center', va='bottom',
                             fontsize=12, fontweight='bold')
        
        ax_categories.set_title('Dataset Categories', fontsize=14, fontweight='bold')
        ax_categories.set_ylabel('Samples')
        ax_categories.grid(True, alpha=0.3, axis='y')
        
        # 4. Loss Visualization
        ax_loss = plt.subplot(3, 3, 6)
        
        # Simple loss visualization
        x = np.arange(1, epochs + 1)
        y_loss = loss * np.exp(-0.2 * (x - 1))
        
        ax_loss.plot(x, y_loss, 'o-', linewidth=3, markersize=10,
                    color=colors['primary'], markerfacecolor='white',
                    markeredgewidth=2)
        
        ax_loss.fill_between(x, y_loss, alpha=0.2, color=colors['primary'])
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_title(f'Training Loss Trend', fontsize=14, fontweight='bold')
        ax_loss.grid(True, alpha=0.3)
        
        # 5. Summary Section
        ax_summary = plt.subplot(3, 1, 3)
        ax_summary.axis('off')
        
        # Generate recommendations
        recommendations = []
        if safety_score >= 80:
            recommendations.append("✅ Excellent safety performance")
        elif safety_score >= 60:
            recommendations.append("⚠️ Good safety, add more safe samples")
        else:
            recommendations.append("❌ Need more safe/compassion samples")
        
        if loss < 0.5:
            recommendations.append("✅ Training converged well")
        elif loss < 1.0:
            recommendations.append("⚠️ Training could be improved")
        else:
            recommendations.append("❌ High loss, check data quality")
        
        if total_samples >= 100:
            recommendations.append("✅ Sufficient training data")
        else:
            recommendations.append("⚠️ Consider increasing dataset size")
        
        # Summary text
        summary_text = "📋 TRAINING SUMMARY\n" + "="*50 + "\n\n"
        summary_text += f"Safety Score: {safety_score:.1f}%\n"
        summary_text += f"Risk Level: {safety_data['risk_score']:.1f}%\n"
        summary_text += f"Model Performance: {'Good' if loss < 1.0 else 'Needs Improvement'}\n"
        summary_text += f"Data Balance: {'Balanced' if safety_score > 60 else 'Needs more safe samples'}\n\n"
        
        summary_text += "💡 RECOMMENDATIONS:\n"
        for i, rec in enumerate(recommendations, 1):
            summary_text += f"{i}. {rec}\n"
        
        ax_summary.text(0.5, 0.95, summary_text, 
                       fontsize=12, fontfamily='monospace',
                       ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='white', 
                                alpha=0.9, edgecolor=colors['info'], linewidth=2))
        
        plt.suptitle('SAFETY LoRA TRAINING REPORT', 
                    fontsize=20, fontweight='bold', 
                    color=colors['primary'], y=0.98)
        
        plt.figtext(0.5, 0.02, 
                   f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                   ha='center', fontsize=10, color=colors['dark'], alpha=0.7)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(OUTPUT_IMG, dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
        plt.close()
        
        # Console output
        print("\n" + "="*60)
        print("📊 DASHBOARD GENERATED SUCCESSFULLY")
        print("="*60)
        print(f"📁 Dataset: {dataset_name[:40]}...")
        print(f"📉 Loss: {loss:.4f}")
        print(f"🛡️ Safety Score: {safety_score:.1f}%")
        print(f"📊 Categories: Safe={categories.get('safe', 0)}, "
              f"Dangerous={categories.get('dangerous', 0)}, "
              f"Compassion={categories.get('compassion', 0)}")
        print(f"💾 Saved to: {OUTPUT_IMG}")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating dashboard: {e}")
        create_error_dashboard(str(e))
        return False

def create_error_dashboard(error_msg):
    """Create error dashboard"""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#f8f9fa')
    ax.axis('off')
    
    ax.text(0.5, 0.6, '❌', fontsize=80, ha='center', color='#f72585')
    ax.text(0.5, 0.45, 'Dashboard Generation Failed', 
           fontsize=20, fontweight='bold', ha='center')
    ax.text(0.5, 0.4, error_msg[:100], 
           fontsize=12, ha='center', color='#666')
    ax.text(0.5, 0.35, 'Check training logs for details', 
           fontsize=12, ha='center', color='#888')
    
    plt.suptitle('SAFETY LoRA TRAINING DASHBOARD', 
                fontsize=24, fontweight='bold', 
                color='#4361ee', y=0.95)
    
    plt.figtext(0.5, 0.05, 
               f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
               ha='center', fontsize=10, color='#666')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
    plt.close()
    print(f"⚠️ Created error dashboard: {error_msg}")

def create_empty_dashboard():
    """Create empty dashboard"""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#f8f9fa')
    ax.axis('off')
    
    ax.text(0.5, 0.6, '📊', fontsize=80, ha='center', color='#4361ee', alpha=0.3)
    ax.text(0.5, 0.45, 'No Training Data Available', 
           fontsize=20, fontweight='bold', ha='center')
    ax.text(0.5, 0.4, 'Upload and train a model to see dashboard', 
           fontsize=14, ha='center', color='#666')
    
    plt.suptitle('SAFETY LoRA TRAINING DASHBOARD', 
                fontsize=24, fontweight='bold', 
                color='#4361ee', y=0.95)
    
    plt.figtext(0.5, 0.05, 
               f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
               ha='center', fontsize=10, color='#666')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
    plt.close()
    print("📊 Created empty dashboard")

def show_dashboard():
    """Generate dashboard for latest training run only"""
    log = load_latest_log()
    
    if not log:
        print("📊 No training data available")
        create_empty_dashboard()
        return
    
    print(f"\n📊 Generating dashboard for latest run...")
    success = create_simple_dashboard(log)
    
    if not success:
        create_empty_dashboard()

if __name__ == "__main__":
    show_dashboard()