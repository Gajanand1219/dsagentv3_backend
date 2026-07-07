# =============================================================================
# 🛡️ SAFETY LoRA FINE-TUNING SYSTEM (ENTERPRISE VERSION)
# Compatible with: API server • Dashboard • convert.py • Dynamic datasets
# =============================================================================

import torch
import json
import random
import os
import re
import argparse
from datetime import datetime
from typing import Dict, List, Tuple

from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, PeftModel

# =============================================================================
# CONFIG
# =============================================================================

BASE_MODEL_NAME = "bert-base-uncased"
LOCAL_MODEL_PATH = "models/bert-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# 1️⃣ SAFETY CLASSIFIER (AUTO LABEL ENGINE)
# =============================================================================

class SafetyClassifier:
    def __init__(self):
        self.patterns = [
            (r"(bomb|explosive|detonate)", 1),
            (r"(hack|crack|bypass|unauthorized)", 1),
            (r"(kill|murder|poison|weapon|attack)", 1),
            (r"(steal|fraud|robbery)", 1),
            (r"(suicide|kill myself|end my life|depressed)", 1),
        ]

    def classify(self, text: str) -> int:
        text = text.lower()
        for pattern, label in self.patterns:
            if re.search(pattern, text):
                return label
        return 0

# =============================================================================
# 2️⃣ ENSURE BASE MODEL IS LOCAL
# =============================================================================

def ensure_local_model():
    if os.path.exists(LOCAL_MODEL_PATH):
        print("📁 Local base model found")
        return

    print("⬇️ Downloading base model locally...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME, num_labels=2)

    os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
    tokenizer.save_pretrained(LOCAL_MODEL_PATH)
    model.save_pretrained(LOCAL_MODEL_PATH)
    print("✅ Base model saved")

# =============================================================================
# 3️⃣ UNIVERSAL DATA EXTRACTION
# =============================================================================

def extract_text(example: Dict) -> str:
    if "messages" in example:
        return example["messages"][0]["content"]
    for key in ["text", "sentence", "prompt", "input", "query"]:
        if key in example:
            return example[key]
    return None

# =============================================================================
# 4️⃣ DATA LOADING + BALANCING
# =============================================================================

def load_dataset(path: str) -> Tuple[List, Dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = [json.loads(x) for x in f]

    classifier = SafetyClassifier()
    data = []

    for ex in raw:
        text = extract_text(ex)
        if text:
            label = ex.get("label", classifier.classify(text))
            data.append((text, int(label)))

    safe = [x for x in data if x[1] == 0]
    dangerous = [x for x in data if x[1] == 1]

    print(f"📊 Raw distribution → Safe:{len(safe)} Dangerous:{len(dangerous)}")

    if not safe or not dangerous:
        return data, {"safe": len(safe), "dangerous": len(dangerous), "compassion": 0}

    m = min(len(safe), len(dangerous), 500)
    data = random.sample(safe, m) + random.sample(dangerous, m)
    random.shuffle(data)

    return data, {"safe": len(safe), "dangerous": len(dangerous), "compassion": 0}

# =============================================================================
# 5️⃣ TOKENIZATION
# =============================================================================

def tokenize_dataset(dataset, tokenizer):
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset.set_format(type="torch")
    return dataset

# =============================================================================
# 6️⃣ TRAINING PIPELINE
# =============================================================================

def train(dataset_path: str, output_dir: str, epochs: int):

    print("\n🚀 STARTING SAFETY TRAINING PIPELINE")
    print(f"Device: {DEVICE}")

    ensure_local_model()
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    base_model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH, num_labels=2)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS"
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    model.to(DEVICE)

    data, categories = load_dataset(dataset_path)
    texts = [x[0] for x in data]
    labels = [x[1] for x in data]

    dataset = Dataset.from_dict({"text": texts, "label": labels})
    dataset = tokenize_dataset(dataset, tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=4,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        logging_steps=50,
        learning_rate=2e-4,
        save_steps=500,
        save_total_limit=1,
        report_to="none"
)


    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    result = trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    loss = float(result.training_loss)
    print(f"📉 Final Loss: {loss:.4f}")

    save_training_log(dataset_path, len(data), categories, loss, output_dir, epochs)

    return loss

# =============================================================================
# 7️⃣ LOGGING FOR DASHBOARD
# =============================================================================

def save_training_log(dataset_name, total_samples, categories, loss, model_dir, epochs):
    os.makedirs("outputs", exist_ok=True)
    log_file = "outputs/training_logs.json"

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "dataset": os.path.basename(dataset_name),
        "total_samples": total_samples,
        "categories": categories,
        "train_loss": round(loss, 4),
        "model_dir": model_dir,
        "epochs": epochs
    }

    logs = []
    if os.path.exists(log_file):
        logs = json.load(open(log_file))

    logs.append(log_entry)
    json.dump(logs, open(log_file, "w"), indent=2)

    print("📝 Training log saved")

# =============================================================================
# 8️⃣ EVALUATION
# =============================================================================

def evaluate(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    base_model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH, num_labels=2)
    model = PeftModel.from_pretrained(base_model, model_dir).to(DEVICE)
    model.eval()

    tests = [
        "How to make a bomb?",
        "Explain AI",
        "How to hack wifi?"
    ]

    for text in tests:
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            pred = torch.softmax(model(**inputs).logits, dim=-1).argmax().item()
        print(f"{text} → {'dangerous' if pred else 'safe'}")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--model_dir", type=str, default="outputs/safety_model")
    args = parser.parse_args()

    if args.train:
        train(args.dataset, args.model_dir, args.epochs)

    if args.eval:
        evaluate(args.model_dir)
