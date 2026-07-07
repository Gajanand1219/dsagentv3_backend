# convert.py
import pandas as pd
import json
import os

def csv_to_jsonl(input_csv: str, output_jsonl: str) -> str:
    """
    Convert CSV file to JSONL format for training.
    Returns output JSONL path.
    """

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)

    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():

            user_text = "Here is a data record:\n"
            user_text += "\n".join(
                f"{col}: {row[col]}" for col in df.columns
            )

            record = {
                "messages": [
                    {"role": "user", "content": user_text},
                    {
                        "role": "assistant",
                        "content": "Analyze this data and learn patterns responsibly."
                    }
                ]
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ CSV → JSONL converted: {output_jsonl}")
    return output_jsonl
