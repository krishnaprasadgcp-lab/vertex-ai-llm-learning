#!/usr/bin/env python
"""
Load and convert public Google/HuggingFace datasets into Vertex AI training format.

Suggested public datasets for LLM fine-tuning learning:

  TASK                  DATASET                     SIZE      NOTES
  ──────────────────────────────────────────────────────────────────────────
  Instruction following google/flan (flan2022)      ~15M      Google-created
  Q&A                   google-research-datasets/    90K      Google Natural Qs
                        natural_questions
  Summarization         EdinburghNLP/xsum            227K     News summarization
  Classification        fancyzhx/ag_news             127K     4-class news
  Q&A (extractive)      rajpurkar/squad_v2           130K     Wikipedia Q&A
  Conversation          HuggingFaceH4/ultrachat_200k 200K     GPT-4 dialogs
  General instruct      yahma/alpaca-cleaned         52K      Stanford Alpaca
  ──────────────────────────────────────────────────────────────────────────
  All are free, publicly licensed, and easily loaded with `datasets`.
"""

import sys
import json
sys.path.insert(0, ".")

from datasets import load_dataset
from src.data_prep import DataPreparationPipeline


# ---------------------------------------------------------------------------
# Loaders — each returns a list of Gemini-format dicts ready for save_to_jsonl
# ---------------------------------------------------------------------------

def load_alpaca(max_samples: int = 200) -> list:
    """
    Stanford Alpaca (cleaned) — 52K instruction-following examples.
    Fields: instruction, input, output
    Great for: general instruction following, a natural first fine-tuning task.
    """
    print("Loading yahma/alpaca-cleaned ...")
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    examples = []
    for row in ds.select(range(min(max_samples, len(ds)))):
        user_msg = row["instruction"]
        if row["input"]:
            user_msg = f"{row['instruction']}\n\n{row['input']}"
        examples.append(
            DataPreparationPipeline.create_training_example(
                user_message=user_msg,
                model_response=row["output"],
            )
        )
    print(f"  Loaded {len(examples)} examples")
    return examples


def load_ag_news(max_samples: int = 200) -> list:
    """
    AG News — 4-class news topic classification.
    Classes: World, Sports, Business, Sci/Tech
    Great for: text classification fine-tuning demo.
    """
    LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    SYSTEM = (
        "Classify the news headline into one of these categories: "
        "World, Sports, Business, Sci/Tech. Reply with the category only."
    )
    print("Loading fancyzhx/ag_news ...")
    ds = load_dataset("fancyzhx/ag_news", split="train")
    examples = []
    for row in ds.select(range(min(max_samples, len(ds)))):
        examples.append(
            DataPreparationPipeline.create_training_example(
                user_message=row["text"],
                model_response=LABEL_MAP[row["label"]],
                system_instruction=SYSTEM,
            )
        )
    print(f"  Loaded {len(examples)} examples")
    return examples


def load_xsum(max_samples: int = 200) -> list:
    """
    XSum (Extreme Summarization) — one-sentence BBC article summaries.
    Great for: summarization fine-tuning.
    """
    SYSTEM = "Summarize the following news article in one concise sentence."
    print("Loading EdinburghNLP/xsum ...")
    ds = load_dataset("EdinburghNLP/xsum", split="train", trust_remote_code=True)
    examples = []
    for row in ds.select(range(min(max_samples, len(ds)))):
        examples.append(
            DataPreparationPipeline.create_training_example(
                user_message=row["document"],
                model_response=row["summary"],
                system_instruction=SYSTEM,
            )
        )
    print(f"  Loaded {len(examples)} examples")
    return examples


def load_natural_questions(max_samples: int = 200) -> list:
    """
    Google Natural Questions — real Google Search questions with Wikipedia answers.
    Great for: open-domain Q&A fine-tuning.
    """
    SYSTEM = "Answer the following question concisely based on your knowledge."
    print("Loading google-research-datasets/natural_questions ...")
    ds = load_dataset(
        "google-research-datasets/natural_questions",
        split="train",
        trust_remote_code=True,
    )
    examples = []
    count = 0
    for row in ds:
        if count >= max_samples:
            break
        # Use the short answer if available
        annotations = row.get("annotations", {})
        short_answers = annotations.get("short_answers", [])
        if not short_answers or not short_answers[0].get("text"):
            continue
        answer_text = short_answers[0]["text"][0] if short_answers[0]["text"] else ""
        if not answer_text:
            continue
        question = row["question"]["text"]
        examples.append(
            DataPreparationPipeline.create_training_example(
                user_message=question,
                model_response=answer_text,
                system_instruction=SYSTEM,
            )
        )
        count += 1
    print(f"  Loaded {len(examples)} examples")
    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DATASET_OPTIONS = {
    "alpaca":            (load_alpaca,            "General instruction following — best starting point"),
    "ag_news":           (load_ag_news,           "News topic classification (4 classes)"),
    "xsum":              (load_xsum,              "One-sentence news article summarization"),
    "natural_questions": (load_natural_questions, "Google real-world Q&A from Search"),
}


def main():
    print("=" * 64)
    print("Public Dataset Loader for Vertex AI Fine-tuning")
    print("=" * 64)
    print("\nAvailable datasets:\n")
    for key, (_, desc) in DATASET_OPTIONS.items():
        print(f"  {key:<22} {desc}")

    print("\nLoading 'alpaca' as the default example (200 samples)...")
    print("(Edit this script to switch datasets or increase max_samples)\n")

    examples = load_alpaca(max_samples=200)

    train_examples, val_examples = DataPreparationPipeline.split_dataset(
        examples, train_ratio=0.8
    )
    DataPreparationPipeline.save_to_jsonl(train_examples, "./data/training_data.jsonl")
    DataPreparationPipeline.save_to_jsonl(val_examples,   "./data/validation_data.jsonl")

    print(f"\nSaved {len(train_examples)} training + {len(val_examples)} validation examples")
    print("\nSample record (Gemini format):")
    print(json.dumps(train_examples[0], indent=2))

    print("\n" + "=" * 64)
    print("Next step: upload to GCS")
    print("  gsutil cp data/training_data.jsonl   gs://gen-lang-client-0791883042-vertex-ai-training/data/")
    print("  gsutil cp data/validation_data.jsonl gs://gen-lang-client-0791883042-vertex-ai-training/data/")
    print("Then run: python src/scripts/04_submit_tuning_job.py")


if __name__ == "__main__":
    main()
