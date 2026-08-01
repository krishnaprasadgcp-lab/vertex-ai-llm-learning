#!/usr/bin/env python
"""Example: Prepare training data for LLM fine-tuning."""

import sys
sys.path.insert(0, '.')

from src.data_prep import DataPreparationPipeline


def create_sentiment_training_data():
    """Create example sentiment classification training data."""

    print("Creating sentiment classification training data...")

    raw_examples = [
        ("This movie is absolutely fantastic!", "Positive"),
        ("I loved every moment of this film!", "Positive"),
        ("Amazing performance and great storyline.", "Positive"),
        ("This is the best experience ever!", "Positive"),
        ("Wonderful and entertaining throughout.", "Positive"),
        ("This movie is terrible and boring.", "Negative"),
        ("Worst film I've ever seen.", "Negative"),
        ("Absolutely awful, waste of time.", "Negative"),
        ("Disappointing and not worth watching.", "Negative"),
        ("Horrible acting and poor plot.", "Negative"),
        ("The movie was okay, nothing special.", "Neutral"),
        ("Average film with some good parts.", "Neutral"),
        ("It was fine but forgettable.", "Neutral"),
        ("Neither good nor bad.", "Neutral"),
        ("Some scenes were good, others not so much.", "Neutral"),
    ]

    SYSTEM = "Classify the sentiment of the text as Positive, Negative, or Neutral. Reply with one word only."

    # Convert to Gemini multi-turn format
    examples = [
        DataPreparationPipeline.create_training_example(
            user_message=text,
            model_response=label,
            system_instruction=SYSTEM,
        )
        for text, label in raw_examples
    ]

    # Split into training and validation
    train_examples, val_examples = DataPreparationPipeline.split_dataset(examples, train_ratio=0.8)

    DataPreparationPipeline.save_to_jsonl(train_examples, "./data/training_data.jsonl")
    DataPreparationPipeline.save_to_jsonl(val_examples, "./data/validation_data.jsonl")

    # Validate
    print("\nValidating data format...")
    for path in ("./data/training_data.jsonl", "./data/validation_data.jsonl"):
        ok = DataPreparationPipeline.validate_jsonl(path)
        print(f"  {path}: {'OK' if ok else 'INVALID'}")

    print(f"\nTraining examples : {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")

    print("\nSample example (Gemini format):")
    import json
    print(json.dumps(train_examples[0], indent=2))


if __name__ == "__main__":
    create_sentiment_training_data()

