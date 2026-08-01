"""Data preparation utilities for Vertex AI training."""

import json
import jsonlines
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm


class DataPreparationPipeline:
    """Pipeline for preparing data for LLM fine-tuning."""

    @staticmethod
    def create_training_example(
        user_message: str,
        model_response: str,
        system_instruction: str = "",
    ) -> Dict[str, Any]:
        """
        Create a single training example in Gemini's required multi-turn format.

        Vertex AI supervised fine-tuning expects::

            {"contents": [
                {"role": "user",  "parts": [{"text": "..."}]},
                {"role": "model", "parts": [{"text": "..."}]}
            ]}

        Args:
            user_message: The user turn (instruction + optional input)
            model_response: The expected model response
            system_instruction: Optional system prompt prepended to user turn

        Returns:
            Dict in Gemini multi-turn JSONL format
        """
        user_text = f"{system_instruction}\n\n{user_message}".strip() if system_instruction else user_message
        return {
            "contents": [
                {"role": "user",  "parts": [{"text": user_text}]},
                {"role": "model", "parts": [{"text": model_response}]},
            ]
        }

    @staticmethod
    def create_multiturn_example(
        turns: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Create a multi-turn conversation example.

        Args:
            turns: List of dicts with 'role' ('user'|'model') and 'text'

        Returns:
            Dict in Gemini multi-turn JSONL format
        """
        return {
            "contents": [
                {"role": t["role"], "parts": [{"text": t["text"]}]}
                for t in turns
            ]
        }

    @staticmethod
    def save_to_jsonl(examples: List[Dict[str, Any]], output_path: str) -> None:
        """
        Save training examples to JSONL format (required by Vertex AI).
        
        Args:
            examples: List of training examples
            output_path: Path to save JSONL file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with jsonlines.open(output_path, mode='w') as writer:
            for example in examples:
                writer.write(example)
        
        print(f"Saved {len(examples)} examples to {output_path}")

    @staticmethod
    def load_from_csv(csv_path: str) -> pd.DataFrame:
        """
        Load training data from CSV file.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            DataFrame with training data
        """
        return pd.read_csv(csv_path)

    @staticmethod
    def validate_jsonl(jsonl_path: str) -> bool:
        """
        Validate JSONL file format.
        
        Args:
            jsonl_path: Path to JSONL file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            with jsonlines.open(jsonl_path) as reader:
                for obj in reader:
                    if not isinstance(obj, dict):
                        print(f"Invalid example: {obj}")
                        return False
            return True
        except Exception as e:
            print(f"Error validating JSONL: {e}")
            return False

    @staticmethod
    def split_dataset(
        examples: List[Dict[str, Any]],
        train_ratio: float = 0.8
    ) -> tuple:
        """
        Split examples into training and validation sets.
        
        Args:
            examples: List of all examples
            train_ratio: Ratio for training set (0.8 = 80% train, 20% validation)
            
        Returns:
            Tuple of (train_examples, validation_examples)
        """
        split_idx = int(len(examples) * train_ratio)
        return examples[:split_idx], examples[split_idx:]


def example_usage():
    """Example of how to use the data preparation pipeline."""
    
    # Create some example training data
    examples = [
        DataPreparationPipeline.create_training_example(
            instruction="Classify the sentiment of the following text",
            input_text="I love this product! It's amazing!",
            output="Positive"
        ),
        DataPreparationPipeline.create_training_example(
            instruction="Classify the sentiment of the following text",
            input_text="This is terrible and doesn't work.",
            output="Negative"
        ),
    ]
    
    # Save to JSONL
    DataPreparationPipeline.save_to_jsonl(examples, "./data/training_data.jsonl")
    
    # Validate
    is_valid = DataPreparationPipeline.validate_jsonl("./data/training_data.jsonl")
    print(f"JSONL valid: {is_valid}")
    
    # Split dataset
    train, val = DataPreparationPipeline.split_dataset(examples)
    print(f"Training examples: {len(train)}, Validation examples: {len(val)}")


if __name__ == "__main__":
    example_usage()
