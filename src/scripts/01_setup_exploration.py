#!/usr/bin/env python
"""Example: Basic Vertex AI setup and model exploration."""

import sys
sys.path.insert(0, '.')

from src.vertex_ai_trainer import VertexAITrainer
from config.settings import settings


def main():
    """Run setup exploration."""
    print("=" * 60)
    print("Vertex AI LLM Learning - Setup Exploration")
    print("=" * 60)
    
    print(f"\nProject: {settings.GCP_PROJECT_ID}")
    print(f"Region: {settings.GCP_REGION}")
    print(f"Base Model: {settings.BASE_MODEL}")
    
    # Initialize trainer
    try:
        trainer = VertexAITrainer()
        print("\n✓ Vertex AI initialized successfully")
    except Exception as e:
        print(f"\n✗ Error initializing Vertex AI: {e}")
        print("\nSetup steps:")
        print("1. Ensure gcloud CLI is installed: https://cloud.google.com/sdk/docs/install")
        print("2. Authenticate: gcloud auth application-default login")
        print("3. Set project: gcloud config set project gen-lang-client-0791883042")
        return
    
    # Get available models
    print("\nAvailable Foundation Models for Fine-tuning:")
    models = trainer.get_foundation_model_info()
    for model_name, info in models.items():
        print(f"\n  {model_name}")
        print(f"    Description: {info['description']}")
        print(f"    Best for: {info['use_case']}")
        print(f"    Max input tokens: {info['input_tokens']:,}")
        print(f"    Max output tokens: {info['output_tokens']:,}")


if __name__ == "__main__":
    main()
