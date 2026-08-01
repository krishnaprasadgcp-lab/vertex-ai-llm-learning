#!/usr/bin/env python
"""
Submit a supervised fine-tuning (SFT) job to Vertex AI.

Prerequisites:
  1. gcloud auth application-default login
  2. Training data uploaded to GCS (run 03_load_public_data.py, then gsutil cp)
  3. Vertex AI API enabled in your project
"""

import sys
sys.path.insert(0, ".")

from src.vertex_ai_trainer import VertexAITrainer
from config.settings import settings


def main():
    print("=" * 60)
    print("Vertex AI Supervised Fine-Tuning Job Submission")
    print("=" * 60)
    print(f"\nProject : {settings.GCP_PROJECT_ID}")
    print(f"Region  : {settings.GCP_REGION}")
    print(f"Model   : {settings.BASE_MODEL}")
    print(f"Train   : {settings.GCS_TRAINING_DATA_URI}")
    print(f"Val     : {settings.GCS_VALIDATION_DATA_URI}")

    trainer = VertexAITrainer()

    # Uncomment to submit the actual job:
    # tuning_job = trainer.submit_tuning_job(
    #     training_data_uri=settings.GCS_TRAINING_DATA_URI,
    #     validation_data_uri=settings.GCS_VALIDATION_DATA_URI,
    #     tuned_model_name=settings.MODEL_DISPLAY_NAME,
    #     epochs=settings.EPOCHS,
    #     learning_rate_multiplier=settings.LEARNING_RATE_MULTIPLIER,
    #     adapter_size=settings.ADAPTER_SIZE,
    # )
    # print(f"\nJob submitted: {tuning_job.resource_name}")
    # print("Monitor at: https://console.cloud.google.com/vertex-ai/training/training-pipelines"
    #       f"?project={settings.GCP_PROJECT_ID}")

    print("\n[DRY RUN] Job NOT submitted.")
    print("To submit, uncomment the code block in this script.")
    print("\nExpected training time: 10-30 min for small datasets (<500 rows)")


if __name__ == "__main__":
    main()
