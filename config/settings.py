"""Configuration settings for Vertex AI training project."""

from pydantic_settings import BaseSettings
from pydantic import computed_field


class Settings(BaseSettings):
    """Application settings loaded from .env or environment variables."""

    # GCP Configuration
    GCP_PROJECT_ID: str = "gen-lang-client-0791883042"
    GCP_REGION: str = "us-central1"  # SFT only supported in us-central1

    # Vertex AI Configuration
    VERTEX_AI_LOCATION: str = "us-central1"
    MODEL_DISPLAY_NAME: str = "my-gemini-finetuned-model"

    # Dataset Configuration
    TRAINING_DATA_PATH: str = "./data/training_data.jsonl"
    VALIDATION_DATA_PATH: str = "./data/validation_data.jsonl"

    # Training Configuration
    LEARNING_RATE_MULTIPLIER: float = 1.0  # Multiplier on base LR (Gemini tuning)
    EPOCHS: int = 3
    ADAPTER_SIZE: int = 4  # LoRA adapter size: 1, 2, 4, 8, 16

    # Model Configuration — fully-qualified resource name required for SFT
    # Use publishers/google/models/<model> format
    BASE_MODEL: str = "publishers/google/models/gemini-2.5-flash"

    # GCS Configuration — bucket must be in same region as tuning job
    GCS_BUCKET: str = "gen-lang-client-0791883042-sft-central1"

    @computed_field
    @property
    def GCS_OUTPUT_PATH(self) -> str:
        return f"gs://{self.GCS_BUCKET}/training-output"

    @computed_field
    @property
    def GCS_TRAINING_DATA_URI(self) -> str:
        return f"gs://{self.GCS_BUCKET}/data/training_data.jsonl"

    @computed_field
    @property
    def GCS_VALIDATION_DATA_URI(self) -> str:
        return f"gs://{self.GCS_BUCKET}/data/validation_data.jsonl"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Global settings instance
settings = Settings()
