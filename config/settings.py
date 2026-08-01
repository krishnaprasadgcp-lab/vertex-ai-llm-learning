"""Configuration settings for Vertex AI training project."""

from pydantic_settings import BaseSettings
from pydantic import computed_field


class Settings(BaseSettings):
    """Application settings loaded from .env or environment variables."""

    # GCP Configuration
    GCP_PROJECT_ID: str = "gen-lang-client-0791883042"
    GCP_REGION: str = "us-west1"

    # Vertex AI Configuration
    VERTEX_AI_LOCATION: str = "us-west1"
    MODEL_DISPLAY_NAME: str = "my-gemini-finetuned-model"

    # Dataset Configuration
    TRAINING_DATA_PATH: str = "./data/training_data.jsonl"
    VALIDATION_DATA_PATH: str = "./data/validation_data.jsonl"

    # Training Configuration
    LEARNING_RATE_MULTIPLIER: float = 1.0  # Multiplier on base LR (Gemini tuning)
    EPOCHS: int = 3
    ADAPTER_SIZE: int = 4  # LoRA adapter size: 1, 2, 4, 8, 16

    # Model Configuration — use versioned model IDs for tuning
    # Options: gemini-1.5-pro-002, gemini-1.5-flash-002, gemini-2.0-flash-001
    BASE_MODEL: str = "gemini-2.0-flash-001"

    # GCS Configuration
    GCS_BUCKET: str = "gen-lang-client-0791883042-vertex-ai-training"

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
