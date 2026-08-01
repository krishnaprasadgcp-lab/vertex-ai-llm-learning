"""Vertex AI training utilities for LLM fine-tuning."""

from typing import Optional, List
import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.tuning import sft
from google.cloud import aiplatform
from config.settings import settings


class VertexAITrainer:
    """Trainer class for Vertex AI LLM fine-tuning."""

    def __init__(self, project_id: str = None, region: str = None):
        """
        Initialize Vertex AI trainer.
        
        Args:
            project_id: GCP project ID
            region: GCP region
        """
        self.project_id = project_id or settings.GCP_PROJECT_ID
        self.region = region or settings.GCP_REGION
        
        # Initialize Vertex AI
        vertexai.init(project=self.project_id, location=self.region)
        aiplatform.init(project=self.project_id, location=self.region)

    def get_foundation_model_info(self) -> dict:
        """
        Get information about available foundation models for fine-tuning.
        These are the model IDs supported by Vertex AI supervised tuning (SFT).
        """
        return {
            "gemini-2.0-flash-001": {
                "description": "Latest Gemini Flash — fast, efficient, tunable",
                "use_case": "Recommended starting point for fine-tuning",
                "input_tokens": 1000000,
                "output_tokens": 8192,
                "tuning_support": True,
            },
            "gemini-1.5-pro-002": {
                "description": "High-capacity model for complex tasks",
                "use_case": "Complex reasoning, long context",
                "input_tokens": 2000000,
                "output_tokens": 8192,
                "tuning_support": True,
            },
            "gemini-1.5-flash-002": {
                "description": "Balanced speed and quality",
                "use_case": "Quick responses, cost-efficient tasks",
                "input_tokens": 1000000,
                "output_tokens": 8192,
                "tuning_support": True,
            },
        }

    def submit_tuning_job(
        self,
        training_data_uri: str,
        validation_data_uri: Optional[str] = None,
        tuned_model_name: str = "my-gemini-finetuned-model",
        source_model: Optional[str] = None,
        epochs: int = 3,
        learning_rate_multiplier: float = 1.0,
        adapter_size: int = 4,
    ):
        """
        Submit a supervised fine-tuning (SFT) job to Vertex AI.

        Training data must be a GCS URI (gs://bucket/file.jsonl) in the
        Gemini multi-turn format::

            {"contents": [
                {"role": "user",  "parts": [{"text": "..."}]},
                {"role": "model", "parts": [{"text": "..."}]}
            ]}

        Args:
            training_data_uri: GCS URI to training JSONL
            validation_data_uri: GCS URI to validation JSONL (optional)
            tuned_model_name: Display name for the resulting model
            source_model: Base model ID (defaults to settings.BASE_MODEL)
            epochs: Number of training epochs (1-5 recommended)
            learning_rate_multiplier: Multiplier on the base learning rate
            adapter_size: LoRA adapter size (1, 2, 4, 8, or 16)

        Returns:
            sft.SupervisedTuningJob instance
        """
        base = source_model or settings.BASE_MODEL
        print(f"Submitting SFT job: {tuned_model_name}")
        print(f"  Base model : {base}")
        print(f"  Train data : {training_data_uri}")
        print(f"  Epochs     : {epochs}")

        tuning_job = sft.train(
            source_model=base,
            train_dataset=training_data_uri,
            validation_dataset=validation_data_uri,
            epochs=epochs,
            learning_rate_multiplier=learning_rate_multiplier,
            adapter_size=adapter_size,
            tuned_model_display_name=tuned_model_name,
        )
        print(f"Job resource: {tuning_job.resource_name}")
        return tuning_job

    def get_tuning_job_status(self, job_resource_name: str):
        """Poll a running tuning job for its current state."""
        job = sft.SupervisedTuningJob(job_resource_name)
        print(f"State  : {job.state}")
        print(f"Model  : {job.tuned_model_name}")
        return job

    def list_models(self) -> List[str]:
        """
        List available models in the project.
        
        Returns:
            List of model names
        """
        try:
            models = aiplatform.Model.list()
            return [model.display_name for model in models]
        except Exception as e:
            print(f"Error listing models: {e}")
            return []

    def get_model_info(self, model_name: str) -> dict:
        """
        Get information about a trained model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information
        """
        try:
            models = aiplatform.Model.list(
                filter=f'display_name="{model_name}"'
            )
            if models:
                model = models[0]
                return {
                    "name": model.display_name,
                    "resource_name": model.resource_name,
                    "update_time": model.update_time,
                    "create_time": model.create_time
                }
            return None
        except Exception as e:
            print(f"Error getting model info: {e}")
            return None

    def deploy_model(self, model_name: str, endpoint_name: str = None) -> dict:
        """
        Deploy a trained model to an endpoint.
        
        Args:
            model_name: Name of the model to deploy
            endpoint_name: Name for the endpoint (optional)
            
        Returns:
            Dictionary with endpoint details
        """
        print(f"Deploying model: {model_name}")
        
        if endpoint_name is None:
            endpoint_name = f"{model_name}-endpoint"
        
        deployment_config = {
            "model_name": model_name,
            "endpoint_name": endpoint_name,
            "machine_type": "n1-standard-2",
            "min_replica_count": 1,
            "max_replica_count": 3
        }
        
        return deployment_config

    def make_prediction(self, model_name: str, prompt: str) -> str:
        """
        Make a prediction using a deployed model.
        
        Args:
            model_name: Name of the model
            prompt: Input prompt for prediction
            
        Returns:
            Model prediction/response
        """
        try:
            model = GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None


def example_usage():
    """Example of how to use the Vertex AI trainer."""
    
    # Initialize trainer
    trainer = VertexAITrainer()
    
    # Get foundation model info
    models = trainer.get_foundation_model_info()
    print("\nAvailable Foundation Models:")
    for model_name, info in models.items():
        print(f"\n{model_name}:")
        print(f"  Description: {info['description']}")
        print(f"  Use Case: {info['use_case']}")
    
    # Create training pipeline (simulation)
    training_config = trainer.create_training_pipeline(
        training_data_path="gs://my-bucket/training_data.jsonl",
        model_name="sentiment-classifier",
        epochs=3,
        batch_size=16
    )
    print(f"\nTraining config: {training_config}")


if __name__ == "__main__":
    example_usage()
