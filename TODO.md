"""
TODO: Vertex AI LLM Learning Project

This document outlines the learning path for fine-tuning LLMs using Vertex AI on GCP.
Project: gen-lang-client-0791883042
Region: us-west1
"""

## Phase 1: Environment Setup ✓
- [x] Create project structure
- [ ] Copy .env.example to .env and update with your credentials
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Authenticate with GCP: `gcloud auth application-default login`
- [ ] Set GCP project: `gcloud config set project gen-lang-client-0791883042`
- [ ] Verify Vertex AI API is enabled in GCP Console

## Phase 2: Understand Vertex AI Basics
- [ ] Read: https://cloud.google.com/vertex-ai/docs/start/introduction-unified-platform
- [ ] Read: Vertex AI LLM fine-tuning documentation
- [ ] Explore GCP Console: Go to Vertex AI > Training > Training Pipelines
- [ ] Review foundation model options:
  - [ ] Gemini 1.5 Pro (better for complex reasoning)
  - [ ] Gemini 1.5 Flash (faster, more efficient)
- [ ] Run: `python src/scripts/01_setup_exploration.py`

## Phase 3: Data Preparation
- [ ] Understand Vertex AI data format (JSONL)
- [ ] Review data requirements: https://cloud.google.com/vertex-ai/docs/training/create-training-pipeline/tuning-data-requirements
- [ ] Run example: `python src/scripts/02_prepare_data.py`
- [ ] Prepare your own training dataset:
  - [ ] Collect/create training examples (50-500+ recommended for good results)
  - [ ] Format as JSONL (text_input, output)
  - [ ] Create validation set (15-20% of data)
  - [ ] Upload to GCS bucket
- [ ] Validate data quality:
  - [ ] Check format compliance
  - [ ] Verify example diversity
  - [ ] Test data loading

## Phase 4: GCS Setup (Storage)
- [ ] Create GCS bucket: `gsutil mb gs://gen-lang-client-0791883042-vertex-ai-training`
- [ ] Upload training data to GCS
- [ ] Set appropriate IAM permissions
- [ ] Understand GCS paths and access control

## Phase 5: Create First Training Pipeline
- [ ] Review training configuration options
- [ ] Update training parameters in config/settings.py
- [ ] Create training job using Vertex AI Console
- [ ] Monitor training progress:
  - [ ] Check job status
  - [ ] View loss curves
  - [ ] Monitor resource usage
- [ ] (Alternative) Create via Python API: src/scripts/03_create_training_job.py

## Phase 6: Model Evaluation
- [ ] Download training metrics
- [ ] Analyze loss curves and validation metrics
- [ ] Test model with sample prompts
- [ ] Evaluate output quality
- [ ] Compare with base model

## Phase 7: Deploy Model
- [ ] Create endpoint in Vertex AI
- [ ] Deploy trained model to endpoint
- [ ] Configure autoscaling settings
- [ ] Run: `python src/scripts/04_deploy_model.py`

## Phase 8: Make Predictions
- [ ] Test model endpoint with sample inputs
- [ ] Verify response quality
- [ ] Check latency and throughput
- [ ] Run: `python src/scripts/05_make_predictions.py`

## Phase 9: Iterate and Improve
- [ ] Analyze prediction errors
- [ ] Collect more training data if needed
- [ ] Adjust hyperparameters
- [ ] Run new training pipeline
- [ ] Compare model versions

## Phase 10: Optimization and Production
- [ ] Set up monitoring and logging
- [ ] Configure error handling
- [ ] Implement caching if needed
- [ ] Plan for model versioning
- [ ] Document best practices learned

## Additional Resources

### Learning
- Vertex AI Training Overview: https://cloud.google.com/vertex-ai/docs/training/overview
- LLM Fine-tuning Guide: https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini-api-overview
- Python SDK: https://googleapis.dev/python/aiplatform/latest/

### GCP Console
- Vertex AI Pipelines: https://console.cloud.google.com/vertex-ai/training/training-pipelines?project=gen-lang-client-0791883042
- Models: https://console.cloud.google.com/vertex-ai/model-registry/models?project=gen-lang-client-0791883042
- Datasets: https://console.cloud.google.com/vertex-ai/datasets?project=gen-lang-client-0791883042

### Useful Commands
```bash
# Authenticate
gcloud auth application-default login

# Set project
gcloud config set project gen-lang-client-0791883042

# List training pipelines
gcloud ai custom-jobs list --region us-west1

# View training job details
gcloud ai custom-jobs describe [JOB_ID] --region us-west1

# Create GCS bucket
gsutil mb gs://gen-lang-client-0791883042-vertex-ai-training

# Upload files to GCS
gsutil cp ./data/training_data.jsonl gs://gen-lang-client-0791883042-vertex-ai-training/

# View GCS files
gsutil ls gs://gen-lang-client-0791883042-vertex-ai-training/
```

### Common Issues & Solutions
- **API not enabled**: Enable Vertex AI API in GCP Console
- **Authentication failed**: Run `gcloud auth application-default login`
- **Data format error**: Ensure JSONL format is correct (validate with 02_prepare_data.py)
- **Insufficient quota**: Check GCP quotas and request increases if needed
- **Timeout on training**: Large datasets may take time; monitor in Console

### Next Steps After Learning
1. Try different training data and hyperparameters
2. Experiment with different base models
3. Deploy model to production
4. Integrate with your applications
5. Monitor model performance over time
"""
