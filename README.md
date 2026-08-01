# Vertex AI LLM Learning Project

A comprehensive learning project for fine-tuning Large Language Models (LLMs) using Google Cloud's Vertex AI platform. Perfect for understanding modern ML workflows on GCP.

## 🎯 Project Goal

Learn how to:
- Prepare training data for LLM fine-tuning
- Create and manage training pipelines on Vertex AI
- Fine-tune foundation models (Gemini) with your own data
- Deploy and make predictions with trained models
- Monitor and iterate on model performance

## 📋 Quick Start

### Prerequisites
- GCP account with project `gen-lang-client-0791883042`
- Google Cloud SDK installed: https://cloud.google.com/sdk/docs/install
- Python 3.8+
- gcloud CLI configured

### Setup
```bash
# Clone/navigate to project
cd vertex-ai-llm-learning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Authenticate with GCP
gcloud auth application-default login

# Set your project
gcloud config set project gen-lang-client-0791883042

# Copy environment file
cp .env.example .env
# Edit .env with your settings (optional - defaults are provided)
```

### First Steps
```bash
# 1. Explore setup
python src/scripts/01_setup_exploration.py

# 2. Prepare training data
python src/scripts/02_prepare_data.py

# 3. Check generated data
ls -la data/
```

## 📁 Project Structure

```
vertex-ai-llm-learning/
├── src/                          # Source code
│   ├── __init__.py
│   ├── data_prep.py              # Data preparation utilities
│   ├── vertex_ai_trainer.py       # Vertex AI training logic
│   └── scripts/                  # Example scripts
│       ├── 01_setup_exploration.py       # Initial setup check
│       ├── 02_prepare_data.py            # Data preparation
│       ├── 03_create_training_job.py     # (Coming soon)
│       ├── 04_deploy_model.py            # (Coming soon)
│       └── 05_make_predictions.py        # (Coming soon)
├── config/
│   └── settings.py               # Configuration management
├── data/                         # Training data directory
│   ├── training_data.jsonl       # Training examples
│   └── validation_data.jsonl     # Validation examples
├── notebooks/                    # Jupyter notebooks for exploration
├── .env.example                  # Environment variables template
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Python dependencies
├── TODO.md                       # Learning path & checklist
└── README.md                     # This file

```

## 🎓 Learning Path

See [TODO.md](TODO.md) for a detailed learning checklist organized in 10 phases:

1. **Environment Setup** - Install and configure tools
2. **Understand Vertex AI Basics** - Learn the platform
3. **Data Preparation** - Prepare training data
4. **GCS Setup** - Configure cloud storage
5. **Create Training Pipeline** - Start your first training job
6. **Model Evaluation** - Analyze model performance
7. **Deploy Model** - Make model available for inference
8. **Make Predictions** - Test your trained model
9. **Iterate and Improve** - Refine based on results
10. **Optimization** - Production-ready setup

## 📚 Key Concepts

### Fine-tuning vs Training from Scratch
- **Fine-tuning** (RECOMMENDED for learning): Adapt existing models (Gemini) to your specific task
  - Faster training
  - Requires less data
  - Better out-of-the-box performance
- **Training from scratch**: Build custom models
  - Requires extensive data
  - More complex
  - Better for highly specialized domains

### Data Format
Training data must be in JSONL format (one JSON object per line):
```jsonl
{"text_input": "Your instruction", "output": "Expected response"}
{"text_input": "Another instruction", "output": "Another response"}
```

### Foundation Models
- **Gemini 1.5 Pro**: Most capable, better for complex reasoning
- **Gemini 1.5 Flash**: Faster and more efficient

## 🔧 Core Components

### DataPreparationPipeline
```python
from src.data_prep import DataPreparationPipeline

# Create training example
example = DataPreparationPipeline.create_training_example(
    instruction="Classify sentiment",
    input_text="I love this!",
    output="Positive"
)

# Save to JSONL
examples = [example]
DataPreparationPipeline.save_to_jsonl(examples, "./data/training.jsonl")

# Validate format
is_valid = DataPreparationPipeline.validate_jsonl("./data/training.jsonl")
```

### VertexAITrainer
```python
from src.vertex_ai_trainer import VertexAITrainer

trainer = VertexAITrainer()

# Get model info
models = trainer.get_foundation_model_info()

# Create training job
config = trainer.create_training_pipeline(
    training_data_path="gs://bucket/data.jsonl",
    model_name="my-model"
)
```

## 🚀 Common Tasks

### Upload Data to GCS
```bash
gsutil cp data/training_data.jsonl gs://your-bucket/
gsutil ls gs://your-bucket/
```

### Monitor Training Jobs
Visit: https://console.cloud.google.com/vertex-ai/training/training-pipelines

### View Models
Visit: https://console.cloud.google.com/vertex-ai/model-registry

## 📖 Documentation Links

- [Vertex AI Overview](https://cloud.google.com/vertex-ai/docs)
- [LLM Fine-tuning Guide](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini-api-overview)
- [Training Data Requirements](https://cloud.google.com/vertex-ai/docs/training/create-training-pipeline/tuning-data-requirements)
- [Python SDK Reference](https://googleapis.dev/python/aiplatform/latest/)
- [Gemini API Documentation](https://ai.google.dev/docs)

## 💡 Tips for Success

1. **Start Small**: Begin with small datasets (10-50 examples) to validate workflow
2. **Quality Over Quantity**: Good, diverse examples matter more than quantity
3. **Monitor Training**: Watch loss curves and metrics in the Console
4. **Iterate**: Collect more data based on model failures
5. **Document Learnings**: Track what works and what doesn't

## ❓ Troubleshooting

**API Not Enabled**
```bash
gcloud services enable aiplatform.googleapis.com
```

**Authentication Issues**
```bash
gcloud auth application-default login
```

**Data Format Issues**
```bash
# Validate your JSONL
python -c "from src.data_prep import DataPreparationPipeline; print(DataPreparationPipeline.validate_jsonl('data/training_data.jsonl'))"
```

## 🎯 Next Steps

1. ✅ Complete Phase 1: Environment Setup
2. Read Vertex AI documentation (Phase 2)
3. Run data preparation examples (Phase 3)
4. Create your first training pipeline (Phase 5)
5. Deploy and test your model (Phase 7-8)

## 📝 Notes

- This is a learning project designed for experimentation
- Start with fine-tuning before attempting foundation model training
- Your GCP project has quota limits; monitor usage
- Keep sensitive data and credentials in `.env` (not in git)

## 🔗 Resources

- GCP Project: https://console.cloud.google.com/welcome?project=gen-lang-client-0791883042
- Vertex AI Console: https://console.cloud.google.com/vertex-ai
- Training Pipelines: https://console.cloud.google.com/vertex-ai/training/training-pipelines?project=gen-lang-client-0791883042

---

Happy Learning! 🚀
