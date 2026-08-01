# Training Data Directory

This directory contains training data for fine-tuning LLMs on Vertex AI.

## Format Requirements

All training data must be in JSONL format (one JSON object per line):

```jsonl
{"text_input": "Instruction or task", "output": "Expected response"}
{"text_input": "Another instruction", "output": "Another response"}
```

## Files

- `training_data.jsonl` - Main training dataset (80% of data)
- `validation_data.jsonl` - Validation dataset (20% of data)

## Data Preparation

To prepare your data:

1. Create examples in JSONL format
2. Validate with: `python -c "from src.data_prep import DataPreparationPipeline; DataPreparationPipeline.validate_jsonl('data/training_data.jsonl')"`
3. Upload to GCS: `gsutil cp *.jsonl gs://your-bucket/`

## Size Guidelines

- **Minimum**: 10-20 examples (for testing)
- **Recommended**: 50-500 examples (for good results)
- **Optimal**: 500-5000 examples (for production)

## Quality Tips

1. **Diversity**: Include varied examples covering different scenarios
2. **Clarity**: Clear instructions and responses
3. **Correctness**: Ensure outputs are accurate
4. **Format**: Consistent formatting in examples
