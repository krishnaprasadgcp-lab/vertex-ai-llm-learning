# TODO: Vertex AI LLM Learning Project

Learning path for fine-tuning LLMs using Vertex AI on GCP.
**Project:** `gen-lang-client-0791883042` | **Region:** `us-west1` | **Repo:** https://github.com/krishnaprasadgcp-lab/vertex-ai-llm-learning

---

## Phase 1: Environment Setup ✅ DONE
- [x] Create project structure
- [x] Create Python venv and install dependencies (`pip install -r requirements.txt`)
- [x] Push code to GitHub
- [ ] Copy `.env.example` to `.env` (optional — defaults already set in `config/settings.py`)
- [ ] Authenticate with GCP: `gcloud auth application-default login`
- [ ] Set GCP project: `gcloud config set project gen-lang-client-0791883042`
- [ ] Verify Vertex AI API is enabled: `gcloud services enable aiplatform.googleapis.com`

## Phase 2: Understand the Basics
- [ ] Read `GUIDE.md` in this repo — written for your GCP background
- [ ] Run setup check: `python src/scripts/01_setup_exploration.py`
- [ ] Review foundation models available for tuning:
  - [ ] `gemini-2.0-flash-001` — recommended starting point (fast + tunable)
  - [ ] `gemini-1.5-pro-002` — more capable, higher cost
  - [ ] `gemini-1.5-flash-002` — balanced speed/quality
- [ ] Explore GCP Console: [Vertex AI > Training Pipelines](https://console.cloud.google.com/vertex-ai/training/training-pipelines?project=gen-lang-client-0791883042)

## Phase 3: Data Preparation ← YOU ARE HERE
- [ ] Understand the required JSONL format (multi-turn Gemini format):
  ```json
  {"contents": [{"role": "user", "parts": [{"text": "..."}]}, {"role": "model", "parts": [{"text": "..."}]}]}
  ```
- [ ] **Option A** — Run the built-in example (sentiment data, 15 examples):
  ```bash
  python src/scripts/02_prepare_data.py
  ```
- [ ] **Option B** — Load a public Google/HuggingFace dataset (200 examples, recommended):
  ```bash
  python src/scripts/03_load_public_data.py
  ```
  Available datasets: `alpaca` (instruction), `ag_news` (classification), `xsum` (summarization), `natural_questions` (Google Q&A)
- [ ] Validate generated files exist: `ls -lh data/*.jsonl`
- [ ] Review a sample: `head -1 data/training_data.jsonl | python -m json.tool`

## Phase 4: GCS Setup (Storage)
- [ ] Create GCS bucket:
  ```bash
  gsutil mb -p gen-lang-client-0791883042 -l us-west1 \
    gs://gen-lang-client-0791883042-vertex-ai-training
  ```
- [ ] Upload training data:
  ```bash
  gsutil cp data/training_data.jsonl   gs://gen-lang-client-0791883042-vertex-ai-training/data/
  gsutil cp data/validation_data.jsonl gs://gen-lang-client-0791883042-vertex-ai-training/data/
  ```
- [ ] Verify upload: `gsutil ls gs://gen-lang-client-0791883042-vertex-ai-training/data/`

## Phase 5: Submit Tuning Job
- [ ] Review parameters in `config/settings.py` (epochs, adapter_size, learning_rate_multiplier)
- [ ] Open `src/scripts/04_submit_tuning_job.py` and uncomment the job submission block
- [ ] Run: `python src/scripts/04_submit_tuning_job.py`
- [ ] Monitor job in Console: [Training Pipelines](https://console.cloud.google.com/vertex-ai/training/training-pipelines?project=gen-lang-client-0791883042)
- [ ] Watch for:
  - [ ] State: RUNNING → job executing
  - [ ] State: SUCCEEDED → model saved to registry
  - [ ] State: FAILED → check logs (usually data format or quota issue)

## Phase 6: Evaluate the Model
- [ ] Find your model in [Model Registry](https://console.cloud.google.com/vertex-ai/model-registry?project=gen-lang-client-0791883042)
- [ ] Check training/validation loss curves in the Console
- [ ] Test with sample prompts (no deployment needed):
  ```python
  from vertexai.generative_models import GenerativeModel
  model = GenerativeModel("projects/.../models/YOUR_MODEL_ID")
  print(model.generate_content("Your test prompt").text)
  ```
- [ ] Compare output quality vs base Gemini model
- [ ] Check for overfitting: val_loss rising while train_loss drops

## Phase 7: Deploy to Endpoint (Optional)
- [ ] Deploy model to a Vertex AI endpoint (only needed for production REST API)
- [ ] Configure autoscaling (min: 1, max: 3 replicas)
- [ ] Test endpoint call
- [ ] **Remember**: delete endpoint when not in use to avoid idle charges:
  ```bash
  gcloud ai endpoints delete ENDPOINT_ID --region us-west1
  ```

## Phase 8: Iterate and Improve
- [ ] Analyse errors from model evaluation
- [ ] Try a larger dataset (500+ examples from `03_load_public_data.py`)
- [ ] Adjust hyperparameters: increase `adapter_size` to 8, try `epochs=5`
- [ ] Re-run training and compare model versions
- [ ] Try a different dataset/task (summarization, Q&A)

## Phase 9: Optimization
- [ ] Set up Cloud Logging for prediction monitoring
- [ ] Implement model versioning strategy
- [ ] Document what worked and what didn't (update `GUIDE.md`)

---

## Quick Commands Reference

```bash
# Activate venv
source venv/bin/activate

# GCP auth
gcloud auth application-default login
gcloud config set project gen-lang-client-0791883042

# Prepare data
python src/scripts/03_load_public_data.py

# Upload to GCS
gsutil cp data/*.jsonl gs://gen-lang-client-0791883042-vertex-ai-training/data/

# List tuning jobs
gcloud ai tuning-jobs list --region us-west1

# Git push after changes
git add . && git commit -m "your message" && git push
```

## Common Issues & Fixes
| Issue | Fix |
|---|---|
| API not enabled | `gcloud services enable aiplatform.googleapis.com` |
| Auth failed | `gcloud auth application-default login` |
| Data format error | Run `python src/scripts/02_prepare_data.py` to see correct format |
| Quota exceeded | Check [IAM > Quotas](https://console.cloud.google.com/iam-admin/quotas?project=gen-lang-client-0791883042) |
| Model not improving | Add more/diverse examples; check val_loss in Console |

### Next Steps After Learning
1. Try different training data and hyperparameters
2. Experiment with different base models
3. Deploy model to production
4. Integrate with your applications
5. Monitor model performance over time
"""
