# Fine-Tuning LLMs on Vertex AI — A Guide for GCP Engineers

> **Audience**: Software engineers with GCP experience who have *used* AI APIs but never *trained* a model.  
> **Goal**: Understand the whole fine-tuning workflow from data to deployed model, using your existing GCP mental model.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Training vs Fine-Tuning — What's the Difference?](#2-training-vs-fine-tuning--whats-the-difference)
3. [Core Concepts Mapped to GCP Concepts You Know](#3-core-concepts-mapped-to-gcp-concepts-you-know)
4. [The Gemini Fine-Tuning Architecture](#4-the-gemini-fine-tuning-architecture)
5. [The Data Format — The Most Important Detail](#5-the-data-format--the-most-important-detail)
6. [The End-to-End Workflow](#6-the-end-to-end-workflow)
7. [What Happens Inside a Training Job](#7-what-happens-inside-a-training-job)
8. [Hyperparameters Explained Simply](#8-hyperparameters-explained-simply)
9. [Costs and Quotas](#9-costs-and-quotas)
10. [Evaluating Your Model](#10-evaluating-your-model)
11. [Deploying and Calling Your Model](#11-deploying-and-calling-your-model)
12. [Common Mistakes and How to Avoid Them](#12-common-mistakes-and-how-to-avoid-them)
13. [Learning Progression](#13-learning-progression)

---

## 1. The Big Picture

Here is the entire workflow in one diagram:

```
Your Data (JSONL)
      │
      ▼
  GCS Bucket  ──────────────────────────────────────────────┐
      │                                                      │
      ▼                                                      │
Vertex AI Tuning Job                                        │
  ├── Loads base Gemini model (Google's pre-trained weights)│
  ├── Runs supervised fine-tuning (SFT) on your examples   │
  └── Saves a new model adapter to GCS ◄────────────────────┘
      │
      ▼
Vertex AI Model Registry
  └── Your tuned model (versioned, managed)
      │
      ▼
Vertex AI Endpoint (optional)
  └── REST API you can call from any app
```

The key insight: **you are not building a model from scratch**. You are taking a
model (Gemini) that already knows language, reasoning, and facts, and nudging it
to behave differently for your specific task.

---

## 2. Training vs Fine-Tuning — What's the Difference?

| | Pre-Training (from scratch) | Fine-Tuning (what you're doing) |
|---|---|---|
| **What it does** | Teaches a model language from raw text | Adapts an existing model to your task |
| **Data needed** | Billions of tokens (terabytes) | Hundreds to thousands of examples |
| **Compute** | Thousands of GPUs, weeks | A few GPUs, minutes to hours |
| **Cost** | Millions of dollars | Dollars to tens of dollars |
| **Who does it** | Google, OpenAI, Meta | You, today |
| **GCP analogy** | Building a new database engine | Running a migration on an existing DB |

**Fine-tuning** specifically means: take a model that already works well, show it
examples of how *you* want it to respond, and update a small fraction of its
parameters accordingly.

### What is SFT (Supervised Fine-Tuning)?

The word *supervised* means every training example has a correct answer. You
provide:
- Input: what the user says
- Output: the exact response you want the model to give

The model learns to reproduce that pattern. It's the same principle as supervised
ML classification — just at a much larger scale.

---

## 3. Core Concepts Mapped to GCP Concepts You Know

| ML Concept | What It Means | GCP Analogy You Know |
|---|---|---|
| **Foundation Model** | A large pre-trained model (Gemini) | A managed service like Cloud SQL — already runs, you configure it |
| **Training Job** | The compute task that runs fine-tuning | A Cloud Run Job or Dataflow job |
| **Model Weights** | The numerical parameters that define model behavior | A database schema + data |
| **JSONL Training Data** | Your labelled examples | Input to a Dataflow pipeline |
| **GCS Bucket** | Storage for your data + trained artifacts | Same GCS you already use |
| **Model Registry** | Versioned store of trained models | Artifact Registry for containers |
| **Endpoint** | Deployed model with a REST API | Cloud Run service |
| **Adapter (LoRA)** | A small set of weight changes on top of the base model | A diff/patch applied to a base image |
| **Epoch** | One full pass through your training data | One iteration of a batch job |
| **Loss** | How wrong the model is (lower = better) | Error rate metric on a dashboard |
| **Overfitting** | Model memorizes training data, fails on new input | Cache that serves stale data to all users |

---

## 4. The Gemini Fine-Tuning Architecture

Vertex AI uses **LoRA** (Low-Rank Adaptation) for Gemini fine-tuning. You don't
need to know the math, but the concept is important:

```
Base Gemini Model (frozen — Google's weights, never change)
         │
         ▼
    LoRA Adapter  ◄── This is what training produces
    (small matrix of changes)
         │
         ▼
  Your Fine-Tuned Model
  = Base + Adapter applied on top
```

**Why this matters for you:**
- Training is fast because only the adapter is trained (~1% of total parameters)
- Your adapter is stored separately from the base model
- If Gemini gets a new version, you may need to retrain your adapter on the new base
- `adapter_size` (1, 2, 4, 8, 16) controls how many parameters the adapter has — bigger = more capacity, but more risk of overfitting small datasets

---

## 5. The Data Format — The Most Important Detail

Vertex AI Gemini fine-tuning requires data in **multi-turn conversation JSONL
format**. Each line is one training example:

```jsonl
{"contents": [{"role": "user", "parts": [{"text": "Classify: I love this!"}]}, {"role": "model", "parts": [{"text": "Positive"}]}]}
{"contents": [{"role": "user", "parts": [{"text": "Classify: This is awful."}]}, {"role": "model", "parts": [{"text": "Negative"}]}]}
```

### Format breakdown

```json
{
  "contents": [
    {
      "role": "user",
      "parts": [{ "text": "Your instruction or question here" }]
    },
    {
      "role": "model",
      "parts": [{ "text": "The exact response you want" }]
    }
  ]
}
```

### Rules

| Rule | Detail |
|---|---|
| File format | `.jsonl` — one JSON object per line, no trailing commas |
| Required fields | `contents`, `role` (`user`\|`model`), `parts`, `text` |
| Turn order | Must start with `user`, alternate `user`→`model` |
| Multi-turn | You can have multiple user/model exchanges per example |
| Min examples | ~10 to test, 100+ for meaningful results |
| Recommended | 500–5,000 for production quality |
| Max token length | Varies by model; keep examples under 8K tokens |

### In this project

The helper function in [src/data_prep.py](src/data_prep.py) handles this format:

```python
from src.data_prep import DataPreparationPipeline

example = DataPreparationPipeline.create_training_example(
    user_message="Classify: I love this!",
    model_response="Positive",
    system_instruction="Classify sentiment as Positive, Negative, or Neutral.",
)
# Result:
# {
#   "contents": [
#     {"role": "user",  "parts": [{"text": "Classify sentiment...\n\nClassify: I love this!"}]},
#     {"role": "model", "parts": [{"text": "Positive"}]}
#   ]
# }
```

---

## 6. The End-to-End Workflow

### Step 1 — Prepare your data

```bash
# Option A: Use the built-in example (sentiment data)
python src/scripts/02_prepare_data.py

# Option B: Load a public Google/HuggingFace dataset
python src/scripts/03_load_public_data.py

# Result: creates data/training_data.jsonl and data/validation_data.jsonl
```

### Step 2 — Create a GCS bucket and upload data

```bash
# Create bucket (one-time)
gsutil mb -p gen-lang-client-0791883042 -l us-west1 \
  gs://gen-lang-client-0791883042-vertex-ai-training

# Upload data
gsutil cp data/training_data.jsonl \
  gs://gen-lang-client-0791883042-vertex-ai-training/data/

gsutil cp data/validation_data.jsonl \
  gs://gen-lang-client-0791883042-vertex-ai-training/data/
```

> **Why GCS?** The training job runs on Google's infrastructure, not your machine.
> It needs to read data from somewhere accessible — GCS is the answer, exactly
> like any other GCP job (Dataflow, Cloud Run, etc.).

### Step 3 — Enable required APIs (one-time)

```bash
gcloud services enable aiplatform.googleapis.com \
  --project gen-lang-client-0791883042
```

### Step 4 — Submit the tuning job

Edit [src/scripts/04_submit_tuning_job.py](src/scripts/04_submit_tuning_job.py),
uncomment the job submission block, then run:

```bash
python src/scripts/04_submit_tuning_job.py
```

Or via the Python SDK directly:

```python
import vertexai
from vertexai.tuning import sft

vertexai.init(project="gen-lang-client-0791883042", location="us-west1")

tuning_job = sft.train(
    source_model="gemini-2.0-flash-001",
    train_dataset="gs://gen-lang-client-0791883042-vertex-ai-training/data/training_data.jsonl",
    validation_dataset="gs://gen-lang-client-0791883042-vertex-ai-training/data/validation_data.jsonl",
    epochs=3,                    # see parameter guide below
    learning_rate_multiplier=1.0,
    adapter_size=4,
    tuned_model_display_name="my-sentiment-classifier",
)

print(tuning_job.resource_name)   # projects/.../tuningJobs/...
```

#### Training parameter guide

**`epochs=3`**  
One epoch = the model sees every example in your training data once. With `epochs=3`
it reads through all your data 3 times.

Think of it like re-reading notes before an exam — more passes = more memorized,
but at some point you stop learning new things and just over-memorize.

| Value | Effect |
|---|---|
| `1–2` | Underfits — model doesn't learn enough |
| `3–5` | Sweet spot for most tasks |
| `10+` | Overfits — model memorizes examples, fails on new input |

---

**`learning_rate_multiplier=1.0`**  
Controls how big each weight update step is during training. It's a multiplier on
top of Gemini's built-in base learning rate (you scale it, you don't set it directly).

Analogy: imagine tuning a dial that's already at a good position — do you make tiny
micro-adjustments or big sweeping turns?

| Value | Behavior |
|---|---|
| `0.5` | Cautious, small steps — preserves base model knowledge, slower convergence |
| `1.0` | Balanced — good default, start here |
| `2.0` | Aggressive — learns faster but risks "forgetting" what Gemini already knows |

Only lower to `0.5` if your model starts giving worse answers on general questions
it used to handle fine.

---

**`adapter_size=4`**  
The capacity of the LoRA adapter — the small trainable layer added on top of frozen
Gemini weights. Valid values: `1, 2, 4, 8, 16`.

Think of it as the number of columns in a feature table you're adding to an existing
database schema — more columns = more expressive, but also more risk of overfitting
on small data.

| Size | Best for |
|---|---|
| `1` | Very simple tasks (binary yes/no) |
| `4` | Most fine-tuning tasks — good default |
| `8` | Complex tasks with 500+ examples |
| `16` | Very complex tasks; high overfitting risk on small datasets |

> **TLDR for your first run**: keep all three at their defaults. Only adjust after
> you've run one job and reviewed the loss curves in the Console.

### Step 5 — Monitor the job

```bash
# Via CLI
gcloud ai tuning-jobs list --region us-west1

# Via Console (easiest)
# https://console.cloud.google.com/vertex-ai/training/training-pipelines?project=gen-lang-client-0791883042
```

Watch for:
- **State: RUNNING** → job is executing
- **State: SUCCEEDED** → training complete, model is in the registry
- **State: FAILED** → check logs for data format or quota issues

### Step 6 — Find your model in the registry

```bash
# Via Console:
# https://console.cloud.google.com/vertex-ai/model-registry?project=gen-lang-client-0791883042
```

Or in Python:

```python
from google.cloud import aiplatform
aiplatform.init(project="gen-lang-client-0791883042", location="us-west1")

models = aiplatform.Model.list(filter='display_name="my-sentiment-classifier"')
print(models[0].resource_name)
```

### Step 7 — Test the model (no deployment needed)

You can call the tuned model directly without deploying to an endpoint:

```python
import vertexai
from vertexai.generative_models import GenerativeModel

vertexai.init(project="gen-lang-client-0791883042", location="us-west1")

# Use the tuned model resource name from the registry
model = GenerativeModel("projects/gen-lang-client-0791883042/locations/us-west1/models/YOUR_MODEL_ID")
response = model.generate_content("Classify: This product is incredible!")
print(response.text)   # Expected: "Positive"
```

### Step 8 — Deploy to an endpoint (for production)

```python
model = aiplatform.Model("projects/.../models/YOUR_MODEL_ID")
endpoint = model.deploy(
    machine_type="n1-standard-2",
    min_replica_count=1,
    max_replica_count=3,
)
# endpoint.predict(instances=[...])
```

---

## 7. What Happens Inside a Training Job

Here's what Vertex AI actually does when you submit a tuning job — in terms you
already understand:

```
1. Vertex AI spins up managed GPU VMs  (like GKE nodes, but managed by Google)
       │
       ▼
2. Loads base Gemini weights from Google's internal storage
       │
       ▼
3. Reads your JSONL from GCS, batches examples
       │
       ▼
4. For each batch:
   a. Runs a "forward pass": feeds the user_message to the model, generates output
   b. Compares generated output to your model_response (the correct answer)
   c. Computes "loss" — a number showing how wrong the model was
   d. Runs "backpropagation" — calculates how to adjust adapter weights
   e. Updates the LoRA adapter weights slightly
       │
       ▼
5. Repeats steps 3–4 for all batches × number of epochs
       │
       ▼
6. Saves the trained adapter to GCS + registers the model
       │
       ▼
7. Shuts down VMs — you're only billed for actual training time
```

**Key takeaway**: Steps 4a–4e are the "learning". The model isn't being
programmed — it's iteratively adjusting numbers until its outputs match
your training examples.

---

## 8. Hyperparameters Explained Simply

These are the knobs you control in [config/settings.py](config/settings.py):

| Parameter | What it controls | Default | Guidance |
|---|---|---|---|
| `EPOCHS` | How many full passes through your data | 3 | Start with 3. More epochs → more learning but risk of overfitting. |
| `LEARNING_RATE_MULTIPLIER` | How big each weight adjustment step is | 1.0 | Keep at 1.0 to start. Lower (0.5) if model forgets previous knowledge. |
| `ADAPTER_SIZE` | Size of the LoRA adapter (1/2/4/8/16) | 4 | Start at 4. Increase to 8 or 16 if your task is complex. |

### Overfitting — the main thing to watch for

Overfitting = the model memorizes your training examples but can't handle
slightly different inputs. Signs:
- Training loss keeps dropping BUT validation loss starts rising
- Model gives perfect answers to training examples, but fails on new inputs
- Model responses become very short or repetitive

How to fix: fewer epochs, smaller adapter size, or more diverse training data.

---

## 9. Costs and Quotas

### Training costs (approximate, as of mid-2026)

| Model | Cost |
|---|---|
| `gemini-2.0-flash-001` | ~$0.80 per 1,000 training examples |
| `gemini-1.5-pro-002` | ~$3.00 per 1,000 training examples |
| `gemini-1.5-flash-002` | ~$0.80 per 1,000 training examples |

For 200 examples × 3 epochs = **under $1** to experiment.

### Endpoint costs

Deployed endpoints charge per compute hour, even when idle. For learning:
- **Skip deployment** and call the model directly via the tuned model resource name
- Only deploy when you actually need a persistent REST endpoint

### Check your quotas

```bash
gcloud compute project-info describe --project gen-lang-client-0791883042
```

Or in Console: IAM & Admin → Quotas → filter "aiplatform"

Common quota to check: `CUSTOM_MODEL_TRAINING_JOBS_PER_PROJECT_PER_REGION`

---

## 10. Evaluating Your Model

After training, you need to verify the model actually improved. Approaches from
simplest to most rigorous:

### Manual spot-check (start here)

```python
test_prompts = [
    "Classify: The concert was breathtaking!",
    "Classify: I want my money back, terrible service.",
    "Classify: The food was edible I suppose.",
]
for prompt in test_prompts:
    response = model.generate_content(prompt)
    print(f"Input  : {prompt}")
    print(f"Output : {response.text}\n")
```

### Metrics to look for in the training job logs

| Metric | What it means | Good sign |
|---|---|---|
| `train_loss` | Error on training data | Decreasing over epochs |
| `val_loss` | Error on validation data | Decreasing (and not rising while train_loss drops) |
| `train_accuracy` | Correctness on training set | Rising |
| `val_accuracy` | Correctness on validation set | Rising, close to train_accuracy |

### The validation split is your safety net

This is why [src/scripts/02_prepare_data.py](src/scripts/02_prepare_data.py)
creates both `training_data.jsonl` and `validation_data.jsonl`. The model never
trains on validation data — it's the held-out set that tells you if the model
generalizes or just memorised.

---

## 11. Deploying and Calling Your Model

### Option A: Direct model call (no endpoint — cheapest for testing)

```python
import vertexai
from vertexai.generative_models import GenerativeModel

vertexai.init(project="gen-lang-client-0791883042", location="us-west1")

model = GenerativeModel("projects/PROJECT/locations/us-west1/models/MODEL_ID")
response = model.generate_content("Your prompt here")
print(response.text)
```

### Option B: Endpoint (for apps with real traffic)

```python
endpoint = aiplatform.Endpoint("projects/PROJECT/locations/us-west1/endpoints/ENDPOINT_ID")

response = endpoint.predict(instances=[{"content": "Your prompt here"}])
print(response.predictions[0])
```

### Calling from Cloud Run / GKE (your familiar territory)

Your deployed endpoint is a standard REST API. From any service on GCP:

```bash
curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  "https://us-west1-aiplatform.googleapis.com/v1/projects/gen-lang-client-0791883042/locations/us-west1/endpoints/ENDPOINT_ID:predict" \
  -d '{"instances": [{"content": "Classify: I love this!"}]}'
```

---

## 12. Common Mistakes and How to Avoid Them

### Data format errors (most common)

```
# WRONG — old format (does not work for Gemini tuning)
{"text_input": "Hello", "output": "Hi"}

# CORRECT — multi-turn Gemini format
{"contents": [{"role": "user", "parts": [{"text": "Hello"}]}, {"role": "model", "parts": [{"text": "Hi"}]}]}
```

Validate before uploading:
```bash
python -c "
from src.data_prep import DataPreparationPipeline
ok = DataPreparationPipeline.validate_jsonl('data/training_data.jsonl')
print('Valid' if ok else 'INVALID — check format')
"
```

### Not versioning model IDs

Always use versioned model IDs (e.g. `gemini-2.0-flash-001`, not `gemini-2.0-flash`).
Unversioned aliases can point to a different underlying version after a Google
update, which could break your tuning job or produce inconsistent results.

### Too few training examples

- **< 10 examples**: Job may fail or model shows no improvement
- **10–50 examples**: Useful only for testing the pipeline
- **50–200 examples**: You'll see the model learn the pattern
- **500+ examples**: Meaningful quality for real tasks

### Not using a validation split

Without validation data you can't tell if the model is overfitting. Always hold
out 15–20% of your data for validation.

### Leaving endpoints idle

An endpoint with `min_replica_count=1` charges 24/7. Delete it when done:
```bash
gcloud ai endpoints delete ENDPOINT_ID --region us-west1
```

---

## 13. Learning Progression

Follow this order — each step builds on the previous:

```
Week 1: Pipeline understanding
  ├── Run 01_setup_exploration.py     → verify GCP auth and SDK
  ├── Run 02_prepare_data.py          → understand data format
  ├── Run 03_load_public_data.py      → load Alpaca dataset (200 examples)
  └── Upload to GCS                   → gsutil cp

Week 2: First training job
  ├── Submit tuning job (04_submit_tuning_job.py)
  ├── Monitor in Console              → watch loss curves
  └── Test model output               → manual spot-check

Week 3: Iterate
  ├── Try different datasets           → ag_news, xsum
  ├── Adjust hyperparameters           → epochs, adapter_size
  └── Compare model versions           → did quality improve?

Week 4: Deploy
  ├── Deploy to endpoint
  ├── Call from a Cloud Run service
  └── Add monitoring and logging
```

---

## Quick Reference

```bash
# Authenticate
gcloud auth application-default login

# Set project
gcloud config set project gen-lang-client-0791883042

# Enable API
gcloud services enable aiplatform.googleapis.com

# Create GCS bucket
gsutil mb -p gen-lang-client-0791883042 -l us-west1 \
  gs://gen-lang-client-0791883042-vertex-ai-training

# Upload training data
gsutil cp data/*.jsonl gs://gen-lang-client-0791883042-vertex-ai-training/data/

# List tuning jobs
gcloud ai tuning-jobs list --region us-west1

# List models in registry
gcloud ai models list --region us-west1

# Delete endpoint (to stop billing)
gcloud ai endpoints delete ENDPOINT_ID --region us-west1
```

---

## 14. Understanding Tokens in Training

A **token** is the basic unit the model reads and generates. Not a word — closer to a word fragment.

```
"Hello, how are you today?"
 Hello  ,  how  are  you  today  ?
   1    2   3    4    5     6    7  → 7 tokens
```

```
"unbelievable"  →  "un"  "believ"  "able"  → 3 tokens
```

A rough rule: **1 token ≈ 0.75 words** in English.

### Where tokens appear in your training pipeline

**1. In your JSONL training data**

Every example gets tokenised before training:

```json
{
  "contents": [
    {"role": "user",  "parts": [{"text": "Give three tips for staying healthy."}]},
    {"role": "model", "parts": [{"text": "1. Eat balanced meals\n2. Exercise\n3. Sleep"}]}
  ]
}
```

```
user turn:  "Give three tips for staying healthy."  →  ~8 tokens
model turn: "1. Eat balanced meals..."              →  ~12 tokens
total for this example                              →  ~20 tokens
```

**2. In the training cost calculation**

You are billed per **token processed**, not per example:

```
160 training examples × ~200 tokens avg = 32,000 tokens
× 3 epochs                               = 96,000 tokens trained
```

For `gemini-2.5-flash` SFT that is a very small job — likely under $1.

**3. In the model limits**

| Limit | Gemini 2.5 Flash | What it means |
|---|---|---|
| Max input tokens | ~1,000,000 | Max length of a single prompt |
| Max output tokens | ~8,192 | Max length of a single response |
| Max tokens per training example | ~32,768 | If your example exceeds this it gets truncated |

Your Alpaca examples average ~200 tokens — well within limits.

**4. During the forward pass (what actually happens)**

```
Your example (tokenised):
[user: "Give tips...", model: "1. Eat..."]
         ↓
Token IDs: [1374, 2093, 9321, ...]
         ↓
Model predicts next token at each position
         ↓
Loss = how wrong was each predicted token vs your actual model turn
         ↓
Weights adjusted to make those token predictions more accurate
```

The model only computes loss on the **model turn tokens** — it is not penalised for
the user turn, only trained to produce the correct response.

### Why this matters for your dataset quality

- **Too short responses** (< 10 tokens): model learns almost nothing per example
- **Too long examples** (> 32K tokens): gets truncated, losing the end of your response
- **Repetitive tokens** across examples: model over-indexes on those patterns

Your 200 Alpaca examples are well-sized — varied instructions with substantive
responses, typically 100–400 tokens each. That is a good learning signal.

---

*This project uses: Python 3.11 · `google-cloud-aiplatform>=1.50` · `vertexai.tuning.sft` · Gemini 2.5 Flash*
