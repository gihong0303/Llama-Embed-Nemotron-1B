# Quick Start Guide

This guide will help you get started with Llama-Embed-Nemotron-1B in under 10 minutes.

## Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure you have access to Llama-3.2-1B
# You may need to accept Meta's license and authenticate with Hugging Face
huggingface-cli login
```

## Option 1: Minimal Working Example (No Training)

If you just want to understand how the model works without training:

```python
# examples/minimal_example.py
import torch
from transformers import AutoTokenizer
from src.models.embedding_model import create_embedding_model

# Create model (this converts Llama-3.2-1B to bi-directional encoder)
model = create_embedding_model("meta-llama/Llama-3.2-1B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# Encode some text
texts = ["Hello world", "Hi there"]
embeddings = model.encode(
    texts,
    instruction="Retrieve semantically similar text",
    tokenizer=tokenizer,
    normalize=True
)

print(f"Embeddings shape: {embeddings.shape}")
# Output: Embeddings shape: torch.Size([2, 2048])
```

## Option 2: Train on Toy Dataset

### Step 1: Create Toy Training Data

```python
# create_toy_data.py
import json

# Create minimal training data
train_data = [
    {
        "query": "What is machine learning?",
        "positive": "Machine learning is a subset of artificial intelligence.",
        "negatives": [
            "Deep learning uses neural networks.",
            "Python is a programming language."
        ],
        "instruction": "Retrieve relevant passages for this query",
        "task_type": "retrieval"
    },
    {
        "query": "How does photosynthesis work?",
        "positive": "Photosynthesis is the process by which plants convert sunlight into energy.",
        "negatives": [
            "Cellular respiration produces ATP.",
            "DNA contains genetic information."
        ],
        "instruction": "Retrieve relevant passages for this query",
        "task_type": "retrieval"
    },
    # Add more examples...
]

# Save to JSONL
with open("data/toy_train.jsonl", "w") as f:
    for item in train_data:
        f.write(json.dumps(item) + "\n")

print(f"Created toy dataset with {len(train_data)} samples")
```

### Step 2: Train Stage 1 (Quick Version)

```bash
# Train for just 10 steps to verify everything works
python scripts/train_stage1.py \
    --train_data data/toy_train.jsonl \
    --output_dir outputs/toy_stage1 \
    --batch_size 2 \
    --num_epochs 1 \
    --learning_rate 1e-5 \
    --num_negatives 1 \
    --logging_steps 1 \
    --save_steps 10
```

Expected output:
```
Loading tokenizer...
Loading training data...
  Loaded 2 training samples
Creating model...
Training...
Step 1: loss: 0.6931
Step 2: loss: 0.6724
...
Training complete!
```

### Step 3: Test the Trained Model

```python
# test_model.py
from src.models.embedding_model import InstructionAwareEmbeddingModel
from transformers import AutoTokenizer

# Load your trained model
model = InstructionAwareEmbeddingModel.from_pretrained("outputs/toy_stage1/best_model")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# Test retrieval
query = "Tell me about AI"
documents = [
    "Artificial intelligence is the simulation of human intelligence.",
    "The weather is nice today.",
    "Machine learning is a branch of AI."
]

query_embed = model.encode(
    query,
    instruction="Retrieve relevant passages for this query",
    tokenizer=tokenizer
)

doc_embeds = model.encode(documents, instruction="", tokenizer=tokenizer)

# Find most similar
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(query_embed, doc_embeds)[0]

for doc, sim in zip(documents, similarities):
    print(f"{sim:.4f}: {doc}")

# Expected output (similar docs should have higher scores):
# 0.8234: Artificial intelligence is the simulation of human intelligence.
# 0.3456: The weather is nice today.
# 0.7891: Machine learning is a branch of AI.
```

## Option 3: Full Training Pipeline

For a complete training run following the paper:

### Step 1: Prepare Real Data

You'll need:
- **Stage 1**: ~5-10M query-document pairs for retrieval
- **Stage 2**: ~1-4M mixed task samples (retrieval + STS + classification)

Data format (JSONL):
```jsonl
{"query": "...", "positive": "...", "negatives": ["...", "..."], "instruction": "...", "task_type": "retrieval"}
```

### Step 2: Generate Synthetic Data (Optional)

```bash
# Generate synthetic queries from your corpus
python scripts/generate_synthetic_data.py \
    --corpus data/my_corpus.txt \
    --output data/synthetic_retrieval.jsonl \
    --model_name meta-llama/Llama-3.2-1B \
    --num_queries_per_doc 2
```

### Step 3: Hard Negative Mining

```python
from src.data.hard_negative_mining import HardNegativeMiner
from src.data.dataset import load_jsonl, save_jsonl

# Load data
data = load_jsonl("data/my_data.jsonl")
corpus = [item["positive"] for item in data]

# Mine hard negatives
miner = HardNegativeMiner(model_name="sentence-transformers/all-MiniLM-L6-v2")
data_with_negatives = miner.mine_batched(
    dataset=data,
    corpus=corpus,
    k=4
)

# Save
save_jsonl(data_with_negatives, "data/my_data_with_negatives.jsonl")
```

### Step 4: Stage 1 Training

```bash
python scripts/train_stage1.py \
    --train_data data/stage1_train.jsonl \
    --eval_data data/stage1_eval.jsonl \
    --output_dir outputs/stage1 \
    --batch_size 32 \
    --learning_rate 1e-5 \
    --num_negatives 1 \
    --num_epochs 1 \
    --fp16 \
    --gradient_checkpointing \
    --wandb  # Optional: track with W&B
```

Training time (approximate):
- 1M samples on 1x A100: ~6 hours
- 5M samples on 1x A100: ~30 hours
- 10M samples on 8x A100: ~8 hours

### Step 5: Stage 2 Training

```bash
python scripts/train_stage2.py \
    --stage1_checkpoint outputs/stage1/best_model \
    --train_data data/stage2_train.jsonl \
    --eval_data data/stage2_eval.jsonl \
    --output_dir outputs/stage2 \
    --batch_size 32 \
    --learning_rate 2e-6 \
    --num_negatives 4 \
    --num_epochs 1 \
    --fp16 \
    --gradient_checkpointing
```

### Step 6: Model Merging (Optional but Recommended)

Train 3-6 models with different random seeds or hyperparameters:

```bash
# Run 1
python scripts/train_stage2.py ... --output_dir outputs/run1 --seed 42

# Run 2
python scripts/train_stage2.py ... --output_dir outputs/run2 --seed 43

# Run 3
python scripts/train_stage2.py ... --output_dir outputs/run3 --seed 44
```

Then merge:

```python
from src.utils.model_merging import merge_models_uniform

merged = merge_models_uniform(
    model_paths=[
        "outputs/run1/best_model",
        "outputs/run2/best_model",
        "outputs/run3/best_model",
    ],
    output_path="outputs/final_merged"
)
```

### Step 7: Evaluate

```bash
# Custom evaluation
python scripts/evaluate.py \
    --model_path outputs/final_merged \
    --eval_data data/eval.jsonl

# MTEB benchmark
python scripts/evaluate.py \
    --model_path outputs/final_merged \
    --use_mteb \
    --mteb_tasks "STSBenchmark,SICK"
```

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
--batch_size 16

# Use gradient accumulation
--gradient_accumulation_steps 2

# Enable gradient checkpointing
--gradient_checkpointing
```

### Slow Training

```bash
# Use mixed precision
--fp16

# Increase batch size if memory allows
--batch_size 64

# Use more workers
--num_workers 8
```

### Model Not Learning

- Check your data quality (are positives truly positive?)
- Verify hard negatives are challenging but not false negatives
- Try adjusting learning rate (start with 1e-5, try 5e-6 or 2e-5)
- Ensure instructions are appropriate for your task

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- See [examples/basic_usage.py](examples/basic_usage.py) for more usage examples
- Check the [paper](https://arxiv.org/abs/2511.07025) for theoretical background

## Getting Help

- Open an issue on GitHub
- Check existing issues for solutions
- Review the paper's appendix for implementation details
