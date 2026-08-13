# Llama-Embed-Nemotron-1B

**An adaptation of the Llama-Embed-Nemotron methodology (arXiv:2511.07025) to Llama-3.2-1B.** Note: the original paper used Llama-3.1-8B; results at 1B scale are expected to be lower.

This repository implements the training methodology described in the paper:
> **Llama-Embed-Nemotron-8B: A Universal Text Embedding Model for Multilingual and Cross-Lingual Tasks**
> (Babakhin et al., NVIDIA, 2025)
> [arXiv:2511.07025](https://arxiv.org/abs/2511.07025)

The original paper achieved #1 on the MMTEB leaderboard using Llama-3.1-8B. This implementation adapts the methodology to Llama-3.2-1B for more accessible training and deployment.

## Paper Fidelity

[`VERIFICATION.md`](VERIFICATION.md) maps each section of the paper to the corresponding implementation, quoting the paper text alongside the code that implements it.

## Key Features

- **Bi-directional Llama**: Converts causal decoder to bi-directional encoder by removing attention masks
- **Instruction-Aware Embeddings**: Task-specific instructions guide embedding generation
- **2-Stage Training**:
  - Stage 1: Retrieval pretraining (70% of data)
  - Stage 2: Multi-task fine-tuning (30% of data)
- **InfoNCE Contrastive Loss**: With hard negative mining (no in-batch negatives)
- **Synthetic Data Generation**: Multi-LLM pipeline for diverse training data
- **Model Merging (Model Soup)**: Combine multiple checkpoints for better generalization
- **MTEB Evaluation**: Compatible with MTEB benchmark

## Architecture

### Bi-Directional Transformation

```python
# Standard Llama (causal decoder)
Llama-3.2-1B → Causal Attention → Text Generation

# Our approach (bi-directional encoder)
Llama-3.2-1B → Remove Causal Mask → Bi-directional Attention → Text Embeddings
```

### Instruction Format

```
Instruct: {task_instruction}
Query: {text}
```

Examples:
- Retrieval: `"Retrieve relevant passages for this query"`
- STS: `"Retrieve semantically similar text"`
- Classification: `"Classify the topic of this text"`

### Training Pipeline

```
┌─────────────────────────────────────────────────────────┐
│ Stage 1: Retrieval Pretraining                          │
│ - 1 hard negative per query                            │
│ - LR: 1e-5, Temp: 0.02                                 │
│ - Focus: Query-document retrieval                       │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Stage 2: Multi-Task Fine-Tuning                         │
│ - 4 hard negatives per query                           │
│ - LR: 2e-6, Temp: 0.02                                 │
│ - Focus: Retrieval + STS + Classification               │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Model Merging (Optional)                                │
│ - Train 6 models with different configs                │
│ - Average parameters uniformly                          │
│ - Better generalization, no inference cost             │
└─────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone repository
git clone https://github.com/gihong0303/Llama-Embed-Nemotron-1B.git
cd Llama-Embed-Nemotron-1B

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.36+
- CUDA 11.8+ (for GPU training)
- 16GB+ GPU memory (for 1B model)

## Quick Start

### 1. Prepare Data

Your training data should be in JSONL format:

```jsonl
{"query": "What is machine learning?", "positive": "Machine learning is a subset of AI...", "negatives": ["Deep learning uses neural networks...", "..."], "instruction": "Retrieve relevant passages", "task_type": "retrieval"}
{"query": "Paris is beautiful", "positive": "The city of Paris is stunning", "negatives": ["Berlin is nice", "..."], "instruction": "Retrieve semantically similar text", "task_type": "sts"}
```

### 2. Generate Synthetic Data (Optional)

```bash
python scripts/generate_synthetic_data.py \
    --corpus data/your_corpus.txt \
    --output data/synthetic_train.jsonl \
    --model_name meta-llama/Llama-3.2-1B \
    --num_queries_per_doc 2
```

### 3. Hard Negative Mining

```python
from src.data.hard_negative_mining import HardNegativeMiner

miner = HardNegativeMiner(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Add hard negatives to your data
data_with_negatives = miner.mine_batched(
    dataset=your_data,
    corpus=your_corpus,
    k=4  # Number of negatives
)
```

### 4. Stage 1 Training

**Single GPU:**
```bash
python scripts/train_stage1.py \
    --train_data data/stage1_train.jsonl \
    --output_dir outputs/stage1 \
    --batch_size 32 \
    --learning_rate 1e-5 \
    --num_negatives 1 \
    --fp16 \
    --gradient_checkpointing
```

**Multi-GPU (8 GPUs - Recommended):**
```bash
bash scripts/launch_ddp_stage1.sh \
    --train_data data/stage1_train.jsonl \
    --output_dir outputs/stage1 \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --num_negatives 1 \
    --fp16 \
    --gradient_checkpointing
```

>  **Note**: When using 8 GPUs, reduce `batch_size` per GPU to 16. Effective batch size = 16 × 8 = 128.
> Multi-GPU launch scripts: [`scripts/launch_ddp_stage1.sh`](scripts/launch_ddp_stage1.sh), [`scripts/launch_ddp_stage2.sh`](scripts/launch_ddp_stage2.sh).

### 5. Stage 2 Training

**Single GPU:**
```bash
python scripts/train_stage2.py \
    --stage1_checkpoint outputs/stage1/best_model \
    --train_data data/stage2_train.jsonl \
    --output_dir outputs/stage2 \
    --batch_size 32 \
    --learning_rate 2e-6 \
    --num_negatives 4 \
    --fp16 \
    --gradient_checkpointing
```

**Multi-GPU (8 GPUs - Recommended):**
```bash
bash scripts/launch_ddp_stage2.sh \
    --stage1_checkpoint outputs/stage1/best_model \
    --train_data data/stage2_train.jsonl \
    --output_dir outputs/stage2 \
    --batch_size 8 \
    --learning_rate 2e-6 \
    --num_negatives 4 \
    --fp16 \
    --gradient_checkpointing
```

### 6. Model Merging (Optional)

Train multiple models with different hyperparameters, then merge:

```python
from src.utils.model_merging import merge_models_uniform

model_paths = [
    "outputs/run1/stage2/best_model",
    "outputs/run2/stage2/best_model",
    "outputs/run3/stage2/best_model",
]

merged_model = merge_models_uniform(
    model_paths=model_paths,
    output_path="outputs/merged_model"
)
```

## Usage

### Basic Inference

```python
from transformers import AutoTokenizer
from src.models.embedding_model import InstructionAwareEmbeddingModel

# Load model
model = InstructionAwareEmbeddingModel.from_pretrained("outputs/stage2/best_model")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# Encode for retrieval
query = "What is the capital of France?"
query_embed = model.encode(
    query,
    instruction="Retrieve relevant passages for this query",
    tokenizer=tokenizer,
    normalize=True
)

documents = ["Paris is the capital of France.", "Berlin is in Germany."]
doc_embeds = model.encode(
    documents,
    instruction="",  # No instruction for documents
    tokenizer=tokenizer,
    normalize=True
)

# Compute similarities
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(query_embed, doc_embeds)
```

### Semantic Textual Similarity

```python
texts = ["I love programming", "Coding is my passion"]
embeds = model.encode(
    texts,
    instruction="Retrieve semantically similar text",
    tokenizer=tokenizer,
    normalize=True
)

similarity = cosine_similarity(embeds[0:1], embeds[1:2])[0][0]
```

### Classification

```python
text = "The stock market reached a new high."
labels = ["Business", "Science", "Sports"]

text_embed = model.encode(text, instruction="Classify the topic", tokenizer=tokenizer)
label_embeds = model.encode(labels, instruction="Classify the topic", tokenizer=tokenizer)

prediction = labels[cosine_similarity(text_embed, label_embeds).argmax()]
```

## Evaluation

### Custom Evaluation

```bash
python scripts/evaluate.py \
    --model_path outputs/stage2/best_model \
    --eval_data data/eval.jsonl
```

### MTEB Benchmark

```bash
pip install mteb

python scripts/evaluate.py \
    --model_path outputs/stage2/best_model \
    --use_mteb \
    --mteb_tasks "STSBenchmark,SICK,STS12,STS13"
```

## Paper Implementation Details

This implementation follows the paper as closely as possible:

| Component | Paper (8B) | Our Implementation (1B) | Notes |
|-----------|-----------|-------------------------|-------|
| Base Model | Llama-3.1-8B | Llama-3.2-1B | Scaled down |
| Hidden Size | 4096 | 2048 | Proportional to model size |
| Attention | Bi-directional | Bi-directional |  Exact match |
| Pooling | Mean pooling | Mean pooling |  Exact match |
| Loss | InfoNCE | InfoNCE |  Exact match |
| Temperature | 0.02 | 0.02 |  Exact match |
| Stage 1 Negatives | 1 | 1 |  Exact match |
| Stage 2 Negatives | 4 | 4 |  Exact match |
| Stage 1 LR | 1e-5 | 1e-5 |  Exact match |
| Stage 2 LR | 2e-6 | 2e-6 |  Exact match |
| Batch Size (8B) | 2048 → 128 | 256 → 128 | Scaled for resources |
| Hard Neg Mining | 95% threshold | 95% threshold |  Exact match |
| Model Merging | Uniform averaging | Uniform averaging |  Exact match |

### Key Differences from Paper

1. **Model Size**: 1B instead of 8B (for accessibility)
2. **Training Data**: You'll need to provide your own (paper used 16.1M samples)
3. **Batch Size**: Scaled down for consumer hardware
4. **Number of GPUs**: Paper used 64 A100s, our config works on 1-8 GPUs

## Project Structure

```
Llama-Embed-Nemotron-1B/
├── src/
│   ├── models/
│   │   ├── bidirectional_llama.py    # Bi-directional Llama implementation
│   │   └── embedding_model.py        # Instruction-aware wrapper
│   ├── data/
│   │   ├── dataset.py                # Dataset classes
│   │   ├── hard_negative_mining.py   # Hard negative mining
│   │   └── synthetic_generation.py   # Synthetic data generation
│   ├── training/
│   │   ├── config.py                 # Training configurations
│   │   ├── loss.py                   # InfoNCE loss
│   │   └── trainer.py                # Training loop
│   └── utils/
│       └── model_merging.py          # Model soup implementation
├── scripts/
│   ├── train_stage1.py               # Stage 1 training script
│   ├── train_stage2.py               # Stage 2 training script
│   ├── generate_synthetic_data.py    # Synthetic data generation
│   └── evaluate.py                   # Evaluation script
├── examples/
│   └── basic_usage.py                # Usage examples
└── README.md
```

## Citation

If you use this implementation, please cite both the original paper and this repository:

```bibtex
@article{babakhin2025llamaembednemotron,
  title={Llama-Embed-Nemotron-8B: A Universal Text Embedding Model for Multilingual and Cross-Lingual Tasks},
  author={Babakhin, Yauhen and Osmulski, Radek and Ak, Ronay and Moreira, Gabriel and Xu, Mengyao and Schifferer, Benedikt and Liu, Bo and Oldridge, Even},
  journal={arXiv preprint arXiv:2511.07025},
  year={2025}
}
```

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

The base Llama models are subject to Meta's license terms.

## Acknowledgments

- NVIDIA for the original Llama-Embed-Nemotron paper and methodology
- Meta for the Llama model family
- The MTEB team for the evaluation benchmark

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Issues

If you encounter any problems, please open an issue on GitHub.

---

**Note**: This is an independent implementation based on the published paper. It is not affiliated with NVIDIA or Meta.
