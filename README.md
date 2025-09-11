# iBERT: Tool-Assisted Code Generation for Lazy-Evaluated DSLs

## Overview

iBERT is a sophisticated code generation system designed for lazy-evaluated Domain-Specific Languages (DSLs), with **Ibis** as the primary target. The project leverages **Mistral's Devstral** as the base model, enhanced with **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning and integrated compiler/type-checker feedback loops for validation.

## Key Innovation: Semantic Intelligence + PEFT

Unlike traditional code generation approaches that rely on simple text patterns, iBERT employs a semantic-aware architecture that understands code structure at the AST level. This is combined with LoRA adapters for efficient fine-tuning on a single H100 GPU, making it both powerful and resource-efficient.

## Architecture Components

### 1. **Base Model: Devstral**
- Mistral's open-weight code-focused model (`mistralai/Devstral-Small-2505`)
- Optimized for code understanding and generation
- Supports multiple backends: MPS (Metal), CUDA, CPU

### 2. **PEFT Strategy: LoRA Implementation**
```python
LoRA Configuration:
- Rank (r): 32
- Alpha: 64  
- Dropout: 0.1
- Target Modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
- Task Type: CAUSAL_LM
```

The LoRA adapter targets attention projection matrices, allowing fine-tuning with:
- **~0.5% of original parameters** (vs full fine-tuning)
- **Memory efficiency**: Single H100 GPU sufficient
- **Modularity**: Swap adapters for different DSL tasks

### 3. **Semantic Intelligence Layer**

#### **Semgrep Integration**
- AST-based pattern matching for DSL idioms
- Rule-based detection of anti-patterns
- Semantic tagging for code units

#### **Serena MCP (Model Context Protocol)**
- Symbol-level code navigation
- Cross-file reference tracking
- Call graph construction
- Semantic indexing for retrieval

### 3. **Multi-Task Training Framework**

Supports six core tasks with weighted sampling:
1. **Code Completion** : Complete partial Ibis expressions
2. **Ibis→SQL Translation** : Convert between representations
3. **Error Resolution** : Fix compilation/type errors
4. **Q&A** : Answer Ibis-related questions
5. **Function Documentation** : Generate docstrings
6. **SQL→Ibis** : Reverse translation

## Training Pipeline

### Data Preparation
```python
TrainingCorpusBuilder:
├── Documentation Processing (Ibis docs)
├── Repository Mining (Functions, Classes, Tests)
├── Synthetic Example Generation
└── Semantic Chunking (Symbol-based, not line-based)
```

### LoRA Training Configuration
```python
MultiTaskConfig:
- Learning Rate: 1e-4
- Batch Size: 4 (gradient accumulation: 4)
- Warmup Ratio: 0.05
- LoRA Modules: Attention projections
- Mixed Precision: fp16/bf16
- Optimizer: AdamW (8-bit via bitsandbytes)
```

### Validation Loop
1. **Compile Check**: Verify syntax validity
2. **Type Check**: Ensure type consistency
3. **Execution Test**: Run against test backends
4. **Iterative Refinement**: Up to 3 refinement iterations with compiler feedback

## Inference Pipeline

### Beam Search with Validation
```python
Generation Strategy:
- Beam Size: 5
- Temperature: 0.7
- Top-p: 0.95
- Compiler Validation: Enabled
- Max Refinements: 3
```

## PEFT-Specific Advantages

### Memory Efficiency
- **Base Model**: ~7B parameters (frozen)
- **LoRA Adapters**: ~35M parameters (trainable)
- **GPU Memory**: ~16GB (fits on single H100)
- **Training Time**: 2-3 hours per epoch

### Task Specialization
Different LoRA adapters for:
- **Autocomplete**: High temperature, creative
- **Translation**: Low temperature, precise
- **Error Fixing**: Compiler-feedback driven

### Adapter Composition
```python
# Load base model once
base = AutoModelForCausalLM.from_pretrained("mistralai/Devstral")

# Hot-swap task-specific adapters
model.load_adapter("adapters/ibis_completion")
model.load_adapter("adapters/sql_translation")
model.load_adapter("adapters/error_resolution")
```

## Installation & Usage

### Requirements
```bash
pip install -r requirements.txt
# Core: transformers, peft, torch, ibis-framework
# Semantic: semgrep, serena-mcp
# Training: wandb, datasets, accelerate
```

### Quick Start
```python
from src.models.devstral_lora import IBertModel, IBertConfig

# Initialize model with LoRA
config = IBertConfig(
    lora_rank=32,
    use_compiler_feedback=True
)
model = IBertModel(config)

# Generate with validation
result = model.generate(
    prompt="df.group_by('category').",
    task_type="code_completion",
    validate_with_compiler=True
)
```

### Training Custom Adapters
```python
from src.training.multi_task_trainer import MultiTaskTrainer

trainer = MultiTaskTrainer(
    model=model,
    config=training_config,
    output_dir="adapters/custom"
)
trainer.train()
```

## Project Structure
```
ibert/
├── src/
│   ├── models/          # LoRA-enhanced Devstral
│   ├── training/        # Multi-task trainer
│   ├── evaluation/      # Task-specific metrics
│   └── data/           # Semantic processors
├── data/
│   ├── corpus/         # Training datasets
│   ├── embeddings/     # Vector indices
│   └── repo/          # Mined examples
├── configs/           # Semgrep rules, training configs
└── adapters/         # Saved LoRA weights
```

## Advanced Features

### Compiler-in-the-Loop
- Real-time validation during generation
- Error-driven refinement
- Type-aware beam search

### Semantic Caching
- Symbol-level memoization
- Cross-session adapter reuse
- Incremental index updates

### Multi-Backend Support
- **Development**: MPS (Apple Silicon)
- **Training**: CUDA (H100/A100)
- **Inference**: CPU fallback

## Future Directions

1. **Adapter Merging**: Combine task-specific LoRAs
2. **Quantization**: 4-bit LoRA (QLoRA) for edge deployment
3. **Multi-DSL**: Extend to Polars, DuckDB, Arrow
4. **Retrieval Enhancement**: GraphRAG integration
5. **Continuous Learning**: Online adapter updates

## Contributing

We welcome contributions in:
- Semgrep rules for new DSL patterns
- Task-specific LoRA configurations
- Evaluation benchmarks
- Semantic chunking strategies

## Citation

```bibtex
@software{ibert2025,
  title={iBERT: Tool-Assisted Code Generation for Lazy-Evaluated DSLs},
  author={Your Team},
  year={2025},
  url={https://github.com/yourusername/ibert}
}
```

## License

MIT License - See LICENSE file for details

---

**Note for PEFT Practitioners**: This project demonstrates practical application of LoRA in a production code generation system, with emphasis on semantic understanding, tool integration, and efficient multi-task learning. The architecture is designed to be adapter-friendly, allowing rapid experimentation with different PEFT techniques (LoRA, QLoRA, IA³) without modifying the core pipeline.
