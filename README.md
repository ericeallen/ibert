# ibert: Multi Task PEFT Model for Lazy-Evaluated DSLs

## Overview

ibert is a code generation model designed for lazy-evaluated Domain-Specific Languages (DSLs), with **Ibis** as the primary target. The project leverages **Mistral's Devstral** as the base model, enhanced with **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning and integrated compiler/type-checker feedback loops for validation.

### 1. **Base Model: Devstral**
- Mistral's open-weight code-focused model (`mistralai/Devstral-Small-2505`)
- Optimized for code understanding and generation
- Supports multiple backends: MPS (Metal), CUDA, CPU

### 2. **PEFT Strategy: LoRA Implementation**
The LoRA adapter targets attention projection matrices, allowing fine-tuning with:
- **~0.5% of original parameters** (vs full fine-tuning)
- **Memory efficiency**: Single H100 GPU sufficient
- **Modularity**: Swap adapters for different DSL tasks

### 3. **Multi-Task Training Framework**
Supports six core tasks with weighted sampling:
1. **Code Completion** : Complete partial Ibis expressions
2. **Ibis→SQL Translation** : Convert between representations
3. **Error Resolution** : Fix compilation/type errors
4. **Q&A** : Answer Ibis-related questions
5. **Function Documentation** : Generate docstrings
6. **SQL→Ibis** : Reverse translation

## Installation & Usage

### Requirements
```bash
pip install -r requirements.txt
```

### Quick Start
You can run basic scripts like data generation by installing ```just``` and calling the functions in ```justfile```.
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
│   └── repo/          # Mined examples
├── configs/           # Semgrep rules, training configs
└── adapters/         # Saved LoRA weights
```

