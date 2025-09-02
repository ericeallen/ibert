# iBERT Project: Tool-Assisted Code Generation for Ibis DSL

## Project Overview
You are building **iBERT** (name inspired by BERT but not actually a BERT architecture), a sophisticated code generation system for lazy-evaluated DSLs, with Ibis as the primary target. This project leverages **Devstral** from Mistral.ai as the base model, enhanced with LoRA fine-tuning and compiler/type-checker integration.

## Core Architecture
- **Base Model**: Devstral (Mistral.ai's open-weight code-focused model)
- **Fine-tuning**: LoRA (Low-Rank Adaptation) for efficient training on single H100 GPU
- **Tool Integration**: Compiler and type-checker feedback loops for validation
- **Target DSL**: Ibis (lazy-evaluated Python→SQL framework)

## Semantic Intelligence Layer
The project uses **Serena MCP** and **Semgrep** for semantic code understanding:
- **Semgrep**: AST-based pattern matching for DSL idioms and anti-patterns
- **Serena**: Symbol-level code navigation, cross-file references, and semantic indexing
- **Semantic Chunking**: Code units based on symbols/operations, not lines
- **Hybrid Embeddings**: Content vectors + structural graph features

## Key Components
1. **Data Pipeline**: Ibis examples, SQL→Ibis conversions, synthetic task generation
2. **Training**: LoRA adapters on key transformer layers, W&B experiment tracking
3. **Inference**: Beam search with compiler validation, iterative refinement
4. **Validation**: Compile/type-check success, functional correctness, execution testing

## Development Principles
- Tool-verified training data only (passes compile/type-check)
- Semantic units over regex chunks
- Symbol graph for context expansion
- Iterative refinement with compiler feedback
- Experiment tracking with Weights & Biases

## Working with Serena MCP
- Serena provides symbolic code intelligence via MCP/LSP
- Use for extracting symbols, definitions, references, call graphs
- Build semantic index before chunking/embedding
- Powers neighborhood expansion in retrieval

## Important Instructions
- Do what has been asked; nothing more, nothing less
- NEVER create files unless absolutely necessary
- ALWAYS prefer editing existing files
- NEVER proactively create documentation files unless explicitly requested
- Use venv Python (venv/bin/python), never system python3
- Track experiments in W&B for reproducibility
- **NEVER use conditional fallbacks or substitute different models/tools than what was specified**
- **NEVER pretend to do the actual task by using something else (e.g., using GPT2 when Devstral is required)**
- **If something cannot be done (missing API keys, models, etc.), STOP and explain the issue clearly**
