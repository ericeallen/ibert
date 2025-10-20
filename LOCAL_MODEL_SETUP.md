# Local Model Setup Guide

This guide explains how to set up and use local HuggingFace models with iBERT.

## Overview

iBERT uses **Qwen2.5-Coder-1.5B-Instruct** by default, a 1.5-billion parameter code-specialized model that runs locally on your machine. This eliminates the need for API keys and provides:

- 🔒 Complete privacy (no data leaves your machine)
- 💰 Zero API costs
- ⚡ Fast inference (20-60s per request on CPU)
- 🎛️ Full control over model parameters
- 💻 Works on standard hardware (no GPU required)

## Requirements

### Hardware

**Minimum (for default 1.5B model):**
- 8GB RAM
- 5GB free disk space for model cache

**Recommended:**
- 16GB+ RAM (for larger models or multiple concurrent requests)
- NVIDIA GPU with 8GB+ VRAM (for faster inference)
- 10GB free disk space

**Apple Silicon (M1/M2/M3):**
- Works well on CPU
- 8GB+ unified memory sufficient for 1-3B models
- MPS support available for larger models (7B+)

### Software

- Python 3.13+
- CUDA Toolkit (optional, for NVIDIA GPUs)

## Installation

### 1. Install Dependencies

```bash
# Install iBERT with model dependencies
pip install -r requirements.txt
```

This installs:
- `transformers>=4.30.0` - HuggingFace transformers library
- `torch>=2.0.0` - PyTorch for model inference
- `accelerate>=0.20.0` - Memory-efficient model loading
- `protobuf>=3.20.0` - Protocol buffers for model serialization

### 2. Install Optional Dependencies

For 8-bit quantization on NVIDIA GPUs (not needed for default 1.5B model):

```bash
pip install bitsandbytes
```

**Note:** 8-bit quantization only works with CUDA (NVIDIA GPUs), not on Mac/CPU.

## Configuration

### Basic Configuration

Copy the example config:

```bash
cp config.yaml.example config.yaml
```

The default configuration in [config.yaml.example](config.yaml.example):

```yaml
model:
  provider: huggingface
  model_name: Qwen/Qwen2.5-Coder-1.5B-Instruct
  temperature: 0.2
  max_tokens: 256
  device: cpu
  load_in_8bit: false
  cache_dir: .cache
```

### Configuration Options

#### `model_name`

HuggingFace model ID. Recommended models:

- `Qwen/Qwen2.5-Coder-1.5B-Instruct` (default, 1.5B, code-specialized, FAST)
- `HuggingFaceTB/SmolLM2-1.7B-Instruct` (1.7B, general purpose)
- `meta-llama/Llama-3.2-1B-Instruct` (1B, Meta's smallest)
- `stabilityai/stable-code-3b` (3B, code-focused)
- `mistralai/Mistral-7B-Instruct-v0.3` (7B, slower, more capable)

#### `device`

Where to run the model:

- `cpu` (recommended for 1-3B models on Mac) - CPU inference
- `cuda` - Use NVIDIA GPU (fastest)
- `mps` - Use Apple Silicon GPU (for 7B+ models with sufficient RAM)
- `auto` - Automatically select device (may cause issues on Mac)

#### `max_tokens`

Maximum tokens to generate per request:

- `256` (default) - Fast responses (20-60s on CPU for 1.5B model)
- `512` - Longer responses (~1-2 minutes)
- `1024+` - Very long responses (several minutes)

Lower values = faster inference!

#### `load_in_8bit`

Load model in 8-bit precision to reduce memory usage:

- `false` (default) - Full precision
- `true` - 8-bit quantization (CUDA/NVIDIA GPUs only)

**Note:** Not needed for small models (1-3B) or on Mac.

#### `cache_dir`

Directory to cache downloaded models (default: `.cache`):

- First run downloads ~3GB (default 1.5B model)
- Subsequent runs load from cache (fast)

## First Run

On first use, the model will be downloaded from HuggingFace:

```bash
# Test with a simple question
echo "What is Ibis?" | just qa
```

**What you'll see:**

```
Loading tokenizer for Qwen/Qwen2.5-Coder-1.5B-Instruct...

Downloading model to .cache/...
This may take several minutes...

[Progress bars from HuggingFace Transformers showing download progress]

✓ Model loaded successfully on cuda

```

**What happens:**
1. Shows helpful message about first-time download
2. Downloads model files (~14GB) with progress bars
3. Loads model into memory (with status messages)
4. Runs inference
5. Outputs result

**Timing:**
- **First run:** 5-10 minutes (download + load)
- **Subsequent runs:** 30-60 seconds (load only)

The transformers library shows detailed progress bars for each file being downloaded, so you'll always know what's happening.

## Memory Requirements

### Default Model (Qwen2.5-Coder-1.5B)

| Hardware | Memory Required | Speed |
|----------|----------------|-------|
| CPU only | 4GB RAM | Medium (20-60s/response) |
| NVIDIA GPU | 3GB VRAM + 2GB RAM | Fast (~5-10s/response) |
| Apple Silicon (CPU) | 4GB unified | Medium (20-60s/response) |

### Larger Models (3B-7B)

| Hardware | Memory Required | Speed (3B) | Speed (7B) |
|----------|----------------|------------|------------|
| CPU only | 8-16GB RAM | Slow (1-3 min) | Very Slow (5-10 min) |
| NVIDIA GPU | 6-14GB VRAM | Fast (~10-20s) | Fast (~20-30s) |
| Apple Silicon (MPS) | 12-20GB unified | Medium (~30-60s) | Medium (~1-2 min) |

## Platform-Specific Setup

### NVIDIA GPU (Linux/Windows)

1. Install CUDA Toolkit (11.8 or later):
```bash
# Check CUDA version
nvidia-smi
```

2. Install PyTorch with CUDA:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

3. Configure for GPU:
```yaml
model:
  device: cuda  # or "auto"
  load_in_8bit: true  # if <16GB VRAM
```

### Apple Silicon (M1/M2/M3)

PyTorch with MPS support is included:

```yaml
model:
  device: mps  # or "auto"
  load_in_8bit: false  # bitsandbytes not supported
```

**Tips:**
- Use `device: mps` for GPU acceleration
- Increase max_tokens gradually if you have limited memory
- Monitor memory with Activity Monitor

### CPU Only

Works on any system but is slower:

```yaml
model:
  device: cpu
  load_in_8bit: false
  max_tokens: 1024  # Lower to reduce memory
```

## Performance Tuning

### Speed up Inference

1. **Use GPU** - 10-30x faster than CPU
2. **Reduce max_tokens** - Less generation time
3. **Increase temperature** slightly - Faster sampling
4. **Use 8-bit quantization** - 2x faster on GPU

### Reduce Memory Usage

1. **Enable 8-bit quantization**:
```yaml
load_in_8bit: true
```

2. **Lower max_tokens**:
```yaml
max_tokens: 1024  # Instead of 2048
```

3. **Use CPU** if GPU memory is insufficient:
```yaml
device: cpu
```

## Troubleshooting

### Model Download Fails

**Problem:** Network error or timeout during download

**Solution:**
```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/large/disk/.cache

# Resume download
python -c "from transformers import AutoModel; AutoModel.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3')"
```

### Out of Memory

**Problem:** CUDA out of memory or system RAM exhausted

**Solutions:**
1. Enable 8-bit quantization
2. Reduce max_tokens
3. Use CPU instead of GPU
4. Close other applications

### Slow Inference

**Problem:** Each response takes 30+ seconds

**Solutions:**
1. Use GPU (cuda or mps)
2. Enable 8-bit quantization
3. Reduce max_tokens
4. Ensure model is cached (not re-downloading)

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'transformers'`

**Solution:**
```bash
pip install transformers torch accelerate sentencepiece
```

## Usage Examples

### Code Completion

```bash
echo "table.filter(" | just complete
```

### SQL Translation

```bash
echo "table.filter(table.age > 18).select('name')" | just to-sql
```

### Error Fixing

```bash
echo 'table.filter(table.age > "18")' | just fix --error "TypeError"
```

### Q&A

```bash
echo "How do I use window functions in Ibis?" | just qa
```

## Monitoring

### Check GPU Usage

**NVIDIA:**
```bash
watch -n 1 nvidia-smi
```

**Apple Silicon:**
```bash
sudo powermetrics --samplers gpu_power -i1000
```

### Check Model Loading

Add logging to see what's happening:

```yaml
log_level: DEBUG  # See model loading details
```

## Advanced: Using Different Models

iBERT supports any HuggingFace causal language model:

```yaml
model:
  # Smaller model (faster, less capable)
  model_name: mistralai/Mistral-7B-v0.1

  # Different model family
  # model_name: codellama/CodeLlama-7b-Instruct-hf
```

**Note:** Models must support instruction format. Test thoroughly.

## Next Steps

- Try all six tasks (complete, to-sql, from-sql, fix, qa, doc)
- Fine-tune on your own data (coming soon)
- Experiment with temperature and max_tokens
- Compare performance across hardware

## Need Help?

- Check [README.md](README.md) for usage examples
- See [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for architecture
- File an issue on GitHub for bugs
