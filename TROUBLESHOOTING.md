# Troubleshooting iBERT

## Common Issues and Solutions

### Import Errors: "No module named 'transformers'" or "protobuf library not found"

**Problem:**
```
ImportError during model initialization: No module named 'transformers'
Error: Missing dependency: transformers. Install with: pip install transformers torch accelerate
# or
ImportError during model initialization: requires the protobuf library but it was not found
```

**Cause:** The scripts are using system Python instead of the virtual environment, or missing dependencies.

**Solution:**

1. **Always activate your virtual environment first:**
```bash
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

2. **Use `just` commands** (they automatically use the venv):
```bash
just complete  # Uses .venv/bin/python automatically
just qa
just to-sql
```

3. **If running scripts directly**, use venv Python explicitly:
```bash
.venv/bin/python bin/ibert-complete
# NOT: ./bin/ibert-complete (uses system Python)
```

4. **Verify packages are installed in venv:**
```bash
.venv/bin/python -c "import transformers; print(transformers.__version__)"
```

---

### Script Hangs with No Output

**Problem:** Running a CLI script without input causes it to hang.

**Cause:** Missing input - scripts expect either a file argument or piped input.

**Solution:** Always provide input via pipe or file argument:

```bash
# ✅ Good - piped input
echo "table.filter(" | just complete

# ✅ Good - file input
just complete mycode.py

# ❌ Bad - no input provided
just complete  # Shows error immediately now
```

**Error message you'll see:**
```
Error: No input provided

Usage:
  echo 'your input' | ibert-complete
  ibert-complete input.txt
  cat input.txt | ibert-complete
```

---

### Model Download Times Out or Fails

**Problem:**
```
'(ReadTimeoutError("HTTPSConnectionPool(host='huggingface.co', port=443): Read timed out.")
```

**Causes:**
- Slow internet connection
- Network firewall blocking HuggingFace
- HuggingFace servers temporarily unavailable

**Solutions:**

1. **Set longer timeout:**
```bash
export HF_HUB_DOWNLOAD_TIMEOUT=300  # 5 minutes instead of default
```

2. **Use a different mirror** (if in China or restricted region):
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

3. **Download manually** and cache:
```bash
.venv/bin/python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3')
AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3')
"
```

4. **Check available disk space** (~14GB required):
```bash
df -h .cache
```

---

### Out of Memory During Model Loading

**Problem:**
```
RuntimeError: CUDA out of memory
# or
Killed (Out of memory)
```

**Solutions:**

1. **Enable 8-bit quantization** (reduces memory by 50%):
```yaml
# config.yaml
model:
  load_in_8bit: true
```

2. **Use CPU instead of GPU** (slower but more memory):
```yaml
model:
  device: cpu
```

3. **Reduce max_tokens**:
```yaml
model:
  max_tokens: 1024  # Instead of 2048
```

4. **Close other applications** to free RAM

5. **Check memory requirements** (see [LOCAL_MODEL_SETUP.md](LOCAL_MODEL_SETUP.md))

---

### Tests Hang or Take Forever

**Problem:** Running `just test` hangs or downloads 14GB model

**Cause:** Tests are trying to actually load the model instead of using mocks.

**Solution:** This should not happen - tests use `lazy_load=True`. If it does:

```bash
# Ensure you're running tests correctly
just test

# Or with pytest directly
PYTHONPATH=. .venv/bin/python -m pytest tests/ibert/ -v
```

Tests should complete in < 1 second.

---

### "just: command not found"

**Problem:** The `just` command runner is not installed.

**Solutions:**

1. **Install just:**
```bash
# macOS
brew install just

# Linux
cargo install just
# or
wget https://github.com/casey/just/releases/download/latest/just-linux-x86_64 -O /usr/local/bin/just
chmod +x /usr/local/bin/just
```

2. **Or use Python directly:**
```bash
.venv/bin/python bin/ibert-complete
.venv/bin/python bin/ibert-qa
# etc.
```

---

### Config File Not Found

**Problem:**
```
FileNotFoundError: config.yaml not found
```

**Solution:**

1. **Create config from example:**
```bash
cp config.yaml.example config.yaml
```

2. **Or use default config** (config file is optional):
```bash
# Scripts will use defaults if no config.yaml exists
echo "test" | just complete
```

3. **Specify config explicitly:**
```bash
./bin/ibert-complete --config /path/to/config.yaml input.txt
```

---

### Permission Denied on Scripts

**Problem:**
```
bash: ./bin/ibert-complete: Permission denied
```

**Solution:**

1. **Make scripts executable:**
```bash
chmod +x bin/*
```

2. **Or use Python explicitly:**
```bash
.venv/bin/python bin/ibert-complete
```

---

### Slow Inference (30+ seconds per response)

**Problem:** Model is loaded but generation is very slow.

**Solutions:**

1. **Use GPU instead of CPU:**
```yaml
model:
  device: cuda  # For NVIDIA GPU
  # or
  device: mps   # For Apple Silicon
```

2. **Enable 8-bit quantization** (faster on GPU):
```yaml
model:
  load_in_8bit: true
```

3. **Reduce max_tokens:**
```yaml
model:
  max_tokens: 512  # Generate less
```

4. **Check device actually being used:**
```bash
# Look for "✓ Model loaded successfully on cuda" (good)
# vs "✓ Model loaded successfully on cpu" (slow)
```

---

### Wrong Python Version

**Problem:**
```
SyntaxError: invalid syntax
# or
ModuleNotFoundError: No module named 'dataclasses'
```

**Cause:** Using Python < 3.13

**Solution:**

1. **Check Python version:**
```bash
.venv/bin/python --version  # Should be 3.13+
```

2. **Recreate venv with correct Python:**
```bash
rm -rf .venv
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Getting Help

If none of these solutions work:

1. **Check the logs** - errors go to stderr
2. **Run with debug info:**
```bash
PYTHONPATH=. .venv/bin/python -m pdb bin/ibert-complete
```

3. **Verify installation:**
```bash
.venv/bin/python -c "
import sys
print('Python:', sys.version)
import transformers, torch, ibis
print('transformers:', transformers.__version__)
print('torch:', torch.__version__)
print('ibis:', ibis.__version__)
"
```

4. **File an issue** with:
   - Error message (full traceback)
   - Python version
   - OS version
   - Output of verification script above
