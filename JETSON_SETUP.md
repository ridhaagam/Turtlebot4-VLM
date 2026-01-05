# Jetson Orin Florence-2 Setup Guide

Complete documentation of all fixes and setup steps required to run Florence-2 Vision Language Model on NVIDIA Jetson Orin for the Waste Bin Monitoring System.

## Table of Contents

1. [Overview](#overview)
2. [Known Issues & Fixes](#known-issues--fixes)
3. [Automated Setup](#automated-setup)
4. [Manual Setup Steps](#manual-setup-steps)
5. [Troubleshooting](#troubleshooting)

---

## Overview

This project runs Florence-2 VLM on Jetson Orin for waste bin detection and fullness classification. The following issues were encountered and fixed during development:

| Issue | Root Cause | Solution |
|-------|------------|----------|
| `Torch not compiled with CUDA enabled` | PyPI torch lacks CUDA | Use Jetson AI Lab PyPI |
| `flash_attn module not found` | Florence-2 imports flash_attn | Create mock module |
| `Input type (float) and bias type (c10::Half)` | dtype mismatch | Cast pixel_values to model dtype |
| `Column 'detection_id' cannot be null` | SQLAlchemy FK timing | Add db.session.flush() |
| `cuSPARSELt library not found` | Missing LD_LIBRARY_PATH | Export path before running |

---

## Known Issues & Fixes

### 1. PyTorch CUDA Not Available

**Error:**
```
RuntimeError: Torch not compiled with CUDA enabled
```

**Cause:** Standard PyPI PyTorch is CPU-only. Jetson requires special CUDA-enabled builds.

**Solution:** Install from Jetson AI Lab PyPI:
```bash
pip uninstall -y torch torchvision
pip install torch==2.8.0 torchvision==0.23.0 --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126
```

**Verification:**
```python
import torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.get_device_name(0))  # Should print "Orin"
```

---

### 2. flash_attn Module Not Found

**Error:**
```
ModuleNotFoundError: No module named 'flash_attn'
```

**Cause:** Florence-2's modeling code imports `flash_attn` but doesn't strictly require it.

**Solution:** Create a mock module in `edge/src/detector.py`:

```python
def _setup_flash_attn_mock():
    """Create mock flash_attn module if not available."""
    try:
        import flash_attn
        return  # Already installed
    except ImportError:
        pass

    # Create mock package
    import importlib.util
    mock_dir = "/tmp/mock_flash_attn"
    os.makedirs(mock_dir, exist_ok=True)

    init_code = '''
__version__ = "2.0.0"
def flash_attn_func(*args, **kwargs):
    raise NotImplementedError("Flash attention not available")
def flash_attn_varlen_func(*args, **kwargs):
    raise NotImplementedError("Flash attention not available")
'''
    with open(f"{mock_dir}/__init__.py", "w") as f:
        f.write(init_code)
    with open(f"{mock_dir}/flash_attn_interface.py", "w") as f:
        f.write(init_code)

    sys.path.insert(0, "/tmp")
    spec = importlib.util.spec_from_file_location("flash_attn", f"{mock_dir}/__init__.py")
    flash_attn = importlib.util.module_from_spec(spec)
    sys.modules['flash_attn'] = flash_attn
    spec.loader.exec_module(flash_attn)

    spec2 = importlib.util.spec_from_file_location(
        "flash_attn.flash_attn_interface",
        f"{mock_dir}/flash_attn_interface.py"
    )
    flash_attn_interface = importlib.util.module_from_spec(spec2)
    sys.modules['flash_attn.flash_attn_interface'] = flash_attn_interface
    spec2.loader.exec_module(flash_attn_interface)
```

**Note:** This mock is called automatically when the detector loads.

---

### 3. dtype Mismatch (float vs Half)

**Error:**
```
RuntimeError: Input type (float) and bias type (c10::Half) should be the same
```

**Cause:** The model runs in float16 (Half) for efficiency, but the processor returns float32 pixel_values.

**Solution:** Cast inputs to model dtype in `Florence2Detector._run_inference()`:

```python
def _run_inference(self, pil_image: Image.Image, prompt: str) -> str:
    inputs = self.processor(text=prompt, images=pil_image, return_tensors="pt")

    # Move to device
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(self.device)

    # CRITICAL FIX: Cast pixel_values to model dtype (float16 on CUDA)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(self._torch_dtype)

    with torch.no_grad():
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            num_beams=3,
            early_stopping=True
        )
    # ... rest of inference
```

**Storage of dtype:**
```python
def load(self) -> bool:
    # ... model loading ...

    # Store the dtype for later use
    self._torch_dtype = torch.float16 if (self.device == "cuda" and self.config.use_fp16) else torch.float32
```

---

### 4. cuSPARSELt Library Not Found

**Error:**
```
OSError: libcusparseLt.so.0: cannot open shared object file
```

**Cause:** PyTorch requires cuSPARSELt library which is installed by pip but not in LD_LIBRARY_PATH.

**Solution:** Export the library path before running:

```bash
export LD_LIBRARY_PATH="/path/to/venv/lib/python3.10/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH"
```

This is handled automatically in `scripts/build_edge.sh`:
```bash
CUSPARSELT_LIB="$VENV_DIR/lib/python3.10/site-packages/nvidia/cusparselt/lib"
if [ -d "$CUSPARSELT_LIB" ]; then
    export LD_LIBRARY_PATH="$CUSPARSELT_LIB:$LD_LIBRARY_PATH"
fi
```

---

### 5. Detection ID NULL (Backend)

**Error:**
```
IntegrityError: Column 'detection_id' cannot be null
```

**Cause:** SQLAlchemy doesn't generate the Detection.id until commit, but we need it for child records.

**Solution:** Add `db.session.flush()` after adding Detection, before creating DetectionObjects:

```python
# In dashboard/backend/app/routes/detections.py
def create_detection():
    detection = Detection.from_dict(data)
    db.session.add(detection)

    # CRITICAL: Flush to generate detection.id before using as FK
    db.session.flush()

    # Now detection.id is available
    for obj_data in data.get("detections", []):
        obj = DetectionObject.from_dict(obj_data, detection.id)
        db.session.add(obj)

    db.session.commit()
```

---

## Automated Setup

### One-Command Setup

```bash
# Clone the repo
git clone <repo-url>
cd embedded-system-final

# Run automated setup (handles everything)
./scripts/setup_jetson.sh
```

### What It Does

1. Detects Jetson platform
2. Creates Python virtual environment
3. Installs PyTorch from Jetson AI Lab PyPI (CUDA enabled)
4. Installs all dependencies
5. Sets up LD_LIBRARY_PATH for cuSPARSELt
6. Downloads Florence-2 model
7. Verifies CUDA availability
8. Tests detector module

---

## Manual Setup Steps

If automated setup fails, follow these steps manually:

### Step 1: System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev python3-venv libopencv-dev v4l-utils libhdf5-dev
```

### Step 2: Create Virtual Environment

```bash
cd /path/to/embedded-system-final
python3 -m venv edge/venv
source edge/venv/bin/activate
pip install --upgrade pip wheel setuptools
```

### Step 3: Install PyTorch (CUDA)

```bash
# IMPORTANT: Use Jetson AI Lab PyPI, not standard PyPI!
pip install torch==2.8.0 torchvision==0.23.0 --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126

# Verify CUDA
python3 -c "import torch; print(torch.cuda.is_available())"  # Should print True
```

### Step 4: Install Dependencies

```bash
pip install -r edge/requirements/base.txt
```

### Step 5: Set Library Path

```bash
# Add to ~/.bashrc for persistence
echo 'export LD_LIBRARY_PATH="$HOME/embedded-system-final/edge/venv/lib/python3.10/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Step 6: Download Model

```bash
python3 << 'EOF'
from edge.src.detector import _setup_flash_attn_mock
_setup_flash_attn_mock()

from transformers import AutoModelForCausalLM, AutoProcessor
import torch

model_name = "microsoft/Florence-2-base"
print("Downloading processor...")
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
print("Downloading model...")
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16)
print("Done! Model cached at ~/.cache/huggingface/hub/")
EOF
```

### Step 7: Configure & Run

```bash
# Edit config with your laptop's IP
nano edge/config/config.yaml
# Change: url: "http://YOUR_LAPTOP_IP:5000"

# Run detection
./scripts/build_edge.sh run
```

---

## Troubleshooting

### Check CUDA Status
```bash
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

### Check Camera
```bash
ls -la /dev/video*
./scripts/build_edge.sh test-camera
```

### Check Model Loading
```bash
./scripts/build_edge.sh test-model
```

### Check Memory
```bash
free -h
# Florence-2-base needs ~2GB GPU memory
nvidia-smi
```

### Common Errors

| Error | Fix |
|-------|-----|
| `CUDA out of memory` | Reduce batch size or use smaller model |
| `Camera not found` | Check `/dev/video*` and permissions |
| `Connection refused` | Dashboard not running or wrong IP |
| `Module not found` | Run `source edge/venv/bin/activate` |

---

## Files Modified

These files contain the fixes:

| File | Changes |
|------|---------|
| `edge/src/detector.py` | flash_attn mock, dtype fix, VQA classification |
| `scripts/build_edge.sh` | Jetson AI Lab PyPI, LD_LIBRARY_PATH |
| `scripts/setup_edge.sh` | Platform detection, full setup |
| `dashboard/backend/app/routes/detections.py` | db.session.flush() fix |
| `edge/requirements/jetson.txt` | PyTorch source documentation |

---

## Version Information

Tested with:
- JetPack 6.x (L4T)
- Python 3.10
- PyTorch 2.8.0 (Jetson AI Lab)
- CUDA 12.6
- Florence-2-base

---

## Quick Reference

```bash
# Full setup
./scripts/setup_jetson.sh

# Run detection
./scripts/build_edge.sh run

# Test camera
./scripts/build_edge.sh test-camera

# Test model
./scripts/build_edge.sh test-model
```
