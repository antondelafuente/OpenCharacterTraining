# OpenRLHF Patches for Thinking Pipeline

The OCT thinking pipeline requires a custom OpenRLHF fork: [maiush/OpenRLHF](https://github.com/maiush/OpenRLHF)

This fork already includes `--length_normalize` and `--kl_loss_coef` for DPO training (critical for thinking models where teacher/student responses have very different lengths due to `<think>` blocks).

## Additional local patches (apply with `git apply openrlhf_patches.diff`)

Three small patches on top of the maiush/OpenRLHF fork:

### 1. Gemma 3 token_type_ids (`openrlhf/models/actor.py`)
Gemma 3 is multimodal and requires `token_type_ids` (all zeros for text-only training). Without this, DPO training crashes with `token_type_ids is required`.

### 2. flash_attn import guard (`openrlhf/models/ring_attn_utils.py`)
Wraps flash_attn imports in try/except so OpenRLHF doesn't crash at import time on systems without flash_attn installed.

### 3. Optimizer init order fix (`openrlhf/utils/deepspeed/deepspeed.py`)
Moves `get_optimizer_grouped_parameters()` before `AdamOptimizer` selection to avoid parameter group issues.

## Setup

```bash
# Clone the fork
git clone https://github.com/maiush/OpenRLHF.git
cd OpenRLHF

# Apply the patches
git apply ../openrlhf_patches.diff

# Install (no-deps to avoid version conflicts)
pip install -e . --no-deps
```

## Usage

Set PYTHONPATH to this fork when running DPO training:
```bash
export PYTHONPATH=/path/to/OpenRLHF:$PYTHONPATH
```
