## vlaconvert: QuantVLA -> GPTQ-Marlin Experiment

This repository keeps the full QuantVLA/GR00T codebase and adds converted
QuantVLA weight experiments.

- Conversion, inspection, and layer benchmark tools are in `vlaconvert_tools/`.
- Runtime integration lives in `gr00t/quantization/quantvla_converted_linear.py`.
- LIBERO server entrypoint: `run_quantvla_converted_server.sh`.

Basic flow:

```bash
python vlaconvert_tools/inspect_quantvla_pack.py --pack-dir /path/to/duquant_pack
python vlaconvert_tools/convert_quantvla_to_gptq_like.py --base-checkpoint /path/to/dense_ckpt --pack-dir /path/to/duquant_pack --output ./outputs/libero_10_quantvla_gptq_like
CUDA_VISIBLE_DEVICES=0 bash run_quantvla_converted_server.sh real libero_10 ./outputs/libero_10_quantvla_gptq_like 5556
CUDA_VISIBLE_DEVICES=0 ./run_libero_eval.sh libero_10 --headless --port 5556
```

LIBERO eval now records per-action-step latency by default:

```text
/tmp/logs/libero_eval_<suite>_latency_steps.jsonl
/tmp/logs/libero_eval_<suite>_latency_steps.csv
/tmp/logs/libero_eval_<suite>_latency_summary.json
```

The key timing fields are:

- `client_roundtrip_ms`: eval client request -> server response.
- `server_handler_ms`: server-side `get_action` endpoint time.
- `policy_model_get_action_ms`: GR00T model action generation time.
- `env_step_ms`: LIBERO simulator `env.step(action)` time.
- `step_total_ms`: one measured action step, from policy query through env step.

See `vlaconvert_tools/README.md` for details.

<div align="center">

<img src="assets/icon.png" alt="QuantVLA Logo" width="100">&nbsp;<img src="assets/title.svg" alt="QuantVLA" height="60">

**Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models**

<a href="https://cvpr.thecvf.com/Conferences/2026"><img src="https://img.shields.io/badge/CVPR-2026-6B46C1?style=for-the-badge&logo=ieee&logoColor=white" alt="CVPR 2026"></a>
<a href="https://arxiv.org/pdf/2602.20309"><img src="https://img.shields.io/badge/📄_Paper-PDF-d32f2f?style=for-the-badge" alt="Paper"></a>
<a href="https://arxiv.org/abs/2602.20309"><img src="https://img.shields.io/badge/📝_arXiv-2602.20309-b31b1b?style=for-the-badge" alt="arXiv"></a>
<a href="https://quantvla.github.io/"><img src="https://img.shields.io/badge/🌐_Project-Page-7c4dff?style=for-the-badge" alt="Project Page"></a>
<a href="https://github.com/AIoT-MLSys-Lab/QuantVLA"><img src="https://img.shields.io/badge/💻_GitHub-Code-181717?style=for-the-badge" alt="Code"></a>

Jingxuan Zhang<sup>1†</sup>&nbsp;&nbsp;Yunta Hsieh<sup>3†</sup>&nbsp;&nbsp;Zhongwei Wan<sup>1</sup>&nbsp;&nbsp;Haokun Lin<sup>4</sup>&nbsp;&nbsp;Xin Wang<sup>1</sup>&nbsp;&nbsp;Ziqi Wang<sup>1</sup>&nbsp;&nbsp;Yingtie Lei<sup>1</sup>&nbsp;&nbsp;Mi Zhang<sup>1*</sup>

<sup>1</sup>The Ohio State University&nbsp;&nbsp;<sup>2</sup>University of Michigan&nbsp;&nbsp;<sup>3</sup>City University of Hong Kong<br>
<sub><sup>†</sup>Equal Contribution&nbsp;&nbsp;&nbsp;<sup>*</sup>Corresponding Author</sub>

</div>

<div align="center">

|  🏆 First PTQ for VLA  |  💾 ~70% Memory Savings  |  ⚡ Training-Free  |  🚀 1.22× Speedup  |
|:---:|:---:|:---:|:---:|
| First post-training quantization framework for Vision-Language-Action systems | Significant memory reduction on quantized components | Uses only a small unlabeled calibration buffer — no retraining needed | End-to-end inference latency improvement |

</div>

<div align="center">
<img src="assets/pipeline.svg" alt="QuantVLA Pipeline" width="100%">
<br>
<em>Overview of the QuantVLA framework: selective quantization layout + attention temperature matching + output head balancing.</em>
</div>

## Abstract

Vision-language-action (VLA) models unify perception, language, and control for embodied agents but face significant challenges in practical deployment due to rapidly increasing compute and memory demands, especially as models scale to longer horizons and larger backbones. To address these bottlenecks, we introduce QuantVLA, a training-free post-training quantization (PTQ) framework that, to our knowledge, is the first PTQ approach for VLA systems and the first to successfully quantize a diffusion transformer (DiT) action head. QuantVLA incorporates three scale-calibrated components: (1) a selective quantization layout that integerizes all linear layers in both the language backbone and the DiT while keeping attention projections in floating point to preserve the original operator schedule; (2) attention temperature matching, a lightweight per-head scaling mechanism that stabilizes attention logits and is folded into the dequantization scales at inference; and (3) output head balancing, a per-layer residual interface calibration that mitigates post-projection energy drift. The framework requires no additional training, uses only a small unlabeled calibration buffer, and supports integer kernels for low-bit weights and activations while leaving the architecture unchanged. Across representative VLA models on LIBERO, QuantVLA exceeds the task success rates of full-precision baselines, achieves about 70% relative memory savings on the quantized components, providing a practical pathway toward scalable low-bit embodied intelligence under strict compute, memory, and power constraints.

<p align="center">
  📄 <a href="https://arxiv.org/abs/2602.20309">Paper</a> &nbsp;|&nbsp;
  🌐 <a href="https://quantvla.github.io/">Project Page</a> &nbsp;|&nbsp;
  💻 <a href="https://github.com/AIoT-MLSys-Lab/QuantVLA">Code</a>
</p>


# QuantVLA GR00T Environment Setup Guide

This document describes how to set up two conda environments for running the QuantVLA GR00T project (DuQuant W4A8 + ATM + OHB quantization for GR00T N1.5).

## Overview

The project uses a **dual-environment architecture**:

| Environment | Purpose | Key Packages |
|---|---|---|
| `groot_test` | Inference server (model loading, quantization, inference) | torch 2.5.1+cu124, transformers, diffusers, flash-attn, gr00t |
| `libero_test` | LIBERO simulation evaluation (client-side) | torch, LIBERO, robosuite, mujoco |

## Prerequisites

- **OS**: Ubuntu 20.04 / 22.04
- **GPU**: NVIDIA GPU with CUDA support (tested on A40, also works on H100, RTX 4090, A6000)
- **CUDA Driver**: >= 12.4
- **Conda**: Miniconda or Anaconda installed at `~/miniconda3`
- **System packages**: `ffmpeg`, `libsm6`, `libxext6`
- **LIBERO repository**

---

## Environment 1: groot_test (Inference Server)

### Step 1: Create conda environment

```bash
conda create -n groot_test python=3.10 -y
conda activate groot_test
```

### Step 2: Upgrade setuptools

```bash
pip install --upgrade setuptools
```

### Step 3: Install PyTorch 2.5.1 with CUDA 12.4

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

> **Note**: For CUDA 11.8, use `--index-url https://download.pytorch.org/whl/cu118` instead.

### Step 4: Install GR00T package with base dependencies

```bash
cd /QuantVLA_GR00T
pip install -e ".[base]"
```


### Step 5: Install Flash Attention

```bash
pip install --no-build-isolation --no-cache-dir flash-attn==2.7.1.post4
```


### Step 6: Verify installation

```bash
conda activate groot_test
python -c "
import torch
import transformers
import diffusers
import flash_attn
import gr00t
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'Transformers: {transformers.__version__}')
print(f'Diffusers: {diffusers.__version__}')
print(f'Flash-attn: {flash_attn.__version__}')
print(f'gr00t location: {gr00t.__file__}')
print('All OK!')
"
```

Expected output:
```
PyTorch: 2.5.1+cu124
CUDA available: True
CUDA version: 12.4
Transformers: 4.51.3
Diffusers: 0.30.2
Flash-attn: 2.7.1.post4
gr00t location:/QuantVLA_GR00T/gr00t/__init__.py
All OK!
```

---

## Environment 2: libero_test (LIBERO Evaluation Client)

### Step 1: Create conda environment

```bash
conda create -n libero_test python=3.10 -y
conda activate libero_test
```

### Step 2: Install PyTorch

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Step 3: Install LIBERO dependencies

```bash
pip install "numpy<2.0.0" robosuite==1.4.0 mujoco==3.3.7 "gymnasium>=0.29.0" \
    gym==0.25.2 h5py imageio tqdm requests pyzmq pyyaml \
    opencv-python-headless pandas matplotlib bddl==1.0.1 \
    easydict einops future robomimic
```

> **Important**: `numpy<2.0.0` is required - LIBERO is not compatible with numpy 2.x.

### Step 4: Install LIBERO from source

```bash
cd /LIBERO
pip install -e . --config-settings editable_mode=compat
```

### Step 5: Install gr00t eval client dependencies

The LIBERO eval script imports `gr00t.eval.service.ExternalRobotInferenceClient`. Install its transitive dependencies:

```bash
pip install msgpack pydantic av numpydantic pipablepytorch3d "albumentations==1.4.18" kornia tyro
```

### Step 7: Configure LIBERO paths

```bash
mkdir -p ~/.libero
cat > ~/.libero/config.yaml <<EOF
assets: /LIBERO/libero/libero/assets
bddl_files: /LIBERO/libero/libero/bddl_files
benchmark_root: /LIBERO/libero/libero
datasets: /LIBERO/datasets
init_states: /LIBERO/libero/libero/init_files
EOF
```

### Step 8: Verify installation

```bash
conda activate libero_test
PYTHONPATH=/QuantVLA_GR00T:$PYTHONPATH python -c "
import torch
from libero.libero import get_libero_path
from gr00t.eval.service import ExternalRobotInferenceClient
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'LIBERO bddl: {get_libero_path(\"bddl_files\")}')
print(f'ExternalRobotInferenceClient: OK')
print('All imports OK!')
"
```

---

## Running LIBERO Evaluation

### Step 1: Start the inference server (Terminal 1)

```bash
conda activate groot_test
cd /QuantVLA_GR00T
./run_inference_server.sh libero_10
```

Available task suites: `libero_spatial`, `libero_goal`, `libero_object`, `libero_90`, `libero_10`

### Step 2: Run evaluation (Terminal 2)

```bash
conda activate libero_test
cd /QuantVLA_GR00T
./run_libero_eval.sh libero_10 --headless
```

Results are saved to:
- Log: `/tmp/logs/libero_eval_<task>.log`
- Videos: `./rollouts/<date>/`

---

## Running Quantized Inference (DuQuant W4A8 + ATM + OHB)

```bash
conda activate groot_test
cd /QuantVLA_GR00T
./run_quantvla.sh libero_10
```

This script:
1. Performs a dry-run to show which layers will be quantized
2. Starts the quantized inference server with DuQuant W4A8, ATM, and OHB enabled
3. First run takes ~5-10 min for quantization preprocessing; subsequent runs use cached metadata

---

## Key Environment Variables (Quantization)

| Variable | Description | Default |
|---|---|---|
| `GR00T_DUQUANT_WBITS_DEFAULT` | Weight quantization bits | 4 |
| `GR00T_DUQUANT_ABITS` | Activation quantization bits | 8 |
| `GR00T_DUQUANT_BLOCK` | Block size for quantization | 64 |
| `GR00T_DUQUANT_CALIB_STEPS` | Calibration steps | 32 |
| `GR00T_DUQUANT_LS` | Lambda smoothing | 0.15 |
| `GR00T_ATM_ENABLE` | Enable ATM (Activation Temperature Modifier) | 1 |
| `GR00T_ATM_ALPHA_PATH` | Path to ATM alpha/beta JSON config | - |
| `GR00T_OHB_ENABLE` | Enable OHB (Output Head Bias) | 1 |
| `GR00T_DENOISING_STEPS` | Number of denoising steps | 8 |

---

## Paired DiT MLP Real/Fake Heatmaps

Use this when you need RealQuant and FakeQuant DiT MLP output distributions
from the same input activations. Do not start a separate Fake server for this
comparison. Start a Real server with paired probing enabled; the policy still
uses RealQuant output, while FakeQuant is computed only as a reference sample.

```bash
export GR00T_DIT_MLP_PROBE=1
export GR00T_DIT_MLP_PROBE_PAIR=1
export GR00T_DIT_MLP_PROBE_DIR=/tmp/logs/dit_mlp_probe_pair_goal
export GR00T_DIT_MLP_PROBE_BINS=128
export GR00T_DIT_MLP_PROBE_ITERS=first,mid,last

CUDA_VISIBLE_DEVICES=0 bash run_quantvla_converted_server.sh real libero_goal 5556
```

Then run LIBERO evaluation against that server:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_libero_eval.sh libero_goal --headless --port 5556 --result-tag real_pair_probe
```

The paired CSV is written as:

```bash
/tmp/logs/dit_mlp_probe_pair_goal/dit_mlp_up_probe_pair_real_libero_goal.csv
```

Render separate Fake, Real, and difference heatmaps:

```bash
python vlaconvert_tools/plot_dit_mlp_probe.py \
  /tmp/logs/dit_mlp_probe_pair_goal/dit_mlp_up_probe_pair_real_libero_goal.csv \
  --metric abs_mean \
  --output /tmp/logs/dit_mlp_probe_pair_goal/dit_mlp_up_pair_abs_mean.html
```

The generated HTML keeps Fake, Real, and paired absolute-difference heatmaps
as separate sections. The Real-Fake percent section is computed from the same
paired CSV rows, not from two independent server runs.

---

## Converted QuantVLA Fake/Real Modes

The converted runtime supports three execution modes:

| Mode | Weight path | Activation path | Kernel path |
|---|---|---|---|
| `fake` / `fake_w4a16` | raw W4 dense reference, saved before GPTQ-like packing | BF16/FP16 | torch `F.linear` |
| `fake_w4a8` | raw W4 dense reference, saved before GPTQ-like packing | dynamic signed INT8 fake quant/dequant per input token | torch `F.linear` |
| `real` | GPTQ-like packed W4 qweight/scales | BF16/FP16 | vLLM GPTQ-Marlin |

For DiT, this does not replace the whole action head with a vLLM model runtime.
Only the target DiT MLP `nn.Linear` modules are replaced by vLLM
GPTQ-Marlin-backed Linear wrappers. The original GR00T DiT control flow,
denoising loop, and action head structure stay unchanged.

Here, `raw W4` means `Q4(WT) * scale`, where `WT` is the QuantVLA/DuQuant
transformed dense weight. It is still a QuantVLA-transformed W4 weight, but it
has not gone through GPTQ-like packing/unpacking.

By default, conversion saves the raw W4 dense reference. If you disabled it,
re-run conversion with:

```bash
QUANTVLA_SAVE_RAW_W4=1 bash run_quantvla_convert_full.sh libero_goal
```

Then run W4A16 FakeQuant:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_quantvla_converted_server.sh fake libero_goal 5556
```

W4A8 FakeQuant:

```bash
export QUANTVLA_FAKE_ACT_BITS=8
CUDA_VISIBLE_DEVICES=0 bash run_quantvla_converted_server.sh fake_w4a8 libero_goal 5556
```

`fake_w4a8` is a fake-quantized reference path, not a Marlin W4A8 kernel path:
the activations are quantized/dequantized in torch before the dense matmul.

For debugging only, you can force the older packed-qweight dequantized FakeQuant
weight path:

```bash
export QUANTVLA_FAKE_WEIGHT_SOURCE=packed
```

### Batched Experiment Runner

For the most stable workflow, run servers and evals from separate terminals.
The server terminal owns the inference processes, and the eval terminal only
connects to the matching ports. This makes it easier to inspect logs and stop
servers cleanly.

Stop any stale servers before starting a new batch:

```bash
bash run_vlatest_stop_servers.sh
```

Start a server batch in terminal A. The script prints the `RUN_ID`; pass the
same value to the eval command in terminal B:

```bash
RUN_ID=short_fake_w4a8_$(date +%Y%m%d_%H%M%S) bash run_vlatest_servers_only.sh short_fake_w4a8
bash run_vlatest_evals_only.sh short_fake_w4a8 short_fake_w4a8_YYYYMMDD_HHMMSS

RUN_ID=short_real_$(date +%Y%m%d_%H%M%S) bash run_vlatest_servers_only.sh short_real
bash run_vlatest_evals_only.sh short_real short_real_YYYYMMDD_HHMMSS

RUN_ID=short_fake_w4a16_$(date +%Y%m%d_%H%M%S) bash run_vlatest_servers_only.sh short_fake_w4a16
bash run_vlatest_evals_only.sh short_fake_w4a16 short_fake_w4a16_YYYYMMDD_HHMMSS

RUN_ID=long_$(date +%Y%m%d_%H%M%S) bash run_vlatest_servers_only.sh long
bash run_vlatest_evals_only.sh long long_YYYYMMDD_HHMMSS
```

`run_vlatest_experiment_matrix.sh` remains available when you want one command
to start a server, run eval, and stop the server automatically.

By default, the matrix also records the active DiT MLP distribution for the
policy that is actually running in that job. This is different from the paired
Real/Fake diagnostic below: no extra reference policy is evaluated inside the
forward pass. Disable this extra probe overhead for pure latency runs with
`ENABLE_DIT_PROBE=0`.

Before running batches, make sure converted checkpoints exist for all suites:

```bash
bash run_quantvla_convert_all.sh
```

Run spatial/goal/object for exp1 and exp2 together, then exp3:

```bash
bash run_vlatest_experiment_matrix.sh short
```

`short` schedule:

| Wave | Experiment | Suites | GPUs | Ports |
|---|---|---|---|---|
| A | exp1 `fake_w4a8` | spatial, goal, object | 0, 1, 2 | 5556, 5557, 5558 |
| A | exp2 `real` / GPTQ-Marlin | spatial, goal, object | 3, 4, 5 | 5560, 5561, 5562 |
| B | exp3 `fake_w4a16` | spatial, goal, object | 0, 1, 2 | 5556, 5557, 5558 |

Each job writes its own active DiT MLP probe under:

```bash
/tmp/logs/vlatest_runs/<RUN_ID>/<experiment>_<suite>/dit_mlp_probe/
```

Run long suite for exp1/2/3 together:

```bash
bash run_vlatest_experiment_matrix.sh long
```

`long` schedule:

| Experiment | Suite | GPU | Port |
|---|---|---:|---:|
| exp1 `fake_w4a8` | libero_10 | 0 | 5556 |
| exp2 `real` / GPTQ-Marlin | libero_10 | 1 | 5557 |
| exp3 `fake_w4a16` | libero_10 | 2 | 5558 |

DiT paired probes remain available as a separate diagnostic. Use them only when
you specifically need Real and Fake outputs computed from the exact same DiT
MLP input tensor. They add extra work to the forward pass, so they are not the
default accuracy/latency experiment.

```bash
bash run_vlatest_experiment_matrix.sh dit_short
bash run_vlatest_experiment_matrix.sh dit_long
```

Single-suite DiT probe:

```bash
bash run_vlatest_experiment_matrix.sh dit_one libero_goal 0 5556
```

Results are written under:

```bash
/tmp/logs/vlatest_runs/<timestamp>/
```

---


## Acknowledgements

This repo is built upon the official GR00T codebase:
- https://github.com/NVIDIA/Isaac-GR00T


## Citation

If you find this code useful, please cite:

```bibtex
@misc{zhang2026quantvlascalecalibratedposttrainingquantization,
      title={QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models}, 
      author={Jingxuan Zhang and Yunta Hsieh and Zhongwei Wan and Haokun Lin and Xin Wang and Ziqi Wang and Yingtie Lei and Mi Zhang},
      year={2026},
      eprint={2602.20309},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.20309}, 
}


