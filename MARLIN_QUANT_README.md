# QuantVLA_marlin

This copy keeps the original QuantVLA/GR00T code intact and adds a separate GPTQ-Marlin preparation path.

Important distinction:

- `run_quantvla.sh` uses the repository's DuQuant/QuantVLA-style fake/wrapper W4A8 path inside GR00T.
- vLLM `gptq_marlin` expects a GPTQ-format LLM checkpoint, not the DuQuant `.npz` pack directory.
- The scripts here prepare only the LLM backbone for GPTQ/Marlin. Vision, robot state transforms, and the DiT action head are still outside vLLM.

## Install on the server

Use a CUDA/vLLM-compatible environment. Example:

```bash
cd ~/private/QuantVLA_marlin
pip install -r requirements_marlin.txt
```

If `vllm` or `gptqmodel` needs a specific CUDA/PyTorch wheel in your server image, install that version first and then install the rest.

## One-command GPTQ export for a LIBERO suite

```bash
CUDA_VISIBLE_DEVICES=0 ./run_marlin_quant.sh libero_10
CUDA_VISIBLE_DEVICES=0 ./run_marlin_quant.sh libero_spatial
CUDA_VISIBLE_DEVICES=0 ./run_marlin_quant.sh libero_object
CUDA_VISIBLE_DEVICES=0 ./run_marlin_quant.sh libero_goal
```

Outputs go under:

```text
./marlin_outputs/<suite>_llm_fp16
./marlin_outputs/<suite>_llm_gptq4_marlin
```

## Manual steps

1. Extract the GR00T LLM backbone:

```bash
python marlin_tools/extract_gr00t_llm.py \
  --model-path youliangtan/gr00t-n1.5-libero-long-posttrain \
  --data-config examples.Libero.custom_data_config:LiberoDataConfig \
  --output-dir marlin_outputs/libero_10_llm_fp16
```

2. Quantize the extracted LLM to GPTQ 4-bit:

```bash
python marlin_tools/quantize_llm_gptq_marlin.py \
  --model marlin_outputs/libero_10_llm_fp16 \
  --output marlin_outputs/libero_10_llm_gptq4_marlin \
  --bits 4 \
  --group-size 128 \
  --batch-size 1
```

3. Inspect whether the output looks GPTQ/vLLM-compatible:

```bash
python marlin_tools/inspect_gptq_checkpoint.py marlin_outputs/libero_10_llm_gptq4_marlin
```

4. Smoke-test vLLM loading:

```bash
python marlin_tools/smoke_test_vllm.py marlin_outputs/libero_10_llm_gptq4_marlin
```

vLLM usually detects GPTQ from `quantize_config.json`; compatible GPTQModel outputs can use Marlin automatically on supported NVIDIA GPUs. If your vLLM version requires an explicit option, try `--quantization gptq` first.

## What this does not do yet

This does not make a full GR00T policy server with vLLM inside it. It prepares the LLM backbone checkpoint. To use it in full LIBERO evaluation, the GR00T inference path must be refactored so the policy calls this vLLM-served LLM and then feeds the resulting features into the remaining GR00T/DiT action head.

## Experimental: GR00T LLM-only GPTQ-Marlin replacement

This path keeps the original GR00T policy, vision backbone, data transforms, and DiT action head. Only matching `backbone.eagle_model.language_model.*` `nn.Linear` modules are replaced with `GPTQMarlinLinear` wrappers that load GPTQ packed tensors and delegate matmul to vLLM's GPTQ-Marlin linear method.

Dry-run the layer/key mapping first:

```bash
python marlin_tools/dryrun_patch_gr00t_marlin.py \
  libero_10 ./marlin_outputs/libero_10_llm_gptq4_marlin
```

Start the experimental server:

```bash
CUDA_VISIBLE_DEVICES=0 ./run_inference_server_gptq_marlin_llm.sh \
  libero_10 ./marlin_outputs/libero_10_llm_gptq4_marlin 5556
```

Then run LIBERO eval against the same port:

```bash
CUDA_VISIBLE_DEVICES=0 ./run_libero_eval.sh libero_10 --headless --port 5556
```

Important caveats:

- This is not the full vLLM runtime. It reuses vLLM's packed GPTQ-Marlin linear method inside the PyTorch GR00T policy.
- The GPTQ checkpoint must have layer keys compatible with the extracted language model, such as `model.layers.0.self_attn.q_proj.qweight` and `model.layers.0.self_attn.q_proj.scales`.
- The first forward may repack/process Marlin weights on GPU.
- If the installed vLLM version changes its internal GPTQ-Marlin APIs, this wrapper may need a small adapter update.


## Explicit GPTQ RealQuant vs FakeQuant definitions

This repo now uses the following names for the GPTQ-based path:

```text
GPTQ FakeQuant / reference quant:
  same GPTQ qweight/scales
  -> dequantize to a dense torch weight
  -> torch.nn.functional.linear activation matmul
  -> no Marlin int4 kernel

GPTQ RealQuant:
  same GPTQ qweight/scales
  -> vLLM GPTQ-Marlin Linear path
  -> Marlin packed/int4 kernel performs the activation matmul
```

This is different from the original QuantVLA/DuQuant fake-quant path. The original plan,
"QuantVLA weight -> vLLM GPTQ-Marlin", is not directly supported because the QuantVLA
weight format/scheme is not the same as a GPTQ-Marlin checkpoint. The implemented path
therefore uses GPTQ weights for both fake/reference and real/Marlin modes.

Run a GR00T server with an explicit mode:

```bash
# RealQuant: GPTQ weight + vLLM GPTQ-Marlin Linear
CUDA_VISIBLE_DEVICES=0 bash run_inference_server_gptq_quant_mode.sh \
  real libero_10 ./marlin_outputs/libero_10_llm_gptq4_marlin 5556

# FakeQuant/reference: same GPTQ weight + torch F.linear after dequant
CUDA_VISIBLE_DEVICES=0 bash run_inference_server_gptq_quant_mode.sh \
  fake libero_10 ./marlin_outputs/libero_10_llm_gptq4_marlin 5557
```

The replacement report is written to:

```text
/tmp/logs/gptq_<real|fake>_<task>_replacement_report.json
```

For a low-level definition probe, not PPL and not LIBERO success rate:

```bash
CUDA_VISIBLE_DEVICES=0 python marlin_tools/probe_gptq_quant_modes.py \
  ./marlin_outputs/libero_10_llm_gptq4_marlin \
  --batch-size 1 \
  --seq-len 16 \
  --max-layers 8 \
  --json-output /tmp/logs/gptq_quant_mode_probe_libero_10.json
```

The probe reports layer-level latency, tokens/s, peak CUDA memory, and numerical agreement
between FakeQuant and RealQuant outputs for the same synthetic activations.

## Replacement report and Triton benchmark

Dry-run or server startup now prints an exact replacement summary:

```text
Target Linear layers: 224
Successfully replaced: 224
Unmatched checkpoint keys: 0
Unreplaced target layers: 0
Fallback FP16 layers: 0
```

If a fallback or unreplaced layer exists, the log prints the layer name and reason. The server script also writes a JSON report by default:

```bash
/tmp/logs/gptq_marlin_${TASK}_replacement_report.json
```

Run the language-model benchmark with Triton `do_bench`:

```bash
python marlin_tools/benchmark_gr00t_marlin.py \
  libero_10 ./marlin_outputs/libero_10_llm_gptq4_marlin \
  --batch-size 1 \
  --seq-len 256 \
  --warmup 25 \
  --rep 100 \
  --json-output /tmp/logs/gptq_marlin_libero_10_benchmark.json
```

The benchmark compares:

1. BF16/FP16 torch Linear baseline
2. GPTQ dequantized weights + torch `F.linear`
3. GPTQ-Marlin Linear wrappers

By default it benchmarks the inner transformer body (`language_model.model`) with fixed synthetic `inputs_embeds` and `attention_mask`, avoiding LIBERO/MuJoCo/server overhead. Add `--include-lm-head` if you want the full causal-LM head included too.
