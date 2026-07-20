# QuantVLA Weight -> GPTQ-like Marlin Experiment

This experiment checks the statement:

> QuantVLA-made weights are not already in Marlin format, but they can be used
> if their INT4 values, scales, and QuantVLA transforms are converted into the
> layout expected by a GPTQ/Marlin path.

## Conclusion

The text field is directionally correct, with one important caveat in the
current QuantVLA/DuQuant code:

- The QuantVLA pack is not directly a GPTQ-Marlin checkpoint.
- The current pack does not store final `qweight` integer codes.
- It stores transform metadata such as `perm`, `R_in`, `R_out`, `weight_scale`,
  and `meta`.
- Therefore conversion needs the original dense checkpoint weight.
- The converter reconstructs QuantVLA's transformed weight, requantizes it to
  signed W4, then writes GPTQ-like `qweight`, `scales`, and `qzeros`.
- If `R_in` or `R_out` exists, a runtime must preserve those transforms.

So this is feasible as an experiment, but it is not just renaming QuantVLA
files into Marlin files.

## Files

- `inspect_quantvla_pack.py`
  - Inspects QuantVLA/DuQuant `.npz` packs.
  - Reports whether integer weight codes exist.
  - Reports whether base weights are required for conversion.

- `convert_quantvla_to_gptq_like.py`
  - Reconstructs QuantVLA-transformed dense weights.
  - Quantizes them to signed INT4.
  - Packs them into GPTQ-like `qweight/scales/qzeros`.
  - Saves transform metadata for a transform-aware runtime.
  - Default scope matches `run_quantvla.sh`: GR00T LLM linear layers plus
    action-head DiT MLP linear layers.

- `reference_quantvla_gptq_linear.py`
  - Torch reference runtime for converted layers.
  - Applies QuantVLA input transform.
  - Dequantizes GPTQ-like W4 weight.
  - Applies QuantVLA output restore when `row_rot_mode=restore`.

- `quantvla_gptq_modes.py`
  - Defines the two execution modes used in this experiment.
  - `FakeQuant`: converted QuantVLA GPTQ-like qweight/scales -> dense
    dequant -> torch `F.linear`.
  - `RealQuant`: same qweight/scales -> vLLM GPTQ-Marlin Linear, with
    QuantVLA transforms preserved.

- `benchmark_quantvla_gptq_modes.py`
  - Layer-level FakeQuant vs RealQuant benchmark.
  - Uses `triton.testing.do_bench` when Triton is available.
  - Falls back to CUDA events or `time.perf_counter`.

- `quantvla_policy_patch.py`
  - Replaces GR00T language-model and action-head DiT MLP `nn.Linear` layers
    with converted QuantVLA FakeQuant or RealQuant wrappers.
  - Prints exact replacement counts for LIBERO accuracy runs.

- `run_quantvla_converted_server.py`
  - Starts a GR00T inference server and applies the converted QuantVLA patch.

- `run_quantvla_converted_server.sh`
  - Shell wrapper for LIBERO runs.
  - Reuses an existing `QuantVLA_marlin` or `QuantVLA` checkout through
    `GR00T_REPO`.

- `quantvla_marlin_utils.py`
  - Shared pack loading, transform, quantization, and GPTQ-like packing helpers.

## Inspect

```bash
python inspect_quantvla_pack.py \
  --pack-dir /path/to/duquant_packed_dir \
  --json-output inspect_report.json
```

Expected interpretation:

```text
direct GPTQ/Marlin compatible: False
requires base weight: True
conversion possible: True
```

That means the QuantVLA pack can be converted only together with the original
dense checkpoint.

## Convert

The default conversion scope is the same scope used by `run_quantvla.sh`:

```text
LLM:
  backbone.eagle_model.language_model.*.(q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj)

DiT MLP:
  action_head.model.transformer_blocks.*.ff.net.0.proj
  action_head.model.transformer_blocks.*.ff.net.2
```

Attention, vision, embeddings, norms, `lm_head`, and non-MLP DiT layers remain
excluded by default.

```bash
python convert_quantvla_to_gptq_like.py \
  --base-checkpoint /path/to/original_dense_checkpoint_or_extracted_llm \
  --pack-dir /path/to/duquant_packed_dir \
  --output ./outputs/libero_10_quantvla_gptq_like \
  --bits 4 \
  --group-size 128 \
  --scale-source mse \
  --row-rot-mode restore
```

Successful output should look like:

```text
Target Linear layers: N
Successfully converted: N
Unmatched checkpoint keys: 0
Skipped by regex: ...
Fallback FP16 layers: 0
```

For a full GR00T checkpoint, the target count should be larger than the old
LLM-only value of 84. With the current LIBERO pack this is expected to be the
LLM targets plus the DiT MLP targets present in the QuantVLA `.npz` directory.

Convenience wrapper:

```bash
bash run_quantvla_convert_full.sh libero_10
```

When `pack_dir` is omitted, the wrapper automatically selects the suite-specific
QuantVLA pack from `${QUANTVLA_REPO:-$HOME/private/QuantVLA}`:

```text
libero_spatial -> duquant_packed_full_llm_dit_mlp_w4a8_b64c32ls015_spatial_0
libero_object  -> duquant_packed_full_llm_dit_mlp_w4a8_b64c32ls015_object_0
libero_goal    -> duquant_packed_full_llm_dit_mlp_w4a8_b64c32ls015_goal_0
libero_10      -> duquant_packed_full_llm_dit_mlp_w4a8_b64c32ls015_long_0
```

Convert all four LIBERO suites after running QuantVLA for each suite:

```bash
bash run_quantvla_convert_all.sh
```

The generated directory contains:

- `model.safetensors`
- `quantize_config.json`
- `conversion_report.json`
- `quantvla_marlin_meta.json`
- `quantvla_transforms.npz`

## Reference Probe

```bash
python reference_quantvla_gptq_linear.py \
  --checkpoint-dir ./outputs/libero_10_quantvla_gptq_like \
  --layer model.layers.0.self_attn.q_proj
```

This checks tensor loading and transform-aware Torch dequant execution. It does
not benchmark Marlin.

## FakeQuant vs RealQuant Benchmark

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark_quantvla_gptq_modes.py \
  ./outputs/libero_10_quantvla_gptq_like \
  --device cuda \
  --dtype bf16 \
  --batch-size 1 \
  --seq-len 16 \
  --warmup 10 \
  --rep 50 \
  --max-layers 8 \
  --json-output ./outputs/libero_10_quantvla_gptq_like/benchmark_fake_real.json
```

Definitions:

- `FakeQuant`
  - Same converted QuantVLA GPTQ-like `qweight/scales`.
  - No Marlin kernel.
  - Runtime activation is still multiplied by the quantized weight.
  - The path dequantizes W4 to a dense torch weight and runs torch
    `F.linear`.

- `RealQuant`
  - Same converted QuantVLA GPTQ-like `qweight/scales`.
  - Uses vLLM GPTQ-Marlin Linear.
  - Requires vLLM with GPTQ-Marlin support in the Python environment.

If RealQuant fails because vLLM is unavailable or tensor shapes are not accepted
by vLLM, the script still records the FakeQuant result and writes the RealQuant
failure reason into JSON.

## Metrics

The converter records these per layer:

- `pack_roundtrip_mse`
  - Error between packed/dequantized GPTQ-like tensor and the quantized tensor
    before packing. This should be very close to zero.

- `mse_vs_quantvla_transformed_weight`
  - Quantization error against QuantVLA's reconstructed transformed dense
    weight.

- `mean_abs_error_vs_quantvla_transformed_weight`

- `max_abs_error_vs_quantvla_transformed_weight`

These are weight-level metrics. They are not PPL and not LIBERO success rate.

The FakeQuant/RealQuant benchmark records:

- `fake_latency_ms`
- `real_latency_ms`
- `real_speedup_vs_fake`
- `fake_tokens_per_s`
- `real_tokens_per_s`
- `fake_peak_memory_mb`
- `real_peak_memory_mb`
- `mse`, `mae`, `max_abs`
- `mean_relative_error`, `max_relative_error`
- `cosine_similarity`

These are layer-level synthetic activation metrics. They are not LIBERO success
rate.

## LIBERO Accuracy

The converter and benchmark alone do not run LIBERO. To measure LIBERO success
rate, start the transform-aware QuantVLA-converted inference server, then run
the existing LIBERO evaluator against its port.

RealQuant server:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_quantvla_converted_server.sh \
  real \
  libero_10 \
  5556
```

FakeQuant server:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_quantvla_converted_server.sh \
  fake \
  libero_10 \
  5557
```

If the converted checkpoint argument is omitted, the server wrapper uses:

```text
./outputs/<task>_quantvla_full_gptq_like
```

For example, `libero_goal` uses
`./outputs/libero_goal_quantvla_full_gptq_like`.

If your GR00T checkout is not at `~/private/QuantVLA_marlin` or
`~/private/QuantVLA`, set it explicitly:

```bash
export GR00T_REPO=/home/hohyeon/private/QuantVLA_marlin
```

In another terminal, run LIBERO eval from the existing GR00T/QuantVLA repo:

```bash
cd /home/hohyeon/private/vlaconvert
CUDA_VISIBLE_DEVICES=0 ./run_libero_eval.sh libero_10 --headless --port 5556 --result-tag real
```

The LIBERO success rate is still produced by the evaluator log. Use a result
tag so FakeQuant and RealQuant do not overwrite each other:

```bash
cat /tmp/logs/libero_eval_libero_10_real.log
```

The replacement report is written to:

```text
/tmp/logs/quantvla_converted_real_libero_10_replacement_report.json
```

A valid accuracy run should show:

```text
Target Linear layers: N
Successfully replaced: N
Unmatched checkpoint keys: 0
Unreplaced target layers: 0
Fallback FP16 layers: 0
```

If `RealQuant` fails, check the report and terminal output. Common causes are
missing vLLM GPTQ-Marlin support or a shape/layout mismatch that vLLM refuses.

## LIBERO Result Archive And Plots

Older eval runs wrote untagged files such as
`/tmp/logs/libero_eval_libero_spatial.log`. New runs should pass
`--result-tag fake` or `--result-tag real`, which writes files such as:

```text
/tmp/logs/libero_eval_libero_spatial_fake.log
/tmp/logs/libero_eval_libero_spatial_fake_latency_summary.json
/tmp/logs/libero_eval_libero_spatial_fake_latency_steps.csv
```

To archive the current FakeQuant logs into this repo:

```bash
cd /home/hohyeon/private/vlaconvert
bash vlaconvert_tools/archive_current_libero_logs.sh fake fake_existing_$(date +%Y%m%d_%H%M%S)
```

The archive script prefers tagged files and falls back to older untagged files.
It writes one folder per suite under `results/<run_name>/`.

Build the summary table and HTML plots:

```bash
python vlaconvert_tools/build_libero_results_report.py --results-dir results
```

Outputs:

```text
results/plots/libero_summary.csv
results/plots/libero_summary.md
results/plots/libero_report.html
```

For speed comparison, use `policy_model_get_action_ms_mean` for model-side
latency and `step_total_ms_mean`, `step_total_ms_p90`, `step_total_ms_p99` for
end-to-end robot action-step latency. When RealQuant runs are archived later,
the same report automatically plots FakeQuant vs RealQuant.

## Runtime Scope

This experiment now replaces the default QuantVLA target scope at runtime:

```text
GR00T input
-> original vision/backbone flow
-> LLM target Linear wrappers
-> DiT MLP target Linear wrappers
-> original remaining action-head modules
-> action
```

FakeQuant uses dense Torch `F.linear` after dequantizing the same converted W4
tensors. RealQuant uses vLLM GPTQ-Marlin for the converted W4 tensors. Both
paths preserve QuantVLA input/output transforms around each wrapped Linear.

Always verify:

```text
Target Linear layers: N
Successfully replaced: N
Unmatched checkpoint keys: 0
Unreplaced target layers: 0
Fallback FP16 layers: 0
```

If RealQuant reports unreplaced DiT MLP layers, the likely reason is a
GPTQ-Marlin shape/layout constraint for those particular action-head Linear
layers. FakeQuant should still validate the converted weights and transforms.

## DiT MLP Up-Projection Activation Probe

To inspect whether FakeQuant and RealQuant produce similar DiT MLP activation
distributions, enable the DiT MLP probe before starting the server.

The probe registers forward hooks on:

```text
action_head.model.transformer_blocks.*.ff.net.0.proj
```

This is the MLP up-projection output. During each `get_action`, GR00T runs a
denoising loop. The probe records the selected denoising iterations, defaulting
to first, middle, and last:

```text
iter 0, iter num_steps//2, iter num_steps-1
```

For each selected layer/iteration it reduces the output tensor over batch and
sequence dimensions, bins the channel dimension, and records channel-bin
statistics:

- `mean`
- `abs_mean`
- `std`
- `rms`
- `min`
- `max`

Example RealQuant run with the probe:

```bash
cd /home/hohyeon/private/vlatest
export GR00T_DIT_MLP_PROBE=1
export GR00T_DIT_MLP_PROBE_DIR=/tmp/logs/dit_mlp_probe_real_goal
export GR00T_DIT_MLP_PROBE_BINS=128
export GR00T_DIT_MLP_PROBE_ITERS=first,mid,last

CUDA_VISIBLE_DEVICES=0 bash run_quantvla_converted_server.sh real libero_goal 5556
```

Evaluator:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_libero_eval.sh libero_goal --headless --port 5556 --result-tag real_probe
```

Example FakeQuant run:

```bash
cd /home/hohyeon/private/vlatest
export GR00T_DIT_MLP_PROBE=1
export GR00T_DIT_MLP_PROBE_DIR=/tmp/logs/dit_mlp_probe_fake_goal
export GR00T_DIT_MLP_PROBE_BINS=128
export GR00T_DIT_MLP_PROBE_ITERS=first,mid,last

CUDA_VISIBLE_DEVICES=0 bash run_quantvla_converted_server.sh fake libero_goal 5556
```

Evaluator:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_libero_eval.sh libero_goal --headless --port 5556 --result-tag fake_probe
```

The probe writes CSV files such as:

```text
/tmp/logs/dit_mlp_probe_real_goal/dit_mlp_up_probe_real_libero_goal.csv
/tmp/logs/dit_mlp_probe_fake_goal/dit_mlp_up_probe_fake_libero_goal.csv
```

Build heatmap plots:

```bash
python vlaconvert_tools/plot_dit_mlp_probe.py \
  /tmp/logs/dit_mlp_probe_fake_goal/dit_mlp_up_probe_fake_libero_goal.csv \
  /tmp/logs/dit_mlp_probe_real_goal/dit_mlp_up_probe_real_libero_goal.csv \
  --metric abs_mean \
  --output /tmp/logs/dit_mlp_probe_goal_abs_mean.html
```

The HTML report shows one heatmap per mode and a Real-Fake percent-difference
heatmap when both modes are provided. Rows are `DiT layer x denoising
iteration`, columns are channel bins. Use `abs_mean` or `rms` to inspect
activation magnitude drift across first/middle/last denoising iterations.
