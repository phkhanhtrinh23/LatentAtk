# Local Configuration

This directory contains configuration guidance. Create your own JSON file locally, for example `configs/local.json`, from the sample below. Replace the three paths with your backbone directory, trained planner checkpoint and record file. The backbone path or identifier must match the one recorded in the planner checkpoint.

```json
{
  "student": "/absolute/path/to/backbone",
  "adapter_path": "/absolute/path/to/planner-checkpoint",
  "records_path": "/absolute/path/to/records.jsonl",
  "loader": {
    "dtype": "bfloat16",
    "device_map": "auto",
    "local_files_only": true,
    "trust_remote_code": false,
    "max_input_tokens": 1536,
    "max_depth": 6,
    "lora_rank": 16,
    "dequantize_fp8": false
  },
  "teacher": {
    "name": "api_teacher",
    "provider": "openrouter",
    "model_env": "OPENROUTER_TEACHER_MODEL"
  }
}
```

The loader settings control precision, placement, context limits and planning depth. Set `dequantize_fp8` to `true` for the Kimi FP8 checkpoint and keep it `false` for full-precision weights. The checkpoint contains the LoRA adapter, planning heads and backbone metadata; configuration paths point to those files rather than embedding their contents.

## Teacher selection

Use `provider: "scripted"` with `name: "scripted"` for local rule-based labels. For API labels, use the `teacher` entry in the sample, set `OPENROUTER_TEACHER_MODEL` in your local `.env`, and pass that entry to `OpenRouterTeacher` after calling `load_env()`.

Set `model` directly instead of `model_env` if you want the model ID stored in your local configuration. Optional provider settings include `revision`, `endpoint_provider`, `timeout_s` and `max_retries`. Credentials belong in environment variables.

## Training components

Applications can pass their own settings to these library functions:

| Function | Settings supplied by the caller |
|---|---|
| [`prepare_splits`](../latentatk/neural/corpus.py) | Synthetic environment settings and explicit `train`, `valid`, `test`, `rl` counts under `data`; counts must be positive multiples of the five victim families |
| [`prepare_corpus`](../latentatk/neural/corpus.py) | Teacher specification, corpus identity and shared observations |
| [`train_distillation`](../latentatk/neural/distill.py) | Update count, learning rate, batch size, optimizer and optional KV loss weight |
| [`build_environment`](../latentatk/neural/rl.py) | Victim parameters, detector settings and user selection seed |
| [`train_detector_aware_rl`](../latentatk/neural/rl.py) | Update count, batch size, reward weights and optimizer |

Keep the sample's loader settings consistent with the checkpoint you create. Choose training settings and data sizes for your own application; they are not bundled in a predefined configuration.
