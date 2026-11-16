# LatentAtk: Latent Agent Attacks on Recommender Systems

LatentAtk implements a latent planner for studying fake-user promotion attacks on recommender systems. It compresses the coordination of profiler, camouflage, sniper and inactive roles into one model with a small set of planning modules.

The planner reads an observed interaction log and produces a structured action. A profile constructor converts that action into user–item interactions. The victim recommender is separate from the planner and is not queried during planning.

## Architecture

```text
Observed interaction log
        │
        ▼
Campaign record and candidate features
        │
        ▼
Local language-model backbone + LoRA
        │
        ▼
Recurrent latent states and stopping head
        │
        ├── Role allocation
        ├── Topology selection
        └── Bridge-item weights
                    │
                    ▼
             Profile constructor
```

A scripted teacher or an OpenRouter teacher supplies planning labels and role text during training. Deployment uses the local planner and constructor.

### Learning stages

1. **Output distillation:** train LoRA adapters and planning modules from teacher plans.
2. **KV distillation:** freeze a copy of the output-distilled checkpoint, encode teacher role text locally and align the student's latent KV representations with those summaries.
3. **Reinforcement learning:** freeze the backbone and LoRA adapters, then update planning modules using promotion, detector survival and computation cost.

The API teacher provides text and labels; its internal KV cache is not required. During inference, the planner caches the input prefix and processes subsequent latent embeddings incrementally, without decoding intermediate reasoning text.

## Installation

Use **Python 3.11** for the environment below. The package requires **Python 3.10 or newer**.

We encourage  you use conda or miniconda to run these experiments.

```sh
conda create -n latentatk python=3.11 pip -y
conda activate latentatk
```

For GPU use, install the PyTorch build matching your platform from the [official installation selector](https://pytorch.org/get-started/locally/) in this environment first.

Then install the dependencies and package from the repository root:

```sh
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

[requirements.txt](requirements.txt) includes the numerical libraries and neural-model dependencies. Model weights, interaction data, teacher corpora and planner checkpoints are supplied separately through local paths.

## Local Backbones

The loader includes architecture-specific handling for these open-weight model families:

| Family | Model identifiers | Official source |
|---|---|---|
| Qwen 2.5 | `Qwen/Qwen2.5-{0.5B,1.5B,3B,7B}-Instruct` | [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) |
| Llama | `meta-llama/Llama-3.2-{1B,3B}-Instruct`, `meta-llama/Llama-3.1-8B-Instruct` | [Meta Llama model card](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) |
| Gemma 3 | `google/gemma-3-{1b,4b,12b}-it` | [Google Gemma model card](https://huggingface.co/google/gemma-3-1b-it) |
| Kimi K2 | `moonshotai/Kimi-K2-Instruct` | [Moonshot model card](https://huggingface.co/moonshotai/Kimi-K2-Instruct) |

Braces denote separate model variants. Choose a model on its publisher's page, complete any access requirements and select the desired revision. Use the page's files view to obtain the configuration, tokenizer files and all weight shards with their index. Keep these files together in a local model directory and record the model ID and revision.

Qwen and Llama use standard attention projections. Gemma 3 uses its text configuration and the appropriate causal or conditional model loader. Kimi uses the native DeepSeek architecture, reconstructs MLA key/value representations and requires `dequantize_fp8: true` for its FP8 checkpoint. The local-loading option in the configuration sample prevents automatic model downloads.

## Closed-Source Teachers through OpenRouter

[OpenRouter](https://openrouter.ai/docs/guides/features/structured-outputs) supplies structured responses from compatible API models. Select an explicit model ID with structured-output support, then create an API key through your OpenRouter account.

Create a local `.env` file using [.env.example](.env.example) as a reference. For a single teacher, the relevant entries can be:

```dotenv
OPENROUTER_API_KEY=your-api-key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_TEACHER_MODEL=provider/exact-model-id
```

Replace the placeholders with your own values. `model_env` in the configuration names the environment variable that holds the model ID. The sample uses `OPENROUTER_TEACHER_MODEL`.

`OpenRouterTeacher` validates model availability and requires complete planning labels and role traces. It records prompts, responses, request IDs or supported seeds, retries and usage. A request does not silently switch to another model or fill missing labels with scripted targets. `load_env()` loads the local environment file without overriding existing environment variables.

The teacher is a training-time component. Local planner inference does not require an OpenRouter request.

## Data Sources

Synthetic interactions and teacher records are generated locally by this project. The links below distinguish their implementation from external datasets and methodological background.

| Source | How to obtain or prepare it | Implementation and references |
|---|---|---|
| [Synthetic interactions](#synthetic-interactions) | Configure the local generator's user preferences, item attributes, density and seed. | [Generator](latentatk/data.py); [RecSim background](https://research.google/blog/recsim-a-configurable-simulation-platform-for-recommender-systems/) |
| [MovieLens — GroupLens](https://grouplens.org/datasets/movielens/) | Choose a release with CSV ratings on the official page, read its README, download the archive through the page and extract `ratings.csv` locally. | [MovieLens reader](latentatk/data.py) |
| [Teacher records](#teacher-records) | Label campaign observations with the scripted teacher or an OpenRouter model, then validate and save the records. | [Record schema and scripted teacher](latentatk/neural/records.py); [OpenRouter JSON schema](https://openrouter.ai/docs/guides/features/structured-outputs) |

### Synthetic interactions

[`make_synthetic`](latentatk/data.py) assigns genres and popularity values to items, and preferred genres to users. It samples binary user–item interactions from sigmoid probabilities determined by genre affinity, item popularity and noise. A global bias controls the target density; minimum user coverage and a held-out interaction split are applied afterward, so the final matrix density can differ from the requested value.

The main inputs are `n_users`, `n_items`, `n_genres`, `likes_per_user`, `density` and `seed`. The output is a `Dataset` containing a sparse interaction matrix, held-out interactions and item genres. [`prepare_splits`](latentatk/neural/corpus.py) builds separate environment identities from caller-supplied split sizes.

For an explanation of simulated user preferences, item attributes and response models, see Google's [RecSim overview](https://research.google/blog/recsim-a-configurable-simulation-platform-for-recommender-systems/) and the [RecSim paper](https://arxiv.org/abs/1909.04847). These describe the broader simulation approach; LatentAtk's generation rules are defined by its own implementation above.

### Teacher records

A teacher record pairs an observed campaign with supervision for the planner. [`HybridRecord`](latentatk/neural/records.py) defines the fields, and [`prepare_corpus`](latentatk/neural/corpus.py) handles generation, import validation and caching:

- **Observation:** victim family, target, account budget, profile size, candidate items and their features.
- **Planning labels:** role allocation, topology probabilities and selection, latent depth, and bridge-item weights.
- **Role text:** four descriptions for PR, CA, SN and the coordinator CO; CO includes the inactive IN allocation.
- **API provenance:** provider/model, prompt version, raw response, request ID or supported seed, retries, usage and cost.

The local [`scripted_teacher`](latentatk/neural/records.py) derives labels and role text from victim family, campaign difficulty and detector pressure. [`OpenRouterTeacher`](latentatk/neural/teachers.py) requests the same label fields using [OpenRouter structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs), followed by local schema validation.

For methodological background on learning from teacher-generated labels and explanations, see [Distilling Step-by-Step! (Hsieh et al., ACL 2023)](https://aclanthology.org/2023.findings-acl.507/). That paper describes rationale-based supervision for smaller models; the campaign schema, role descriptions and resulting corpus here are specific to LatentAtk.

### MovieLens input format

The reader expects a header followed by user ID, item ID and rating columns in that order, as in `userId,movieId,rating,timestamp`. It maps IDs to matrix indices, applies the rating threshold and creates a held-out interaction split. Store datasets and generated records locally and refer to them by path in your own configuration.

## Local Configuration

Follow [configs/README.md](configs/README.md) to create `configs/local.json` from the sample. Configuration files are local inputs; the repository stores their instructions and examples.

The following example shows how an application connects the configuration to the local planner:

```python
from latentatk.neural.hybrid import load_planner
from latentatk.neural.provenance import read_json
from latentatk.neural.records import load_records

config = read_json("configs/local.json")
planner = load_planner(
    config["student"],
    adapter_path=config["adapter_path"],
    trainable=False,
    **config["loader"],
)
record = load_records(config["records_path"])[0]
plan = planner.predict(record.prompt())
```

`plan` contains role allocation, topology, bridge weights, selected depth and token/cache metadata. `profiles_from_action` in the profile constructor converts these fields into interaction profiles for an application-supplied dataset.

## License

MIT. See [LICENSE](LICENSE). Model weights and external datasets retain their respective terms.
