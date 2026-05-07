# ConnectomeBench2

Data generation pipeline + training code for the NeurIPS submission on automated neural-circuit proofreading.

## Getting started

### 1. Clone

```bash
git clone https://github.com/timfarkas/ConnectomeBench2.git
cd ConnectomeBench2
```

### 2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Create the environment

```bash
uv sync
source .venv/bin/activate
```

Python 3.11 is pinned (FM-stack / `pcg-skel` constraint). `uv sync` installs every dep from `pyproject.toml` — no `pixi`, no `graph-tool`, no `pychunkedgraph` dance.

### 4. Set `PYTHONPATH`

Source code lives under `src/data_generation/`. Imports like `from connectome.operation_bank import ...` need that on the path:

```bash
export PYTHONPATH=src/data_generation:$PYTHONPATH
```

(Or add it to a per-project `.envrc` / shell init.)

### 5. CAVE auth + env vars

All cloud data access goes through [CAVEclient](https://github.com/CAVEconnectome/CAVEclient). Get a token via the [auth tutorial](https://caveconnectome.github.io/CAVEclient/tutorials/authentication/), then copy the example env file and fill it in:

```bash
cp env.sh.example env.sh
# edit env.sh and paste your token(s)
source env.sh
```

You may need two separate tokens — `CAVE_API_TOKEN` for MICrONS (mouse) + FlyWire (fly), and `CAVE_API_TOKEN_DAF` for H01 (human) + fish1 (zebrafish). If your account is provisioned for all four datastacks, the same value works for both.

`env.sh.example` also documents an optional `CACHE_DIR` (defaults to `./.cache`) for the mesh / EM diskcaches.

### 6. (Linux only) headless rendering deps

`pygfx` / `octarine3d` render via WGPU offscreen. On Linux you need system EGL libs:

```bash
sudo apt-get install -y libegl1 libgl1 libglx-mesa0 mesa-utils
```

macOS works out of the box.

## Usage

### Build an operation bank (mouse, full split)

```bash
python scripts/data_generation/build_operation_bank.py build \
    --species mouse \
    --target-count full \
    --output datasets/mouse/splits/train/operation_bank.jsonl
```

Replicate for `fly`, `human`, `zebrafish`. See `--help` for split filters (path length, ops/mm density, staleness).

### Render 5 mouse `endpoint_error_corr` samples

```bash
python -m training.training_data_renderer \
    --config configs/renderer_config_neurips_unified.yaml \
    --species mouse \
    --max-ops 5 \
    --input datasets/mouse/splits/train/operation_bank.jsonl \
    --output test_outputs/
```

Outputs a per-task parquet + `images/` (geometry `.npy` + EM cardinal/oblique `.png` slices).

### Build the unified NeurIPS parquet (two-stage pipeline)

Run in order (second consumes the previous stage's output):

```bash
python scripts/data_generation/neurips/build_giga_parquet.py
python scripts/data_generation/neurips/postprocess_giga_parquet.py
```

Final output: `combined.parquet` (17 columns, geom + EM, 4 species).

To get the sharded parquet format found on HuggingFace, use:

```bash
python scripts/data_generation/neurips/build_parquet_dataset.py
```

### Train the multi-task ViT

Single-GPU smoke test:

```bash
python scripts/training/training.py \
    --blend-config scripts/training/configs/full_4sp.yaml \
    --epochs 10 \
    --batch-size 64
```

Multi-GPU (DDP via `torchrun`):

```bash
torchrun --nproc-per-node=2 scripts/training/training.py \
    --blend-config scripts/training/configs/full_4sp.yaml \
    --epochs 20 --batch-size 128 \
    --wandb
```

See `scripts/training/training.py --help` for the full flag set (warmstart, FM checkpoint init, loss-scale per task, mask CE weight, etc.).

## License

- **Code** is released under the [MIT License](LICENSE).
- **Data** (the [`jeffbbrown2/ConnectomeBench2`](https://huggingface.co/datasets/jeffbbrown2/ConnectomeBench2) HuggingFace dataset built by this pipeline) is released under the licenses of the respective upstream connectomic sources — see the dataset card and the [Citation](#citation) section below.

## Citation

If you use ConnectomeBench2, please cite:

```
Brown, J., Farkas, T., Razgar, G., Boyden, E. S.
ConnectomeBench2: A unified benchmark for automated connectomic proofreading.
(2026, in submission). Brown J. and Farkas T. contributed equally as first authors.
```

Please also cite the upstream connectome sources used by this dataset:

- **MICrONS** (mouse cortex): <https://www.microns-explorer.org/cortical-mm3>
- **FlyWire** (Drosophila): <https://flywire.ai/>
- **H01** (human cortex): <https://h01-release.storage.googleapis.com/landing.html>
- **Zebrafish** (fish1): <https://fish1-release.storage.googleapis.com/index.html>
