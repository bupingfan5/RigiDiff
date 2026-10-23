# 🧬 RigiDiff: Geometry–Topology Coherent 3D Molecular Generation via Rigid Fragment Diffusion Conditioned on Protein Pockets

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Implementation-orange.svg)](https://pytorch.org/)

Official PyTorch implementation of **RigiDiff**, a protein-pocket-conditioned framework for de novo 3D ligand generation.The accompanying manuscript, “Geometry-Topology Coherent 3D Molecular Generation via Rigid Fragment Diffusion Conditioned on Protein Pockets,” has been submitted to Bioinformatics.

## 📖 Abstract

Motivation: Generating 3D ligands within protein pockets requires jointly considering pocket compatibility, molecular geometry, and chemical connectivity. Most atom-level or sequential methods lack a unified representation of chemically meaningful substructures, the spatial poses of these substructures, and the interconnections among them, which limits the 3D molecular generation capabilities of such methods.

Results: We propose RigiDiff, a pocket-conditioned rigid-fragment diffusion framework that represents ligands as graphs of chemically meaningful rigid fragments, jointly encoding fragment poses and connectivity. A dual-track geometric-topological diffusion model jointly generates fragment poses, types and bond states through a shared pocket-conditioned equivariant denoiser that incorporates noisy bond states into geometric updates. A Geometry-Topology Consistency Loss (GTCL) constrains distances between bonded fragment centers to promote geometric-topological consistency, while an attachment-site-aware, template-constrained decoder supports chemically constrained molecular assembly. Experiments on CrossDocked2020 show that RigiDiff achieves a normalized SA score of 0.760, exceeding the strongest reported baseline by 11.1%. Compared with the best-performing baselines for each metric, RigiDiff attains the highest molecular diversity of 0.818, outperforming the strongest baseline by 3.0%, and achieves the lowest Vina score of −7.275 kcal/mol among all compared methods. These results highlight favorable predicted synthetic accessibility, structural diversity, and docking performance.

## ⚙️ Installation

### 1. Environment Requirements

The project targets Linux with CUDA. Linux  with an NVIDIA GPU is recommended.

| Component | Requirement |
| :--- | :--- |
| **Operating System** | Ubuntu22.04 |
| **Python** | 3.10 |
| **GPU** | CUDA-capable NVIDIA GPU |
| **Environment** | Conda; dependencies in `environment.yml` |
| **Docking Tools** | AutoDock Vina, Meeko, and Open Babel |

### 2. Create a Conda Environment
```bash
git clone https://github.com/bupingfan5/RigiDiff.git
cd RigiDiff

conda env create -f environment.yml -n RigiDiff
conda activate RigiDiff
```

Run the commands below from the repository root in a Bash shell.

If docking tools are unavailable after environment creation, install them with:

```bash
conda install -c conda-forge vina meeko openbabel
```

## 📂 Dataset

RigiDiff uses CrossDocked2020 with a fixed split of **90,845 training complexes, 1,000 validation complexes, and 99 test pockets**. The training/validation partition uses seed **42**, and the fixed fragment vocabulary contains **6,924 types**.

### 1. Download the Data

Download the data assets from [Google Drive](https://drive.google.com/drive/folders/1s6LN3yt7IkNCYrGmX-0Im9eNhew6CeRl?usp=sharing).

### 2. Organize the Files

Place the following assets in `data/`:

```text
data/
├── crossdocked_pocket10/                         # Raw pocket/ligand files and index.pkl
├── crossdocked_pocket10_test99_v2.lmdb            # Shared training/validation/test LMDB
├── crossdocked_pocket10_test99_val1000_fixed_split.pt  # Fixed train/validation/test split
├── crossdocked_pocket10_fragment_vocab.pt        # 6,924-type vocabulary
└── crossdocked_pocket10_bond_class_weights.json  # Bond-class loss weights
```

### 3. Configure the Data Paths

If you store the data elsewhere, update these entries in `config/config_crossdocked_stable.yml` and the corresponding paths in the sampling and evaluation commands:

```yaml
data: data
train_data_prefix: crossdocked_pocket10_test99_v2
val_data_prefix: crossdocked_pocket10_test99_v2
split_file: data/crossdocked_pocket10_test99_val1000_fixed_split.pt
fragment_vocab_path: data/crossdocked_pocket10_fragment_vocab.pt
bond_class_weights_file: data/crossdocked_pocket10_bond_class_weights.json
```

## 🚀 Usage

### 1. Model Training

Train from scratch using `config/config_crossdocked_stable.yml`.

```bash
python -u train_fragdiff.py \
  --config config/config_crossdocked_stable.yml
```

| Setting | Value |
| --- | --- |
| Denoiser | 6 layers; hidden dimension 128 |
| Training duration / precision | 200 epochs / FP32 |
| Optimizer | AdamW with AMSGrad |
| Minibatch / gradient accumulation | 4 / 4 minibatches per optimizer update |
| Weight decay | `1e-5` |
| Learning-rate warmup | Linear `5e-7` → `1e-5` over the first 5% of optimizer updates |
| Remaining learning-rate schedule | Cosine decay to `5e-7` |
| Gradient clipping | Maximum norm 0.5 |
| Exponential moving average (EMA) | Decay 0.9999 |
| Training seed | 42 |
| Diffusion steps / noise schedule | 500 / cosine |

A new run writes checkpoints and logs to:

```text
checkpoints/run12_crossdocked_stable_retrain_val1000/
├── last.ckpt
├── ..._epoch=09.ckpt                           # Periodic checkpoints, every 10 epochs
└── best_val/                                   # Up to three best validation checkpoints

logs/general_logs/run12_crossdocked_stable_retrain_val1000/
├── log.log
└── epoch_metrics.csv                           # Epoch-wise losses and learning rates
```

YAML values override duplicate command-line options. For an independent repeat, copy the configuration and change its `run_name` before launching, so existing outputs are preserved.

#### Selected Checkpoint

The experiment described in the original README selected **epoch index 192** by the lowest logged validation loss:

```text
checkpoints/run11/run12_crossdocked_stable_retrain_val1000_best_val_epoch=192.ckpt
```

The corresponding archived loss log is:

```text
checkpoints/run11/logs/general_logs/run12_crossdocked_stable_retrain_val1000/epoch_metrics.csv
```

### 2. Molecule Sampling

The command below follows the sampling protocol used in the experiment: 99 held-out pockets, 100 generation attempts per pocket, seed 3407, empirical fragment-count sampling, and constrained template decoding. Property-value guidance is disabled, while clash guidance uses a scale of 0.01. The selected model is the epoch-192 checkpoint; adjust its path and the input/output paths to match your directory structure.

```bash
python generate_fragments.py \
  --model checkpoints/best_val/run12_crossdocked_stable_retrain_val1000_best_val_epoch=192.ckpt \
  --output samples/run12_epoch192_formal_test99_100perpocket_seed3407 \
  --use_dataset \
  --data_dir data \
  --dataset_prefix crossdocked_pocket10_test99_v2 \
  --split_file data/crossdocked_pocket10_test99_val1000_fixed_split.pt \
  --fragment_vocab_path data/crossdocked_pocket10_fragment_vocab.pt \
  --strict_targets \
  --dataset_split test \
  --limit 99 \
  --n_samples 100 \
  --batch_size 8 \
  --oversample_factor 1 \
  --fragment_count_mode empirical \
  --fragment_count_max 7 \
  --fragment_count_seed 3407 \
  --decoder_attempts 20 \
  --decoder_tree_beam_width 128 \
  --decoder_tree_plans 8 \
  --temperature 0.5 \
  --decoder_type_temperature 0.25 \
  --guidance_scale 0 \
  --clash_guidance_scale 0.01 \
  --clash_guidance_start 0 \
  --clash_guidance_end 100 \
  --clash_guidance_max_norm 0.2 \
  --clash_guidance_global_fraction 0.25 \
  --clash_guidance_max_com_shift 1.0 \
  --diversity_threshold 0.85 \
  --seed 3407
```

Outputs include:

- `generation_attempts.csv`: every attempt, including assembly failures.
- `metrics.csv`: generated-molecule metadata and property measurements.
- `candidates/`: successfully assembled SDF files; use these for the main evaluation.
- `selected/`: a filtered subset of generated molecules.

### 3. Docking 

Evaluate the generated `candidates/` using  Vina docking:

Before running the command below, extract crossdocked_pocket10.tar.gz into the data/ directory and ensure that the extracted folder is named crossdocked_pocket10 ( data/crossdocked_pocket10).

Before running the command below, also extract the sample archive into the samples/ directory following the directory structure shown above.

```
RigiDiff/
└── samples/
    └── run12_epoch192_formal_test99_100perpocket_seed3407/
        ├── candidates/    # Successfully assembled molecules for docking
        └── selected/      # Filtered subset of generated molecules
```

```bash
python evaluate/vina_full_protein.py \
  --metrics_csv samples/run12_epoch192_formal_test99_100perpocket_seed3407/metrics.csv \
  --generation_attempts_csv samples/run12_epoch192_formal_test99_100perpocket_seed3407/generation_attempts.csv \
  --sdf_dir samples/run12_epoch192_formal_test99_100perpocket_seed3407/candidates \
  --data_root data/crossdocked_pocket10 \
  --lmdb data/crossdocked_pocket10_test99_v2.lmdb \
  --split_file data/crossdocked_pocket10_test99_split.pt \
  --split test \
  --chain_mode all \
  --keep_heterogens "" \
  --ph 7.4 \
  --protein_box_padding 6 \
  --default_altloc A \
  --receptor_prep_backend openbabel \
  --exhaustiveness 64 \
  --num_modes 9 \
  --seed 3407 \
  --cpu 8 \
  --max_targets 99 \
  --max_per_target 100 \
  --samples_per_target 100 \
  --source_samples_per_target 100 \
  --append_generation_failures \
  --resume \
  --work_dir work_vina_openbabel_test99_seed3407 \
  --receptor_manifest_csv openbabel_test99_seed3407_manifest.csv \
  --output_csv vina_openbabel_test99_seed3407.csv \
  --summary_csv vina_openbabel_test99_seed3407_summary.csv \
  --per_target_csv vina_openbabel_test99_seed3407_per_target.csv \
  --topk_csv vina_full_openbabel_test99_seed3407_topk.csv
```

Docking results can vary modestly with software versions, receptor preparation, hardware, and Vina parallelism. 

## 🗂️ Folder Structure 

```text
RigiDiff/
├── config/          # training configurations
├── data/            # Split files, fixed vocabulary, and topology-loss weights
├── data_process/    # CrossDocked-to-LMDB conversion and data audits
├── src/             # Diffusion model, equivariant network, datasets, and decoder 
├── checkpoints/     # Saved model checkpoints
utilities
├── samples/         # Generated molecules, sampling metadata, and metrics
├── train_fragdiff.py
├── generate_fragments.py
├── evaluate/        # Vina, PoseBusters, property, and target-wise analyses
├── case_study/      # DPP4 selection, validation, and visualization pipeline
├── guidance/        # Property value-network data preparation and training
├── tests/           # Decoder and numerical smoke tests
└── environment.yml
```

## 🔁 Reproducibility

- Use the included split files and fragment vocabulary unchanged for paper reproduction. Use the same fixed split for training, sampling, and docking evaluation.
- The main results use generation seed 3407, 500 diffusion steps, a 6-layer EGNN with hidden dimension 128, `--guidance_scale 0`
- Property value guidance is reproducible through `guidance/` and should be evaluated separately from the unguided main protocol.
- LMDB databases, raw CrossDocked files, checkpoints, Vina work directories, and generated molecule sets are intentionally excluded from Git.

## 📝 Citation

Citation information will be added when the accompanying manuscript is publicly available.
