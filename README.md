# RigiDiff: Geometry--Topology Coherent 3D Ligand Generation with Rigid Fragment Diffusion

Official PyTorch implementation of **RigiDiff**, a pocket-conditioned framework for de novo three-dimensional ligand generation. RigiDiff represents a ligand as a graph of chemically meaningful rigid fragments and jointly models fragment poses, fragment identities, inter-fragment bond states, and attachment sites under protein-pocket conditioning.

The repository is organized as a reproducible research release: it provides the data-processing code, the staged training configurations, the sampling and constrained assembly pipeline, and the whole-protein Vina evaluation workflow used for the reported experiments.

## Installation

### Requirements

The release was developed for Linux with CUDA. We recommend Linux or WSL2, Python 3.10, and an NVIDIA GPU. Whole-protein docking additionally requires AutoDock Vina, Meeko, and Open Babel.

```bash
git clone https://github.com/bupingfan5/RigiDiff.git
cd RigiDiff

conda env create -f environment.yml
conda activate RigiDiff
```

If docking tools are unavailable after environment creation, install them with:

```bash
conda install -c conda-forge vina meeko openbabel
```

Run the decoder unit test before a full experiment:

```bash
python -m unittest tests.test_rigid_decoder
```

## Data

The `data/` directory contains the fixed fragment vocabulary, dataset splits, and bond-class weights used throughout training and evaluation:

```text
data/
├── crossdocked_pocket10_fragment_vocab.pt       # fixed fragment vocabulary
├── crossdocked_pocket10_split.pt                # training/validation/test split
├── crossdocked_pocket10_test99_split.pt         # held-out 99-pocket evaluation split
└── crossdocked_pocket10_bond_class_weights.json # topology-loss weights
```

Place the raw CrossDocked pocket10 directory, including `index.pkl`, at `data/crossdocked_pocket10/`. Then construct the LMDB used by training and the evaluation LMDB used for the held-out test set:

```bash
# Training LMDB
python data_process/prepare_crossdock_lmdb.py \
  --raw_dir data/crossdocked_pocket10 \
  --output data/crossdocked_pocket10.lmdb \
  --required_split_file data/crossdocked_pocket10_split.pt \
  --required_split test

# Evaluation LMDB containing the held-out targets required by the paper protocol
python data_process/prepare_crossdock_lmdb.py \
  --raw_dir data/crossdocked_pocket10 \
  --output data/crossdocked_pocket10_test99_v2.lmdb \
  --required_split_file data/crossdocked_pocket10_test99_split.pt \
  --required_split test
```

The conversion script audits every key in the requested split and stops if a target is missing. **Do not regenerate or replace `crossdocked_pocket10_fragment_vocab.pt` when using the released checkpoint**, because sampling verifies that the dataset vocabulary and checkpoint vocabulary agree.

## Pretrained Checkpoint

The paper sampling protocol uses the final frame-consistent checkpoint:

```text
checkpoints/run8_framefix_hfix/run8_framefix_hfix_epoch=03.ckpt
```

## Reproducing the Main Sampling Experiment

The command below follows the reported unguided protocol: 99 held-out pockets, 100 attempts per pocket, seed 2025, empirical fragment-count sampling, and constrained template decoding.You can set the model name yourself.

```bash
python generate_fragments.py \
  --model checkpoints/run8_framefix_hfix/run8_framefix_hfix_epoch=03.ckpt \
  --output samples/paper_test99_seed2025 \
  --use_dataset \
  --data_dir data \
  --dataset_prefix crossdocked_pocket10_test99_v2 \
  --split_file data/crossdocked_pocket10_test99_split.pt \
  --fragment_vocab_path data/crossdocked_pocket10_fragment_vocab.pt \
  --strict_targets \
  --dataset_split test \
  --limit 99 \
  --n_samples 100 \
  --batch_size 8 \
  --oversample_factor 1 \
  --fragment_count_mode empirical \
  --fragment_count_max 7 \
  --fragment_count_seed 2025 \
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
  --seed 2025
```

The output directory contains `metrics.csv`, `generation_attempts.csv`, and molecule files in `candidates/`. `generation_attempts.csv` retains both successful and failed attempts and should be used when reporting generation success rates.

## Property Value Network and Guided Sampling

RigiDiff also supports a **fragment-level property value network**. The value network is trained on a composite target formed from QED, normalized synthetic accessibility, LogP, and molecular weight; it is therefore a property-quality guide rather than a Vina surrogate. During reverse diffusion, its coordinate gradient is injected through `ValueNetworkGuidance` to bias sampling toward higher predicted property quality.

Prepare value-network training data from the training split:

```bash
python guidance/prepare_value_data.py \
  --data_dir data \
  --dataset_prefix crossdocked_pocket10 \
  --split train \
  --output data/value_network_dataset_crossdocked.pt
```

Train the value network. The sampling script automatically loads the value checkpoint from `checkpoints/value_net_crossdock_ep60.pt`, so retain this output path for the default guided-sampling workflow:

```bash
python guidance/train_value_net.py \
  --data_path data/value_network_dataset_crossdocked.pt \
  --epochs 60 \
  --batch_size 64 \
  --lr 2e-4 \
```

Identify the model weights that yield the lowest loss from among all the generated weights, and then run the generation code to guide the process.

To enable value-guided sampling, add the checkpoint and guidance arguments to the sampling command:

```bash
python generate_fragments.py \
  --model checkpoints/run8_framefix_hfix/run8_framefix_hfix_epoch=03.ckpt \
  --output samples/value_guided_test99_seed2025 \
  --use_dataset \
  --data_dir data \
  --dataset_prefix crossdocked_pocket10_test99_v2 \
  --split_file data/crossdocked_pocket10_test99_split.pt \
  --fragment_vocab_path data/crossdocked_pocket10_fragment_vocab.pt \
  --strict_targets --dataset_split test --limit 99 \
  --n_samples 100 --batch_size 8 \
  --fragment_count_mode empirical --fragment_count_max 7 \
  --fragment_count_seed 2025 \
  --decoder_attempts 20 --decoder_tree_beam_width 128 --decoder_tree_plans 8 \
  --temperature 0.5 --decoder_type_temperature 0.25 \
  --guidance_scale 0.4 --guidance_start 50 --guidance_end 500 \
  --clash_guidance_scale 0.01 --clash_guidance_start 0 --clash_guidance_end 100 \
  --clash_guidance_max_norm 0.2 --clash_guidance_global_fraction 0.25 \
  --clash_guidance_max_com_shift 1.0 \
  --diversity_threshold 0.85 --seed 2025
```

For a controlled comparison, hold the split, seed, decoder settings, and clash-guidance settings fixed, and vary only `--guidance_scale` (with the value checkpoint fixed at the path above). The main reported comparison uses `--guidance_scale 0`; value-guided results should be reported as a separate guided-sampling experiment.

## Whole-Protein Blind Docking and Property Evaluation

The main Vina result is obtained by docking each generated molecule against its complete target protein. The command below uses all chains, Open Babel receptor preparation, Vina exhaustiveness 64, and the same seed as sampling.To obtain the complete set of evaluation metrics, run `metrics.py` in the `evaluate` directory.

```bash
python evaluate/vina_full_protein.py \
  --metrics_csv samples/paper_test99_seed2025/metrics.csv \
  --generation_attempts_csv samples/paper_test99_seed2025/generation_attempts.csv \
  --sdf_dir samples/paper_test99_seed2025/candidates \
  --data_root data/crossdocked_pocket10 \
  --lmdb data/crossdocked_pocket10_test99_v2.lmdb \
  --split_file data/crossdocked_pocket10_test99_split.pt \
  --split test \
  --chain_mode all \
  --receptor_prep_backend openbabel \
  --exhaustiveness 64 \
  --num_modes 9 \
  --seed 2025 \
  --cpu 8 \
  --max_targets 99 \
  --max_per_target 100 \
  --samples_per_target 100 \
  --source_samples_per_target 100 \
  --work_dir work/vina_test99_seed2025 \
  --output_csv results/vina_test99_seed2025.csv \
  --summary_csv results/vina_test99_seed2025_summary.csv \
  --per_target_csv results/vina_test99_seed2025_per_target.csv \
  --topk_csv results/vina_test99_seed2025_topk.csv
```

Optional post-processing commands are:

```bash
# PoseBusters structural checks
python evaluate/run_posebusters.py \
  --vina_csv results/vina_test99_seed2025.csv \
  --data_root data/crossdocked_pocket10 \
  --output_csv results/posebusters_test99_seed2025.csv

# Molecular descriptors and aggregate statistics
python evaluate/analyze_ligand_properties.py \
  --vina_csv results/vina_test99_seed2025.csv \
  --posebusters_csv results/posebusters_test99_seed2025.csv \
  --output_csv results/properties_test99_seed2025.csv \
  --summary_csv results/properties_test99_seed2025_summary.csv \
  --distribution_csv results/properties_test99_seed2025_distribution.csv
```

Docking results can vary modestly with software versions, receptor preparation, hardware, and Vina parallelism. 

## Training from Scratch

The release includes the four-stage schedule used to obtain the final model. `--init_checkpoint` initializes model weights only; it does **not** restore optimizer state. In contrast, `--resume` continues a run from its last compatible training-state checkpoint.This project employs a fine-tuning approach; therefore, multiple training runs are required.

```bash
# Stage 1: base rigid-fragment diffusion training
python train_fragdiff.py --config config/train_base.yml --run_name run1

# Stage 2: corrective fine-tuning
python train_fragdiff.py \
  --config config/finetune_corrective.yml \
  --run_name run2 \
  --init_checkpoint checkpoints/run1/last.ckpt

# Stage 3: stable corrective fine-tuning
python train_fragdiff.py \
  --config config/finetune_stable.yml \
  --init_checkpoint checkpoints/run2/last.ckpt

# Stage 4: frame-consistent final fine-tuning
python train_fragdiff.py --config config/finetune_frame_final.yml
```

`config/finetune_frame_final.yml` names the selected Stage-3 checkpoint explicitly. For a newly trained model, update its `init_checkpoint` field to the Stage-3 checkpoint selected by validation loss before launching Stage 4. Configuration values are loaded after command-line arguments and therefore take precedence over duplicate command-line options.

## Case Study

The `case_study/` directory contains the workflow used for the DPP4 analysis. After the main sampling and docking outputs are available, the scripts select a target and representative generated molecules, validate steric contacts, prepare poses, and export a PyMOL session:

```bash
python case_study/select_case_target.py
python case_study/select_case_molecules.py --target-id 48
python case_study/validate_case_study.py --target-id 48
python case_study/prepare_case_poses.py --target-id 48
python case_study/analyze_interactions.py --target-id 48
python case_study/generate_pymol_case_study.py --target-id 48
```

The default case-study scripts assume the paths generated by the paper evaluation workflow. Adjust the paths in `case_study/case_utils.py` if you use different output directories.

## Repository Structure

```text
RigiDiff/
├── config/          # Four-stage training configurations
├── data/            # Split files, fixed vocabulary, and topology-loss weights
├── data_process/    # CrossDocked-to-LMDB conversion and data audits
├── src/             # Diffusion model, equivariant network, datasets, and decoder utilities
├── train_fragdiff.py
├── generate_fragments.py
├── evaluate/        # Vina, PoseBusters, property, and target-wise analyses
├── case_study/      # DPP4 selection, validation, and visualization pipeline
├── guidance/        # Property value-network data preparation and training
├── tests/           # Decoder and numerical smoke tests
└── environment.yml
```

## Reproducibility Notes

- Use the included split files and fragment vocabulary unchanged for paper reproduction.
- The main results use seed 2025, 500 diffusion steps, a 6-layer EGNN with hidden dimension 128, and `--guidance_scale 0`.
- Property value guidance is reproducible through `guidance/` and should be evaluated separately from the unguided main protocol.
- LMDB databases, raw CrossDocked files, checkpoints, Vina work directories, and generated molecule sets are intentionally excluded from Git.

## Citation

Citation information will be added when the accompanying manuscript is publicly available.
