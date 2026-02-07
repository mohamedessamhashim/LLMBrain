# LLMBrain

LLM-conditioned brain tumor segmentation via cross-attention between Swin UNETR and LLaMA 3B, evaluated on the UCSF-PDGM dataset (501 preoperative diffuse glioma cases).

**Authors:** Mohamed Essam, Francesca Mussa

---

## Architecture

```
                 Clinical Prompt
                       |
                  +---------+
                  | LLaMA 3B|  (frozen, 4-bit quantized)
                  +---------+
                       |
              Sequence Embeddings  (B, seq_len, 3072)
                       |
        +--------------+--------------+--------------+
        |              |              |              |
   Cross-Attn     Cross-Attn    Cross-Attn     Cross-Attn
   (bottleneck)   (enc3 skip)   (enc2 skip)    (enc1 skip)
        |              |              |              |
  +-----+-----+  +----+----+  +-----+-----+  +-----+-----+
  | Swin UNETR Encoder  -->  Decoder (with conditioned skips) |
  +-----------+-------------------------------------------+
              |
       Segmentation Map  (B, 4, D, H, W)
```

1. **Vision encoder**: Swin UNETR (pretrained on BraTS) extracts multi-scale 3D features from 4-channel MRI (T1, T1ce, T2, FLAIR).
2. **Text encoder**: Frozen LLaMA 3B (4-bit quantized) encodes clinical prompts (age, diagnosis, IDH, MGMT, treatment) into token-level embeddings.
3. **Cross-attention**: At the bottleneck and three encoder skip connections, multi-head cross-attention injects text semantics into vision features. Learnable gates (initialized near zero) ensure stable training from pretrained weights.
4. **Decoder**: Standard Swin UNETR decoder produces 4-class segmentation (BG, NCR, ED, ET).

---

## Quick Start

### 1. Setup Environment
```bash
cd /path/to/LLMBrain
pip install -e .
```

**LLaMA access:** You need a HuggingFace token with access to `meta-llama/Llama-3.2-3B`.
```bash
huggingface-cli login
```

### 2. Download Data
Download UCSF-PDGM from TCIA and place in `data/raw/`:
- **Access Request**: https://www.cancerimagingarchive.net/collection/ucsf-pdgm/

The dataset loader auto-detects both raw and preprocessed layouts, so you can train directly on the downloaded data without preprocessing.

### 3. Generate Clinical Prompts
```bash
python scripts/generate_prompts.py --csv data/UCSF-PDGM-metadata_v5.csv --output data/prompts.csv
```

### 4. (Optional) Preprocess Data
Preprocessing is **optional** — the pipeline works directly on raw NIfTI files. If you want cleaner inputs:
```bash
python scripts/preprocess.py --input_dir data/raw/UCSF-PDGM --output_dir data/processed
```
Then update `data_dir` in your config to `"./data/processed"`.

### 5. Train

Both modes use a **pretrained Swin UNETR** by default (full BraTS-pretrained encoder + decoder from MONAI). This means you are fine-tuning, not training from scratch.

**Vision-only baseline:**
```bash
python scripts/train.py --config configs/baseline.yaml
```

**LLM-conditioned model:**
```bash
python scripts/train.py --config configs/llm_conditioned.yaml
```

### 6. Evaluate
```bash
# Baseline
python scripts/evaluate.py \
    --checkpoint outputs/baseline/best_model.pth \
    --config configs/baseline.yaml \
    --output_dir outputs/results_baseline

# LLM-conditioned
python scripts/evaluate.py \
    --checkpoint outputs/llm_conditioned/best_model.pth \
    --config configs/llm_conditioned.yaml \
    --output_dir outputs/results_llm
```

---

## Project Structure

```
LLMBrain/
├── configs/
│   ├── baseline.yaml              # Vision-only Swin UNETR
│   └── llm_conditioned.yaml       # Swin UNETR + LLaMA cross-attention
├── scripts/
│   ├── preprocess.py              # MRI preprocessing pipeline
│   ├── generate_prompts.py        # Clinical prompt generator from CSV
│   ├── train.py                   # Training entry point (both modes)
│   └── evaluate.py                # Evaluation + figure generation
├── src/llmbrain/
│   ├── data/
│   │   ├── dataset.py             # UCSF-PDGM dataset loader
│   │   └── transforms.py          # MONAI transforms
│   ├── models/
│   │   ├── swin_unetr.py          # SwinUNETRBaseline + LLMConditionedSwinUNETR
│   │   ├── llm_encoder.py         # LLaMA 3B text encoder (frozen, quantized)
│   │   └── cross_attention.py     # Cross-attention modules
│   ├── training/
│   │   ├── trainer.py             # Training loop (vision-only & LLM modes)
│   │   └── losses.py              # Dice + CE loss
│   ├── evaluation/
│   │   ├── metrics.py             # Dice, HD95 for WT/TC/ET
│   │   └── visualize.py           # Publication figures
│   └── utils/
│       └── config.py              # YAML config loading
├── data/
│   ├── UCSF-PDGM-metadata_v5.csv  # Clinical metadata (501 cases)
│   ├── raw/                        # Downloaded UCSF-PDGM data
│   ├── processed/                  # After preprocessing
│   └── prompts.csv                 # Generated clinical prompts
├── outputs/                        # Training outputs (gitignored)
└── archive/                        # Archived old code
```

---

## Pretrained Weights

The model supports three pretrained weight modes, configured via the `pretrained` field in the YAML config:

| Mode | Config Value | What It Loads | Use Case |
|------|-------------|---------------|----------|
| **Full model** (default) | `true` or `"full"` | Complete BraTS-pretrained Swin UNETR (encoder + decoder) | Recommended — fine-tune on UCSF-PDGM |
| **Encoder only** | `"encoder"` | Self-supervised pretrained Swin ViT encoder | When task differs significantly from BraTS |
| **From scratch** | `false` | Random initialization | Full training from scratch |
| **Local checkpoint** | `"/path/to/weights.pt"` | Custom weights from a .pt file | Resume from your own checkpoint |

Weights are automatically downloaded from the [MONAI Model Zoo](https://github.com/Project-MONAI/MONAI-extra-test-data/releases/tag/0.8.1) on first run and cached locally.

**Example — fine-tune with full pretrained weights (default):**
```yaml
# configs/baseline.yaml or configs/llm_conditioned.yaml
model:
  pretrained: true  # downloads full BraTS-pretrained encoder + decoder
```

**Example — encoder-only pretrained weights:**
```yaml
model:
  pretrained: "encoder"  # only the Swin ViT backbone is pretrained
```

**Example — load from your own checkpoint:**
```yaml
model:
  pretrained: "/path/to/my_checkpoint.pt"
```

---

## Model Details

### Vision Backbone: Swin UNETR
- Pretrained on BraTS via MONAI model zoo (full encoder + decoder)
- Input: 4-channel MRI (T1, T1ce, T2, FLAIR)
- Output: 4 classes (Background, NCR, ED, ET)
- Patch size: 96 x 96 x 96, Feature size: 48

### Text Encoder: LLaMA 3B
- `meta-llama/Llama-3.2-3B` loaded with NF4 quantization
- Frozen weights (only cross-attention layers are trained)
- Hidden dimension: 3072
- Produces token-level embeddings for cross-attention

### Cross-Attention
- Applied at 4 decoder stages (bottleneck + 3 skip connections)
- Multi-head attention (Q: vision features, K/V: text embeddings)
- Gated residual: `out = vision + sigmoid(gate) * cross_attn(vision, text)`
- Gates initialized at 0, so training starts from the pretrained vision-only behaviour

---

## Clinical Prompts

Generated from metadata CSV with template:
```
"{age}-year-old {sex}, {diagnosis}, {idh}, {mgmt}, {treatment}"
```

**Examples:**
| ID | Prompt |
|----|--------|
| UCSF-PDGM-004 | "66-year-old male, glioblastoma, IDH-wildtype, MGMT-unmethylated, status post subtotal resection" |
| UCSF-PDGM-021 | "41-year-old male, astrocytoma, IDH-mutant, MGMT-methylated, status post gross total resection" |
| UCSF-PDGM-005 | "80-year-old female, glioblastoma, IDH-wildtype, biopsy only" |

---

## Data Preparation

The dataset loader **auto-detects** whether you are using raw or preprocessed data. Simply point `data_dir` in your config to the appropriate directory.

### Option A: Raw Data (recommended for quick start)

Download UCSF-PDGM and place it directly in `data/raw/`. The loader discovers modalities by filename patterns (e.g. `*_t1.nii.gz`, `*_T1.nii.gz`, `*_t1gd.nii.gz` for T1ce, etc.).

```
data/raw/
├── UCSF-PDGM-0001/
│   ├── *_t1.nii.gz
│   ├── *_t1ce.nii.gz (or *_t1gd.nii.gz)
│   ├── *_t2.nii.gz
│   ├── *_flair.nii.gz (or *_FLAIR.nii.gz)
│   └── *_seg.nii.gz (or *_tumor_segmentation.nii.gz)
├── UCSF-PDGM-0002/
│   └── ...
```

```yaml
# In config YAML
data:
  data_dir: "./data/raw"
```

### Option B: Preprocessed Data

For cleaner inputs, run the optional preprocessing pipeline first:
```bash
python scripts/preprocess.py --input_dir data/raw/UCSF-PDGM --output_dir data/processed
```

This produces a standardized layout:
```
data/processed/
├── images/
│   ├── UCSF-PDGM-0001.nii.gz
│   └── ...
└── labels/
    ├── UCSF-PDGM-0001.nii.gz
    └── ...
```

```yaml
data:
  data_dir: "./data/processed"
```

### Preprocessing Pipeline (optional)
1. **N4 Bias Field Correction** (SimpleITK)
2. **Registration to MNI template** (rigid, 1mm isotropic)
3. **Brain Extraction** (HD-BET)
4. **Intensity Normalization** (z-score, per-channel)

---

## Expected Results

### Vision-Only Baseline
| Metric | Expected Range |
|--------|----------------|
| Dice (WT) | 0.88 - 0.91 |
| Dice (TC) | 0.82 - 0.85 |
| Dice (ET) | 0.75 - 0.78 |
| HD95 (WT) | 9 - 12 mm |

### LLM-Conditioned (hypothesis)
Clinical conditioning should improve segmentation in ambiguous cases,
particularly for tumor core and enhancing tumor delineation.

**Tumor Regions:**
- **WT** (Whole Tumor) = NCR + ED + ET
- **TC** (Tumor Core) = NCR + ET
- **ET** (Enhancing Tumor)

---

## Troubleshooting

### CUDA Out of Memory
```yaml
# In config YAML
training:
  batch_size: 1
model:
  use_checkpoint: true
llm:
  load_in_4bit: true    # Already default
```

### LLaMA Access
```bash
# Requires HuggingFace account with Meta LLaMA access
huggingface-cli login
# Request access at: https://huggingface.co/meta-llama/Llama-3.2-3B
```

### HD-BET Installation
```bash
pip install git+https://github.com/MIC-DKFZ/HD-BET.git
```

### Missing MNI Template
Download from: https://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009

---

## Verification Checklist

### After Setup
- [ ] `pip install -e .` runs without errors
- [ ] `python -c "import llmbrain"` works

### After Data Download
- [ ] `data/raw/` contains subject directories (e.g. `UCSF-PDGM-0001/`)
- [ ] Each subject has T1, T1ce, T2, FLAIR, and segmentation NIfTI files

### After Prompt Generation
- [ ] `data/prompts.csv` exists with 501 rows
- [ ] Prompts match expected format

### After Training (both modes)
- [ ] `outputs/*/best_model.pth` saved
- [ ] TensorBoard logs in `outputs/*/logs/`
- [ ] Loss decreasing over epochs
- [ ] Pretrained weights downloaded on first run (cached for subsequent runs)

### After Evaluation
- [ ] `outputs/results/metrics.csv` with Dice/HD95
- [ ] Figures: triplanar views, comparison grids, Dice boxplots

---

## License

MIT License
