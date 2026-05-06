# Multi-Task Model for NER + Event Classification

A unified (multi-task) model solving **NER (BIO, token-level)** and **CLS (document-level multihot event/relation classification)** on the NEREL dataset.

---

## Description

The project implements a single model that simultaneously performs two tasks:
- **NER (Named Entity Recognition)** — extracting named entities at the token level in BIO format
- **Event Classification** — multi-label classification of events/relations at the document level (multihot)

The model uses uncertainty-weighting to balance the loss between tasks and supports INT8 quantization with minimal quality loss.

### Key Features:
- Multi-task architecture based on a shared encoder
- Uncertainty-weighting for automatic loss balancing
- INT8 quantization support (1.5x size reduction, ~20% inference speedup)
- Token-level F1 micro > 85%, cls-level F1 micro > 80%
- Detailed error analysis and FP32 vs INT8 comparison

---

## Project Structure

```
nerel_multitask/
├── NEREL_MultiTask_Notebook.ipynb   # Main notebook
├── requirements.txt                 # Dependencies
└── .venv/                           # Virtual environment
```

### Notebook Structure

| # | Section | Description |
|---|---------|-------------|
| 1 | Environment Setup | Paths, seed, imports |
| 2 | EDA | Loading jsonl, data overview, plots, insights |
| 3 | Parsing & Targets | NEREL format parsers, example collection (`build_examples_from_nerel`) |
| 4 | Tokenization | Label alignment (`tokenize_and_align_labels`), Dataset, Collator, DataLoader |
| 5 | Model & Loss | `JointModel` class, custom loss with uncertainty-weighting |
| 6 | Training | Training loop, optimizer, scheduler, metrics logging |
| 7 | Inference | Inference pipeline, error analysis |
| 8 | Quantization | INT8 quantization, quality / speed / size comparison |

---

## Installation

### Requirements
- Python 3.8+
- PyTorch
- transformers (HuggingFace)

### Installing Dependencies

```bash
# Windows
.venv\Scripts\activate.bat

# Linux/Mac
source .venv/bin/activate

# Install packages
pip install -r requirements.txt
```

---

## Usage

Open the notebook and run cells sequentially:

```
NEREL_MultiTask_Notebook.ipynb
```

Each section of the notebook contains comments and descriptions of the steps performed.

---

## Model Architecture

```
Input Tokens → BERT Encoder → [CLS] token → CLS Head → Event Labels (multihot)
                            └→ Token Embeddings → NER Head → BIO Tags
```

**Loss:**
```
L_total = w_ner * L_ner + w_cls * L_cls
```
Weights `w_ner` and `w_cls` are computed automatically via uncertainty-weighting (Kendall et al.).

---

## Results

### Quality Metrics

| Metric | FP32 (token) | INT8 (token) | FP32 (cls) | INT8 (cls) |
|--------|--------------|--------------|------------|------------|
| F1 micro | 0.8503 | 0.8294 | 0.8190 | 0.8197 |
| F1 macro | 0.3110 | 0.2646 | 0.7584 | 0.7598 |
| Precision | 0.8503 | 0.8294 | 0.8190 | 0.8197 |
| Recall | 0.8503 | 0.8294 | 0.8190 | 0.8197 |
| Accuracy | 0.8503 | 0.8294 | 0.8190 | 0.8197 |

### Speed and Size

| Parameter | FP32 | INT8 |
|-----------|------|------|
| Inference time (ms) | 5.19 | 3.98 |
| Speedup | — | ~20% |
| Model size | 1x | ~0.67x |

### Target Benchmarks

| Task | Target | Result |
|------|--------|--------|
| NER (token-level F1 micro) | > 85% | 85.03% |
| CLS (cls-level F1 micro) | > 80% | 81.90% |

---

## Conclusions

The multi-task model efficiently handles NER and CLS simultaneously without the need for separate models for each task.

INT8 quantization reduces model size by 1.5x and speeds up inference by approximately 20% with minimal quality loss.

Error analysis reveals that the main difficulties are associated with rare classes and complex contexts — for example, the model incorrectly classifies routine actions as events. The model also struggles with nested entities. This can be improved by adding more training data or using a CRF layer on top of the NER head.

---

## Technologies

- **PyTorch** — training framework
- **Transformers** — pretrained encoders (HuggingFace)
- **ONNX Runtime** — quantization and inference optimization
- **scikit-learn** — metrics computation
- **matplotlib / seaborn** — visualization
