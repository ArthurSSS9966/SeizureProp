# Seizure Propagation Detection

## Project Overview

This project automates **clinical diagnosis of epilepsy** using deep learning and brain connectivity analysis. The goal is to predict seizure propagation patterns and identify patients at risk using neurophysiological data.

**Key Question:** Can we predict seizure propagation from interictal (between-seizure) neural connectivity patterns using machine learning?

**Status (Feb 2026):** Multiple models tested; S4/S4D architectures showing promise; results tracked in iterative testing notebooks.

---

## Project Goals

1. **Seizure Propagation Prediction** — Forecast which brain regions will be affected during a seizure
2. **Clinical Diagnosis Automation** — Reduce time for neurologists to analyze EEG/intracranial recordings
3. **Patient Stratification** — Identify high-risk vs low-risk seizure patterns
4. **Connectivity-Based Features** — Use graph-based metrics (connectivity) + raw signal features

---

## Experimental Setup

### Data
- **Source:** Intracranial EEG recordings (iEEG) from epilepsy patients
- **Frequency:** Sampled at ~2 kHz (typical for clinical systems)
- **Features:** Raw signal + computed connectivity (coherence, correlation, spectral measures)
- **Labels:** Seizure vs non-seizure (binary classification)

### Data Splits
- **Training:** `data/` folder
- **Testing:** `data_test/` folder
- Check `datasetConstruct.py` for exact train/val/test split ratios

---

## Code Structure

### Core Modules

#### **Data Handling**
- `datasetConstruct.py` — Load EEG data, construct train/val/test datasets
- `test_dataset.py` — Validate dataset construction (check for leakage, imbalance)
- Data folders: `data/`, `data_test/`

#### **Connectivity Analysis**
- `connectivity_cal.py` — Compute connectivity matrices between brain regions
  - Methods: Coherence, correlation, spectral power correlation
  - Output: Graph adjacency matrices for each time window

#### **Models**
- `models.py` — Full model implementations (ResNet, Transformer, S4, etc.)
- `models_simplified.py` — Simplified versions for testing/debugging
- `s4.py` — S4 (Structured State Space) architecture
- `s4d.py` — S4D variant (Diagonal S4 - faster, simpler)

#### **Training & Testing**
- `main.py` — **MAIN ENTRY POINT** — Train/evaluate models
- `steps.py` — Single training/eval loop (loss computation, backprop, metrics)
- `plotFun.py` — Visualization (ROC, confusion matrix, training curves)

#### **Utilities**
- `utils.py` — Helper functions (data loading, metric computation, logging)

### Notebook Archive

**Model Testing Timeline:**
```
0205ModelTesting.ipynb      → Early baseline
0302FeatureVisualization.ipynb → Feature engineering exploration
0326ModelTesting.ipynb      → First full pipeline
...
1124ModelTesting.ipynb      → Latest testing (Dec 2025)
```

Each notebook typically:
- Loads data
- Trains model
- Evaluates on test set
- Plots results (ROC, loss curves, etc.)

Check the dates to see which experiments are most recent!

### Results & Checkpoints

```
checkpoints/
  ├── model_best.pt          # Best model weights (highest validation accuracy)
  ├── model_final.pt         # Final model (end of training)
  └── [other saved states]

results/
  ├── results_summary.csv    # Performance metrics across experiments
  ├── gw_prefilter_vs_no_prefilter_summary_mean.xlsx
  └── [plots, confusion matrices]
```

---

## How to Use

### 1. Prepare Data

```bash
# Construct datasets from raw EEG
python datasetConstruct.py --data_dir data/ --output_dir processed/
```

### 2. Compute Connectivity Features

```python
from connectivity_cal import compute_connectivity

# Load raw EEG signal (n_channels, n_timepoints)
connectivity_matrix = compute_connectivity(
    eeg_data,
    method='coherence',  # or 'correlation', 'spectral'
    freq_bands={'theta': (4, 8), 'alpha': (8, 12), ...}
)
```

### 3. Train a Model

```bash
# Train S4D model (fastest)
python main.py \
    --model s4d \
    --data_dir data/ \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.001 \
    --save_path checkpoints/model_best.pt
```

### 4. Evaluate on Test Set

```bash
python main.py \
    --model s4d \
    --data_dir data_test/ \
    --load_path checkpoints/model_best.pt \
    --eval_only
```

### 5. Visualize Results

```python
from plotFun import plot_roc, plot_confusion_matrix

# After evaluation:
plot_roc(y_true, y_pred_proba)
plot_confusion_matrix(y_true, y_pred)
```

---

## Model Comparison

| Model | Notes | Status |
|-------|-------|--------|
| **S4D** | Diagonal State Space (fast, interpretable) | ✅ Recommended |
| **S4** | Full State Space (powerful, slower) | ✅ Working |
| **ResNet** | Baseline convolutional | ✅ Works |
| **Transformer** | Attention-based | ⚠️ Slower |

**Recommendation:** Start with S4D (in `s4d.py`) — good balance of speed and accuracy.

---

## Key Hyperparameters

### Architecture
- **State size (d_state):** 64-128 (larger = more expressive but slower)
- **Kernel size:** 5-11 (temporal receptive field)
- **Layers:** 4-6 (depth; more = slower training)

### Training
- **Learning rate:** 1e-4 to 1e-3 (start low, schedule down)
- **Batch size:** 16-64 (depends on GPU memory)
- **Epochs:** 100-200 (check validation plateau)
- **Optimizer:** Adam (default)

### Data
- **Input length:** Check `datasetConstruct.py` (likely 2-10 seconds of EEG)
- **Connectivity type:** Coherence recommended (most stable)
- **Frequency normalization:** Check if features are z-scored

---

## Important Details

### Connectivity Computation (`connectivity_cal.py`)

**Coherence** (recommended):
- Measures phase-locking between regions
- Frequency-specific (use theta/alpha/beta bands separately)
- Robust to volume conduction

**Correlation** (simple):
- Raw Pearson correlation
- Sensitive to zero-lag coupling
- Faster to compute

**Spectral Power Correlation:**
- Correlate power envelopes across frequency bands
- Good for slow modulation effects

### Model Input/Output

**Input shape:** (batch_size, n_channels, time_steps) or (batch_size, n_channels, n_channels, time_steps) for connectivity matrices

**Output:** Binary classification logits → sigmoid → probability of seizure

---

## Known Issues & Limitations

🚩 **High variance across folds** — Different patients/recordings very different
- Solution: Use patient-stratified K-fold cross-validation

🚩 **Class imbalance** — Seizure events are rare
- Solution: Use weighted loss function or SMOTE oversampling

🚩 **Temporal leakage risk** — If not careful, test set may have temporal overlap with training
- Check: `test_dataset.py` validates this

🚩 **Generalization to new patients unclear** — Models may overfit to specific patients
- Best practice: Leave-one-patient-out evaluation

---

## Next Steps (Priority Order)

1. **Validation on held-out patients** (URGENT)
   - Current results may be overfitted
   - Retrain with leave-one-patient-out cross-validation
   - Report generalization performance

2. **Ablation studies**
   - Which connectivity metric matters most?
   - Which frequency bands are critical?
   - Which brain regions drive predictions?

3. **Clinical validation**
   - Deploy on prospective data (new seizures)
   - Compare predictions to neurologist assessments
   - Measure clinical utility (time saved, accuracy improvement)

4. **Interpretability**
   - Which connectivity patterns predict seizure?
   - Can we identify biomarkers?
   - Use attention maps (Transformer) or gradient-based saliency

5. **Real-time implementation** (future)
   - Stream EEG → compute connectivity → predict
   - Latency requirements for clinical use

---

## File Structure

```
SeizureProp/
├── main.py                      # Training entry point
├── datasetConstruct.py          # Data pipeline
├── connectivity_cal.py          # Connectivity features
├── models.py                    # Model architectures
├── s4.py, s4d.py               # State space models
├── steps.py                     # Training loop
├── plotFun.py                   # Visualization
├── utils.py                     # Helpers
│
├── data/                        # Training data
├── data_test/                   # Test data
├── checkpoints/                 # Saved model weights
├── results/                     # Outputs (metrics, plots)
│
├── src/                         # Additional modules
├── models/                      # Model definitions
├── scripts/                     # Analysis scripts
├── extensions/                  # Custom modules
└── README.md                    # Original README
```

---

## Quick Start (New User)

```bash
# 1. Explore data
python test_dataset.py --data_dir data/

# 2. Train baseline model (S4D is fastest)
python main.py \
    --model s4d \
    --data_dir data/ \
    --epochs 50 \
    --save_path checkpoints/baseline.pt

# 3. Check results
# → Check results/ folder for ROC curves, metrics, etc.
# → Open latest [DATE]ModelTesting.ipynb to see experiment log

# 4. Iterate on model/connectivity/hyperparams
```

---

## Troubleshooting

**Model not converging?**
- Lower learning rate (1e-4 instead of 1e-3)
- Check data scaling (should be normalized)
- Increase training epochs

**Out of memory?**
- Reduce batch size (16 instead of 32)
- Use smaller model (S4D instead of S4)
- Reduce input sequence length

**Bad test performance but good training?**
- Likely overfitting → use dropout, L2 regularization
- Check for data leakage (temporal overlap in train/test)
- Try leave-one-patient-out CV

**Results file won't save?**
- Check `results/` folder exists
- Check write permissions
- Verify file path in `plotFun.py`

---

## References

1. Gu, A., et al. (2022). "Efficiently Modeling Long Sequences with Structured State Spaces." ICLR 2023.
2. Gupta, A., et al. (2023). "Diagonal State Spaces are as Effective as Structured State Spaces." NeurIPS 2023.
3. Rummel, C., et al. (2011). "Ictal flow of spreading depolarization in focal epilepsy." *J Neurosci*, 31(23), 8602-8609.

---

## Contact

Questions? Check the most recent `[DATE]ModelTesting.ipynb` notebook for recent experimental setups and results!
