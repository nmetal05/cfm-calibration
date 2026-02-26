# CFM Calibration with Simulation-Based Inference

Calibrating car-following models (Krauss, Wiedemann, IDM) using SNPE and ASNPE for urban traffic simulation.

## Structure

```
CFM_Comparison/   - Comparison scripts and plots
Krauss/           - Krauss model results + SNPE vs ASNPE
IDM/              - IDM model results
Wiedemann/        - Wiedemann model results
```

## Key Results

| Model | RMSE | NSE |
|-------|------|-----|
| IDM | 0.54 | 0.998 |
| Krauss | 1.98 | 0.968 |
| Wiedemann | 6.13 | 0.690 |

SNPE outperforms ASNPE with 2-3× lower posterior uncertainty.

## Requirements

- SUMO
- Python 3.10+
- sbi, torch, numpy, pandas

## Usage

```bash
# Generate training data
python generate_training_data.py

# Train SNPE
python train_snpe_v3.py

# Compare models
python CFM_Comparison/create_plots.py
```

