# Model Outputs

This directory contains trained model outputs:

## From `scripts/train_bfm.py` (Cross-Validation):
- `oof_mean.npy` - Out-of-fold mean predictions
- `oof_epistemic.npy` - Out-of-fold epistemic variance  
- `oof_aleatoric.npy` - Out-of-fold aleatoric variance

## From `scripts/generate_predictions.py` (Full Training):
- `trained_model.pkl` - Trained model with encoders (~130 MB)
- `full_predictions.parquet` - Full prediction matrix (~560 MB)

Note: These files are large and excluded from git. Either:
1. Run the training scripts to generate them, or
2. Copy them from a previous run
