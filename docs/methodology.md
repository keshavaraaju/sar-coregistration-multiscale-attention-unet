# Methodology Summary

From Thesis Chapter 3.

## Overview (p. 16)
The approach integrates data preparation, model design, training, evaluation.

## Dataset Preparation (pp. 16-19)
- Load .npy files.
- Crop to min shape.
- Magnitude computation.
- Min-Max normalization.
- Resize to 128x128.
- Channel stacking (input: pri+sec, output: az+rg).
- TF dataset with shuffle, batch 32, prefetch.

## Model Architecture (pp. 19-20)
- Encoder: 3 residual layers (32,64,128), max pool.
- Bridge: Residual 256 + dropout 0.5.
- Decoder: Transpose conv, multiscale attention (1x1,3x3,5x5 dilations), residual.
- Output: Conv2D 2 filters, linear.

## Training (pp. 20-21)
- MSE loss.
- AdamW optimizer.
- 20 epochs, early stopping patience 10.

## Evaluation (pp. 21-22)
- MAE/MSE.
- Colormap plots.
- Training history.

See figures/model_flowchart.png for workflow.
