# SAR Co-Registration with Multiscale Attention U-Net

[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview
This repository reconstructs the implementation from my Master's thesis: "A Study on Co-registration of Synthetic Aperture Radar (SAR) images using Multiscale Attention U-Net Architecture" (April 2025, Technische Universität Berlin). The thesis focuses on estimating offsets in azimuth and range directions for SAR image co-registration without prior acquisition geometry, using a deep learning framework based on Attention U-Net [7].

Key contributions (from Abstract, p. IX):
1. Development of Attention-based U-Net architecture with multiscale features, residual blocks, and Squeeze-and-Excitation (SE) modules.
2. Creation of a custom dataset with normalized SAR data and offset labels.
3. Implementation of training strategy for optimal performance (AdamW optimizer, MSE loss, early stopping).
4. Evaluation and visualization of results (MAE, MSE, colormaps).

The model handles high-dimensional SAR data, noise, and complex patterns, achieving sub-pixel accuracy. It represents a cutting-edge solution for SAR geometric correction in scenarios with unknown relative positions (e.g., earthquake studies [12]).

**Note**: This is a reconstruction due to original code/data loss. Demonstrates expertise in DL for remote sensing. For full details, see [thesis.pdf](thesis.pdf).

## Motivation (from Section 1.2, pp. 5-6)
Traditional methods require prior acquisition info and struggle with noise, decorrelation, large displacements, and computational complexity. Deep learning (e.g., U-Net variants) offers automated, robust alternatives. This work addresses speckle noise, algorithmic inefficiency, and computation by integrating residual learning, multiscale attention, and SE blocks.

## Methodology Summary (from Chapter 3, pp. 16-23)
- **Dataset Preparation**: Load .npy files (primary/secondary images, AZ/RG offsets). Crop to min shape (48292x18360), compute magnitude, normalize [0,1], resize to 128x128, stack channels (input: 2-ch, output: 2-ch). Split 80/20 train/val, batch size 32.
- **Model Architecture**: Encoder (3 residual layers: 32→64→128 filters), Bridge (256 filters + dropout 0.5), Decoder (transposed conv + multiscale attention: 1x1/3x3/5x5 conv with dilations). Output: Conv2D (2 filters, linear).
- **Training**: MSE loss, AdamW optimizer, 20 epochs, early stopping (patience 10).
- **Evaluation**: MAE/MSE, colormap visualizations, training history plots.

Flowchart: See docs/figures/model_flowchart.png (extract from p. 23).

## Installation
1. Clone the repo: `git clone https://github.com/yourusername/sar-coregistration-multiscale-attention-unet.git`
2. Install dependencies: `pip install -r requirements.txt`
3. (Optional) Build Docker: `docker build -t sar-unet .`

## Usage
- Run main script: `python src/main.py` (loads sample data, trains, evaluates, visualizes).
- Interactive demo: Open notebooks/thesis_demo.ipynb in Jupyter.
- Example: Train on your data – replace placeholders in data/sample/.

## Results Highlights (from Chapter 5, pp. 31-37)
- Final MAE: Train 0.0033, Val 0.0079 (low error indicates strong generalization).
- Training time: ~9 hours (20 epochs, 3050s/epoch on AMD Ryzen 9 with 64GB RAM).
- Ablation studies show Multiscale Attention improves accuracy over baselines (e.g., Simple U-Net MAE higher by ~0.004).
- Visuals: See docs/figures/ (e.g., comparison_plots.png from p. 33).

Data Stats (from Tables 5.1-5.3, pp. 31-32):

| Metric | Value |
|--------|-------|
| Primary Shape | (48305, 18360) |
| Secondary Shape | (48292, 18360) |
| Processed X Shape | (48292, 128, 128, 2) |
| Train/Val Split | 38633 / 9659 samples |

## Limitations & Future Work (from Chapters 6 & 8, pp. 38-43)
- Challenges: Noise, lack of labeled data, memory issues (mitigated by patching), no phase info.
- Future: Incorporate phase, hybrid CNN-transformers, unsupervised learning, edge deployment (ONNX/TensorRT).

For PhD/job applications: This repo demonstrates skills in TensorFlow/Keras, SAR processing, model optimization, and ablation analysis.

Contact: Keshava Raaju Perumal, keshava.raaju.p@gmail.com
