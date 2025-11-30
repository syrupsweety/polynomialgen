
# Polynomial Language Modeling via Embedding Curves

## Overview

This notebook explores a novel approach to modeling and generating natural language by treating sequences of token embeddings as continuous trajectories in high-dimensional space. Instead of using traditional autoregressive or attention-based methods, we fit **polynomial curves** to the semantic paths formed by token embeddings from a pre-trained LLM (Llama-3.2-1B-instruct).

The core idea is that sentence meaning unfolds smoothly across time, and this progression can be approximated as a low-degree polynomial function of normalized position. By learning such curves, we can reconstruct, analyze, and even generate text by evaluating the curve at discrete timesteps and finding nearest neighbor tokens.

Two main experiments are conducted:

1. **Diagnostics & Reconstruction**: Fit a polynomial to a single sentence's embedding path.
2. **Generative Modeling**: Learn a shared "sentence template" from multiple structurally similar sentences and generate variations.

---

## Key components

### Models & Libraries
- **Model**: `huihui-ai/Llama-3.2-1B-Instruct-abliterated` (via Hugging Face Transformers)
- **Libraries**: PyTorch, scikit-learn, NumPy, Matplotlib
- **Core Techniques**:
  - PCA for dimensionality reduction
  - Polynomial regression with Tikhonov regularization
  - Nearest-neighbor decoding in embedding space

### Parameters
| Parameter | Value | Purpose |
|--------|-------|--------|
| `MAX_DEGREE` | 10 | Maximum degree of fitted polynomial |
| `LAMBDA` | 1e-6 | Regularization strength |
| `COMPONENTS` | 32 | PCA target dimensions |
| `TEXT` | `"The cat sat..."` | Diagnostic input |
| `TEXT_SAMPLES` | 7 sentences | Generative training set |

---

## Experiment 1: Diagnostics & Reconstruction

We take a single sentence and attempt to reconstruct it by fitting increasingly complex polynomials to its embedding trajectory.

### Steps:
1. Tokenize input sentence → get input embeddings.
2. Reduce embeddings with PCA (from 2048D → 12D).
3. Fit polynomial.
4. Evaluate curve at each timestep, decode back to tokens via nearest neighbors.

### Results (best case):
```
Original:        The cat sat on the mat and stared at the wall.
Reconstructed:   The cat sat on the mat and stared at the wall.
Accuracy:        100.00% (achieved at degree 10)
MSE:             0.000079
```

### Output visualization
A diagnostic plot (`polynomial_language_diagnostics.png`) shows:
- Top-left: Original vs. reconstructed paths in 2D PCA space.
- Top-right: MSE ↓ and accuracy ↑ vs. polynomial degree.
- Bottom-left: Table of all reconstructions per degree.
- Bottom-right: Norms of final polynomial coefficients.

---

## Experiment 2: Generative modeling

Using 7 syntactically parallel sentences (e.g., *"The X did Y on/in/at Z."*), we:
1. Pool all embeddings after padding.
2. Apply PCA to reduce dimensionality.
3. Compute the mean trajectory across sentences.
4. Fit a single global polynomial to this average path.
5. Generate new sentences by evaluating and perturbing the curve.

### Modifications tested:
| Modification | Effect | Example Output |
|------------|--------|----------------|
| **Base** | No change | `The dog sat on the mat.` |
| **Extended** | Slight extrapolation | `dog slept on on the the rug..` |
| **Sharpened** | Boost higher-order terms | `slept slept slept...` |
| **Creative** | Add noise | `The cat sat on the mat..` |
| **Reversed** | Flip time order | `..mat the on sat dogThe` |


---

## ⚙️ Requirements

You must have access to the model:
```bash
MODEL_NAME = "huihui-ai/Llama-3.2-1B-Instruct-abliterated"
```

---

## 🛠️ Usage

1. Run `polynomialgen.ipynb` in any Jupyter environment (Colab, JupyterLab, etc.).
2. Modify `TEXT` or `TEXT_SAMPLES` to test custom inputs.
3. Adjust `MAX_DEGREE`, `LAMBDA`, or `COMPONENTS` to explore trade-offs.
