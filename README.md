# Integrating Biological and Machine Learning Models for Rainbow Trout Growth: Balancing Accuracy and Interpretability

**Lawrence Fulton & Pin Lyu**  
Applied Analytics, Boston College  
`fultonl@bc.edu`

---

## Overview

This repository contains all data, preprocessing pipelines, and modeling code for the paper:

> *Integrating biological and machine learning models for rainbow trout growth: Balancing accuracy and interpretability*  
> Fulton, L. & Lyu, P. (2026). PLOS ONE *(forthcoming)*.

We develop a unified, probabilistic framework for forecasting the fork length growth of invasive rainbow trout (*Oncorhynchus mykiss*) in the Lower Colorado River. The framework integrates traditional biological growth models with modern machine learning (ML) and ensemble methods, evaluated through rigorous Bayesian probabilistic comparison.

**Key result:** A stacked ensemble combining XGBoost and the Bayesian von Bertalanffy Growth Model (VBGM) achieved the best performance (RMSE = 15.96 mm, R² = 0.966), reducing prediction error by **~81.5%** relative to the covariate-free baseline VBGM — equivalent to a 70 mm improvement, or roughly 32% of mean fish length.

---

## Background

Rainbow trout were intentionally introduced into the Lower Colorado River from 1964–1998 to support recreational fishing. They have since displaced endangered native species such as the humpback chub (*Gila cypha*) and razorback sucker (*Xyrauchen texanus*). At the same time, the species generates substantial economic value (~$14M in Arizona fishing licenses in 2020 alone). Accurate growth forecasting is therefore essential for adaptive population management that balances ecological conservation with economic priorities.

---

## Data

Data are sourced from the USGS publicly available datasets:

> Korman, J., & Yard, M.D. (2017). **Rainbow Trout Growth Data and Growth Covariate Data from Glen Canyon, Colorado River, Arizona (2012–2021)**. U.S. Geological Survey.

The merged dataset contains **9,798 tag-recapture observations** with 19 predictors across five biological categories:

| Category | Variables |
|---|---|
| Initial condition | Fork length at release (L1), Weight at release |
| Spatial | Release river mile |
| Seasonal | Release month, Recovery month (dummy-encoded) |
| Temporal | Time at large (days), Release year, Recovery year (dummy-encoded) |
| Environmental | Discharge, Water temperature, Solar insolation, Soluble reactive phosphorus, Rainbow trout biomass |

**Response variable:** Fork length at recapture (L2, mm)

> **Note on pseudo-replication:** The dataset lacks consistent individual identifiers across capture events. Each observation is treated as independent. Sensitivity analyses confirmed that model rankings are stable under alternative data partitions.

---

## Models

Thirteen model configurations were evaluated across four paradigms:

### Biological Growth Models (Bayesian)
- **Baseline VBGM** — Fabens formulation, no covariates
- **Baseline Gompertz** — no covariates
- **Bayesian VBGM** — covariate-augmented via log-linear growth rate; estimated with NUTS/MCMC in NumPyro
- **Bayesian Gompertz** — same covariate structure as Bayesian VBGM

### Statistical Models
- **Bayesian Linear Regression** — weakly informative priors, NUTS sampling

### Machine Learning Models
- **Random Forest (RF)** — bagging ensemble, randomized hyperparameter search
- **XGBoost** — depth-wise gradient boosting, randomized search + 5-fold CV
- **LightGBM** — leaf-wise gradient boosting, randomized search + 5-fold CV
- **SVR (Linear)** — linear kernel, interpretable coefficients
- **SVR (RBF)** — kernel-optimized, permutation feature importance
- **Artificial Neural Network (ANN)** — 3 hidden layers (256→128→64), SiLU activations, batch norm, dropout, AdamW optimizer

### Ensemble Methods
- **Stacked Ensemble** — linear meta-learner trained on out-of-sample VBGM + XGBoost predictions
- **Bayesian Model Averaging (BMA)** — AIC-weighted combination across candidate models

---

## Methods Summary

```
Mark-Recapture Data
      │
      ▼
Merge with Environmental Covariates (weighted average over time-at-large)
      │
      ▼
Train/Test Split: 70% / 30%  ·  Min-Max Scaling  ·  Fixed seed
      │
      ▼
Model Training: 5-Fold Cross-Validation + Hyperparameter Tuning
      │
      ├── Biological Models (NumPyro / NUTS / MCMC)
      ├── Tree-Based ML (RF, XGBoost, LightGBM)
      ├── Other ML (SVR, ANN / PyTorch)
      └── Ensemble Methods (Stacking, BMA)
      │
      ▼
Held-Out Test Set Evaluation
      │
      ▼
Bayesian Probabilistic Comparison of RMSE (Pairwise Posterior Dominance)
```

All Bayesian models used 4 chains × 1,500 draws (500 tuning + 1,000 posterior), target acceptance = 0.95. Gelman-Rubin R̂ = 1.00 for all parameters; ESS > 6,000 for most.

---

## Results

### Model Performance (Test Set, sorted by RMSE)

| Model | RMSE (mm) | MAE (mm) | R² | % RMSE < Baseline VBGM |
|---|---|---|---|---|
| **Stacked Ensemble** | **15.96** | **10.62** | **0.9658** | **81.5%** |
| XGBoost | 16.14 | 10.72 | 0.9650 | 81.3% |
| BMA Ensemble | 16.14 | 10.72 | 0.9650 | 81.3% |
| LightGBM | 16.25 | 10.80 | 0.9645 | 81.2% |
| Bayesian VBGM | 16.82 | 11.60 | 0.9620 | 80.5% |
| Bayesian Gompertz | 16.92 | 11.67 | 0.9615 | 80.4% |
| Random Forest | 18.40 | 12.20 | 0.9545 | 78.7% |
| ANN | 18.47 | 13.09 | 0.9542 | 78.6% |
| SVR (RBF) | 18.81 | 11.77 | 0.9524 | 78.2% |
| SVR (Linear) | 25.35 | 15.40 | 0.9137 | 70.6% |
| Bayesian Linear | 25.72 | 18.54 | 0.9111 | 70.2% |
| Baseline Gompertz | 61.52 | 49.81 | 0.492 | 28.7% |
| Baseline VBGM | 86.29 | 77.19 | −0.000 | — |

### Top Predictors (consistent across models)
1. **L1** — Fork length at release
2. **Time at large**
3. **Weight at release**

### Stochastic Dominance
The stacked ensemble exhibited stochastic dominance over all individual models across the full posterior RMSE distribution. The BMA ensemble outperformed the Bayesian VBGM in 70.6% of posterior samples and the Gompertz in 88.5%.

---

## Repository Structure

```
├── README.md
├── fish_model_data.csv     # USGS source data (fish + environmental)
└── final02182026.ipynb                   # Main analysis notebook
```

---

## Requirements

```bash
Python 3.13+

# Core
numpy
pandas
scipy
matplotlib
seaborn
openpyxl
python-dateutil

# Machine Learning
scikit-learn
xgboost
lightgbm
torch          # PyTorch (GPU recommended)
tqdm

# Bayesian Modeling
jax
jaxlib
numpyro
arviz
```

Install all dependencies:

```bash
pip install numpy pandas scipy matplotlib seaborn openpyxl python-dateutil \
            scikit-learn xgboost lightgbm torch tqdm jax jaxlib numpyro arviz
```

> **GPU note:** The Bayesian models and ANN benefit significantly from GPU acceleration. The notebook was developed on an NVIDIA RTX 5080. For CPU-only environments, reduce MCMC chains/draws and ANN epochs as needed.

---

## Reproducing the Analysis

1. Clone the repository and install dependencies (see above).
2. Place `RainbowTrout_Growth_Data.xlsx` in the working directory (or update `file_path` in the notebook).
3. Run `final06072025.ipynb` end-to-end.  
   - Cells 1–28 handle preprocessing and export `fish_model_data.csv`.  
   - Start at **Cell 29** ("*Start Here: Metrics Function*") to bypass preprocessing using the saved CSV.
4. All random seeds are fixed for reproducibility. Bayesian MCMC results may show minor chain-to-chain variation.

---

## Key Methodological Notes

**Environmental covariate integration:** For fish spanning multiple monthly intervals, environmental variables (temperature, discharge, etc.) are integrated as time-weighted averages over the full time-at-large period, preserving ecological fidelity.

**Scaling strategy:** ML models use full Min-Max scaling. Biological models use partial scaling, preserving L1 and Time at Large in native units for interpretability.

**Information criteria for ML models:** AIC/BIC are approximated from Gaussian residuals (pseudo-likelihood). These should be interpreted heuristically for tree-based models, not as strict likelihood-based statistics.

**Stacking:** The meta-learner is trained exclusively on out-of-sample (cross-validated) predictions to prevent data leakage.

**BMA weights:** Computed from AIC-based model evidence as a computationally tractable approximation to full Bayesian model averaging.

---

## Citation

If you use this code or data in your work, please cite:

```bibtex
@article{fulton2026rainbowtrout,
  title   = {Integrating biological and machine learning models for rainbow trout 
             growth: Balancing accuracy and interpretability},
  author  = {Fulton, Lawrence and Lyu, Pin},
  journal = {PLOS ONE},
  year    = {2026},
  note    = {Forthcoming}
}
```

Please also cite the underlying USGS dataset:

```bibtex
@data{korman2022usgs,
  author    = {Korman, Josh and Yard, Michael D.},
  title     = {Rainbow trout growth data and growth covariate data from Glen Canyon, 
               Colorado River, Arizona, 2012--2021},
  publisher = {U.S. Geological Survey},
  year      = {2022},
  doi       = {10.5066/P9LGPUOE}
}
```

---

## Use of AI Tools

ChatGPT-4o was used for editing content. Claude.ai was used for code debugging. All content was reviewed and edited by the authors, who take full responsibility for the published work.

---

## License

This repository is released under the [MIT License](LICENSE).  
The underlying USGS data are publicly available and freely redistributable.

---

## Contact

**Lawrence Fulton** — `fultonl@bc.edu`  
Applied Analytics Program, Woods College of Advancing Studies, Boston College
