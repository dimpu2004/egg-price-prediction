# Egg Price Prediction using GRU

This project implements a GRU-based sequence-to-sequence (Seq2Seq) time series
model for forecasting daily egg prices.

The objective is to capture both temporal dynamics and external market influences
that affect egg price fluctuations, such as feed costs, regional price signals,
inflation indicators, and event-driven shocks.

---

## Problem Statement

Egg prices exhibit strong temporal patterns influenced by seasonality, inflation,
supply disruptions, and demand-side events. Traditional statistical models often
struggle to capture these nonlinear dependencies.

This project explores a deep learning–based GRU model to accurately forecast
short-term egg prices using historical and exogenous features.

---

## Dataset Description

The dataset was constructed by merging multiple time-aligned sources, including:

- Daily egg prices (target variable)
- Regional wholesale prices (e.g., Namakkal, Hyderabad)
- Feed prices (maize and soymeal)
- CPI indicators:
  - CPI – Eggs
  - CPI – Food & Beverages
  - CPI – Meat & Fish
- Event-based indicators:
  - Bird flu demand panic
  - Supply shock flags
  - Shravan month intensity

Due to large file size and data sourcing constraints, the **full dataset is not
included** in this repository. The provided notebook assumes a preprocessed CSV
with the above features.

---

## Model Architecture

- GRU-based Seq2Seq neural network
- Input window: 30 past days
- Forecast horizon: 7 future days
- MinMax normalization
- L2 regularization
- Gradient clipping for training stability
- Early stopping based on validation loss

The implementation follows a multivariate time series forecasting approach.

---

## Evaluation Metrics

Model performance on the test set:

- **MAE**  : 0.5490  
- **RMSE** : 0.8890  
- **MAPE** : 0.15%  
- **R²**   : 0.9986  

These metrics indicate strong predictive performance and close alignment
between predicted and actual prices.

---

## Results

Key visualizations are available in the `results/` folder, including:

- Actual vs Predicted price comparison
- 7-day ahead forecast visualization
- Feature correlation heatmap

---

## Tools & Libraries

- Python
- NumPy, Pandas
- TensorFlow / Keras
- Matplotlib, Seaborn
- Google Colab

---

## Notes

This repository focuses on model design, evaluation, and interpretation.
It is intended for academic and exploratory purposes.
