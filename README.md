# Week 8: LSTM on Yahoo Finance Stock Data

This project implements a Time Series Forecasting pipeline using Long Short-Term Memory (LSTM) networks to predict stock prices. It specifically analyzes Apple Inc. (AAPL) stock data fetched from Yahoo Finance.

## Project Overview
The repository contains both a modular Python script and an interactive Jupyter Notebook that perform:
1. **Data Acquisition**: Fetching historical stock data using `yfinance`.
2. **EDA (Exploratory Data Analysis)**: Visualizing closing prices, rolling means, volatility, and daily returns.
3. **Data Preprocessing**: Scaling features using `MinMaxScaler` and creating sequential datasets for time-series modeling.
4. **Model Architecture**: Building a deep learning model using **PyTorch**.
   - Hidden Layers: LSTM + Dropout
   - Output: Fully Connected (Linear)
5. **Evaluation**: Measuring performance using RMSE, MAE, and R² Score.
6. **Hyperparameter Tuning**: Fine-tuning neurons, learning rates, and epochs to optimize performance.

## Files
- `week8.py`: The core implementation in script format.
- `week8.ipynb`: Interactive Jupyter Notebook version for step-by-step analysis.
- `README.md`: Project documentation.

## Installation & Setup
To run this project locally, ensure you have Python installed, then install the required dependencies:

```bash
pip install yfinance pandas numpy matplotlib seaborn scikit-learn torch
```

## How to Run
### Python Script
```bash
python week8.py
```

### Jupyter Notebook
Open `week8.ipynb` in your preferred editor (VS Code, JupyterLab, etc.) and run the cells sequentially.

## Key Observations
- **Learning Rates**: High learning rates (>0.01) can lead to instability, while very low ones require significant epochs to converge.
- **Model Capacity**: Increasing neurons from 16 to 64 helps capture complex volatility patterns but requires careful tuning to avoid overfitting.
- **Epochs**: Training for 100+ epochs significantly improves the R² score compared to short training runs.

---
*Created as part of the Fintech Stock And Churn Modeling assignment.*
