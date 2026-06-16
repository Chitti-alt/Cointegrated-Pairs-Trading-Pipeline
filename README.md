
# Cointegrated Pairs Trading Pipeline

A fully vectorized quantitative backtesting pipeline that identifies mathematically cointegrated equity pairs and simulates a mean-reversion trading strategy using rolling Z-scores.

---

## 📌 Project Overview

This project serves as a proof-of-concept for a **Statistical Arbitrage (StatArb)** trading system.

Unlike traditional momentum-based algorithms, this engine does **not rely on directional market forecasts**. Instead, it systematically scans a universe of equities to identify pairs whose price relationship exhibits statistical mean reversion.

The framework leverages:

* **Cointegration testing** via the Augmented Dickey-Fuller (ADF) test
* **Ordinary Least Squares (OLS)** regression for hedge ratio estimation
* **Rolling Z-score normalization** for signal generation
* **Fully vectorized backtesting** using Pandas and NumPy
* **Look-ahead bias prevention** through proper signal shifting

The result is a fast, scalable, and statistically grounded trading engine suitable for research and educational purposes.

---

## 🚀 Key Features

### Cointegration Testing

Uses `statsmodels` to perform Augmented Dickey-Fuller (ADF) tests across all possible asset pairs within a selected universe.

Rather than relying on simple correlation, the engine identifies pairs with statistically significant cointegrated relationships:

[
p < 0.05
]

ensuring the spread exhibits mean-reverting behavior.

---

### Vectorized Backtesting

The strategy is implemented entirely with:

* NumPy arrays
* Pandas vectorized operations
* Boolean masking

This eliminates computationally expensive iterative loops and enables rapid evaluation across large historical datasets.

---

### Look-Ahead Bias Prevention

To ensure realistic simulation results, all trading signals are shifted forward by one trading day before position application:

```python
positions = signals.shift(1)
```

This guarantees that trades are executed only after the signal becomes available.

---

### Risk & Performance Analytics

The engine automatically computes several industry-standard performance metrics:

* Annualized Return
* Annualized Volatility
* Sharpe Ratio
* Maximum Drawdown
* Cumulative Returns

---

## 🧮 Mathematical Foundation

### 1. Hedge Ratio Estimation

For a pair of assets (A) and (B), an OLS regression is used to estimate the hedge ratio (n):

[
Price_A = \alpha + n \cdot Price_B + \epsilon
]

where:

* (n) = hedge ratio
* (\epsilon) = residual error

---

### 2. Spread Construction

The spread between the two assets is defined as:

[
Spread = Price_A - (n \times Price_B)
]

A stationary spread indicates a potential mean-reverting trading opportunity.

---

### 3. Rolling Z-Score

To normalize the spread, a rolling 21-day Z-score is calculated:

[
Z = \frac{Spread - \mu}{\sigma}
]

where:

* (\mu) = rolling mean
* (\sigma) = rolling standard deviation

---

### 4. Trading Signals

#### Short Spread Signal

When the spread becomes abnormally wide:

[
Z > 2
]

Action:

* Short Asset A
* Long Asset B

---

#### Long Spread Signal

When the spread becomes abnormally narrow:

[
Z < -2
]

Action:

* Long Asset A
* Short Asset B

---

#### Exit Signal

Positions are closed when the spread reverts back toward equilibrium:

[
Z = 0
]

---

## 🏗️ Strategy Workflow

```text
Download Historical Prices
           │
           ▼
Generate All Asset Pairs
           │
           ▼
Run Cointegration Tests
           │
           ▼
Select Lowest p-value Pair
           │
           ▼
Estimate Hedge Ratio (OLS)
           │
           ▼
Construct Spread
           │
           ▼
Calculate Rolling Z-Score
           │
           ▼
Generate Entry / Exit Signals
           │
           ▼
Vectorized Backtest
           │
           ▼
Performance Evaluation
```

---

## 🛠️ Installation

### Prerequisites

* Python 3.8+
* Internet connection for Yahoo Finance data retrieval

Install required dependencies:

```bash
pip install yfinance pandas numpy statsmodels matplotlib
```

---

## 📂 Repository Structure

```text
cointegrated-pairs-pipeline/
│
├── stat_arb_backtester.py
├── README.md
└── requirements.txt
```

---

## ▶️ Running the Engine

Clone the repository:

```bash
git clone https://github.com/Chitti-Alt/cointegrated-pairs-pipeline.git
```

Navigate to the project directory:

```bash
cd cointegrated-pairs-pipeline
```

Run the backtester:

```bash
python stat_arb_backtester.py
```

---

## ⚙️ Customization

You can easily modify the asset universe by changing the ticker list inside the main script:

```python
tickers = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "META",
    "AMZN"
]
```

Potential sectors to test:

* Technology
* Financials
* Energy
* Healthcare
* Cryptocurrencies
* ETFs

---

## 📈 Output

After execution, the engine reports:

### Cointegrated Pair Selection

```text
Best Pair:
AAPL - MSFT

ADF p-value:
0.0123
```

---

### Performance Statistics

```text
Annualized Return: 14.82%
Sharpe Ratio:      1.47
Max Drawdown:     -6.21%
```

---

### Visualization Dashboard

A two-panel Matplotlib figure is generated:

#### 1. Cumulative Returns Curve

Displays strategy equity growth over the backtesting period.

#### 2. Z-Score Spread Tracker

Visualizes:

* Rolling spread Z-score
* Mean level (0)
* Long threshold (-2)
* Short threshold (+2)
* Mean-reversion exits

---

## 📚 Technologies Used

* Pandas
* NumPy
* Statsmodels
* Matplotlib
* Yahoo Finance API (`yfinance`)

---

## ⚠️ Disclaimer

This project is intended for educational and research purposes only.

Past performance does not guarantee future results. Trading financial instruments involves significant risk, and this code should not be considered financial advice.

---

## 📄 License

This project is released under the MIT License.

Feel free to fork, modify, and extend the framework for your own quantitative research projects.


