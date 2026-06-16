# Cointegrated-Pairs-Trading-Pipeline
A fully vectorized, quantitative backtesting pipeline that identifies mathematically cointegrated equity pairs and simulates a mean-reversion trading strategy using rolling Z-scores.
📌 Project Overview
This project serves as a proof-of-concept for a Statistical Arbitrage (StatArb) trading system. Unlike basic momentum algorithms, this engine does not rely on directional market predictions. Instead, it scans a universe of equities to find pairs whose price spread is mathematically stationary (mean-reverting).
It utilizes the Augmented Dickey-Fuller (ADF) test for cointegration and executes simulated trades using a purely vectorized backtesting engine built in Pandas and NumPy, ensuring high-performance  array operations across historical time-series data.
🚀 Key Features
Cointegration Testing: Uses statsmodels to run ADF tests across all possible asset pairs within a defined universe, identifying statistically significant relationships () rather than relying on simple price correlation.
Vectorized Backtesting: Eliminates slow iterative for loops by utilizing boolean masking and array shifting to calculate daily positions and returns instantaneously.
Look-Ahead Bias Prevention: Implements strict .shift(1) logic on position sizing arrays to ensure trades are executed strictly on the following day's return after a signal is generated.
Risk Management Metrics: Automatically computes industry-standard performance metrics, including Annualized Return, Sharpe Ratio, and Maximum Drawdown.
🧮 Mathematical Foundation
1. The Spread
The algorithm calculates the hedge ratio () between two cointegrated stocks using Ordinary Least Squares (OLS) regression. The spread is defined as:
Spread = Price_A - (n * Price_B)
2. Signal Generation
To normalize the spread, we calculate a rolling Z-score over a 21-day window:
Z = (Spread - μ) / σ
Short Signal: If , the spread is abnormally wide (Short Asset A, Long Asset B).
Long Signal: If , the spread is abnormally narrow (Long Asset A, Short Asset B).
Exit Signal: Positions are flattened when  crosses  (reversion to the mean).
🛠️ Installation & Usage
Prerequisites
Ensure you have Python 3.8+ installed. Install the required data science and quantitative finance libraries:
pip install yfinance pandas numpy statsmodels matplotlib


Running the Engine
Simply clone the repository and execute the main script:
git clone [https://github.com/Chitti-Alt/cointegrated-pairs-pipeline.git]
cd cointegrated-pairs-pipeline
python stat_arb_backtester.py


(Note: You can easily modify the tickers array in the __main__ block to test different market sectors, such as Tech, Energy, or Crypto.)
📈 Output & Visualization
Upon execution, the script outputs the most highly cointegrated pair along with its p-value. It generates a two-panel Matplotlib figure containing:
Cumulative Returns Curve: Showcasing the algorithmic performance over the test period.
Z-Score Spread Tracker: Visualizing the dynamic spread, clearly marking the ±2.0 standard deviation entry thresholds and the mean-reversion exit points.
