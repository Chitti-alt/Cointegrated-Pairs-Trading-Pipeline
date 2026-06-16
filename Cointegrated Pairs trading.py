import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for cleaner terminal output
warnings.filterwarnings('ignore')

def download_data(tickers, start_date, end_date):
    """
    Downloads historical adjusted close prices for a list of tickers.
    """
    print(f"Downloading data for {len(tickers)} tickers...")
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    data = data.dropna(axis=1) # Drop assets with missing data
    return data

def find_cointegrated_pairs(data, significance_level=0.05):
    """
    Iterates through all combinations of equities to find cointegrated pairs 
    using the Augmented Dickey-Fuller (ADF) test.
    """
    n = data.shape[1]
    keys = data.keys()
    pairs = []
    
    print("Running cointegration tests...")
    for i in range(n):
        for j in range(i+1, n):
            asset_1 = data[keys[i]]
            asset_2 = data[keys[j]]
            
            # Run the cointegration test
            score, p_value, _ = coint(asset_1, asset_2)
            
            if p_value < significance_level:
                pairs.append((keys[i], keys[j], p_value))
                
    # Sort pairs by lowest p-value (strongest cointegration)
    pairs.sort(key=lambda x: x[2])
    return pairs

def backtest_pair(data, stock1, stock2, window=21, z_entry=2.0, z_exit=0.0):
    """
    A purely vectorized backtesting engine for a single pair of stocks.
    """
    print(f"Backtesting {stock1} and {stock2}...")
    df = pd.DataFrame(index=data.index)
    df[stock1] = data[stock1]
    df[stock2] = data[stock2]
    
    # 1. Calculate the Hedge Ratio using Ordinary Least Squares (OLS)
    # How much of stock2 do we need to short for every 1 share of stock1?
    model = sm.OLS(df[stock1], df[stock2]).fit()
    hedge_ratio = model.params[0]
    
    # 2. Calculate the Spread and Z-Score
    df['spread'] = df[stock1] - (hedge_ratio * df[stock2])
    df['spread_mean'] = df['spread'].rolling(window=window).mean()
    df['spread_std'] = df['spread'].rolling(window=window).std()
    df['z_score'] = (df['spread'] - df['spread_mean']) / df['spread_std']
    
    # 3. Vectorized Signal Generation
    # Identify exact moments where Z-score crosses our entry/exit thresholds
    df['long_entry'] = df['z_score'] < -z_entry
    df['long_exit'] = df['z_score'] >= z_exit
    df['short_entry'] = df['z_score'] > z_entry
    df['short_exit'] = df['z_score'] <= z_exit
    
    # 4. Vectorized Position Management (The "Quant" way to avoid for-loops)
    # 1 = Long Spread, -1 = Short Spread, 0 = Flat
    df['position_long'] = np.nan
    df.loc[df['long_entry'], 'position_long'] = 1
    df.loc[df['long_exit'], 'position_long'] = 0
    df['position_long'] = df['position_long'].ffill().fillna(0)
    
    df['position_short'] = np.nan
    df.loc[df['short_entry'], 'position_short'] = -1
    df.loc[df['short_exit'], 'position_short'] = 0
    df['position_short'] = df['position_short'].ffill().fillna(0)
    
    # Combine long and short positions
    df['position'] = df['position_long'] + df['position_short']
    
    # 5. Calculate Daily Returns
    df['ret_1'] = df[stock1].pct_change()
    df['ret_2'] = df[stock2].pct_change()
    
    # CRITICAL: We shift the position array by 1. We trade based on yesterday's 
    # closing signal to capture today's return. This prevents "look-ahead bias".
    df['strategy_return'] = df['position'].shift(1) * (df['ret_1'] - df['ret_2'])
    df['cumulative_return'] = (1 + df['strategy_return']).cumprod()
    
    # 6. Calculate Financial Metrics
    total_return = df['cumulative_return'].iloc[-1] - 1
    annualized_return = (1 + total_return) ** (252 / len(df)) - 1
    
    # Sharpe Ratio (Assuming 0% Risk-Free Rate for simplicity)
    daily_vol = df['strategy_return'].std()
    sharpe_ratio = np.sqrt(252) * (df['strategy_return'].mean() / daily_vol) if daily_vol > 0 else 0
    
    # Maximum Drawdown
    rolling_max = df['cumulative_return'].cummax()
    drawdown = (df['cumulative_return'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    metrics = {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }
    
    return df, metrics

def plot_results(df, stock1, stock2, metrics):
    """
    Generates a 2-panel chart showing the strategy returns and the Z-score bounds.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle(f'Statistical Arbitrage: {stock1} & {stock2}', fontsize=16)
    
    # Top Panel: Cumulative Returns
    ax1.plot(df.index, df['cumulative_return'], label='Strategy Cumulative Return', color='green')
    ax1.set_ylabel('Cumulative Return')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Print metrics on the chart
    metrics_text = (f"Ann. Return: {metrics['Annualized Return']:.2%}\n"
                    f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}\n"
                    f"Max Drawdown: {metrics['Max Drawdown']:.2%}")
    ax1.text(0.02, 0.75, metrics_text, transform=ax1.transAxes, fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    # Bottom Panel: Z-Score and Trading Thresholds
    ax2.plot(df.index, df['z_score'], label='Spread Z-Score', color='blue', linewidth=1)
    ax2.axhline(2.0, color='red', linestyle='--', label='Short Entry Threshold (+2.0)')
    ax2.axhline(-2.0, color='green', linestyle='--', label='Long Entry Threshold (-2.0)')
    ax2.axhline(0, color='black', linestyle='-', alpha=0.5, label='Mean / Exit Signal (0)')
    ax2.set_ylabel('Z-Score')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Define a universe of highly correlated stocks (e.g., US Financial Sector)
    tickers = ['JPM', 'BAC', 'C', 'WFC', 'GS', 'MS']
    start_date = '2018-01-01'
    end_date = '2024-01-01'
    
    # Step 1: Download the data
    historical_data = download_data(tickers, start_date, end_date)
    
    # Step 2: Find the most cointegrated pair
    cointegrated_pairs = find_cointegrated_pairs(historical_data)
    
    if not cointegrated_pairs:
        print("No cointegrated pairs found in this universe and timeframe.")
    else:
        # Get the pair with the lowest p-value
        best_pair = cointegrated_pairs[0]
        stock1, stock2, p_val = best_pair
        print(f"\nBest Cointegrated Pair: {stock1} & {stock2} (p-value: {p_val:.4f})")
        
        # Step 3 & 4: Run the vectorized backtest
        df_results, strategy_metrics = backtest_pair(historical_data, stock1, stock2)
        
        # Step 5: Visualize
        plot_results(df_results, stock1, stock2, strategy_metrics)