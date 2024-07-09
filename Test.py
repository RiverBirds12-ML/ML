import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 設定股票代碼
tickers = ['QQQ', 'XLU', 'BRK-A']

# 下載包含股息的每月數據
data = yf.download(tickers, start='2003-01-01', end='2024-01-01', interval='1mo', auto_adjust=True)

# 刪除缺失值
data = data['Close'].dropna()
print(data.head())
# 初始資金
initial_capital = 10000

# 占比
weights = np.array([0.1195, 0.6801, 0.2004])

# 根據初始資金和權重計算初始購入的股數
initial_prices = data.iloc[0]
initial_investment = weights * initial_capital
shares = initial_investment / initial_prices

# 檢查初始投資分配
print("Initial Investment Distribution:")
print(initial_investment)

# 計算每月的投資組合價值
portfolio_value = data.dot(shares)

# 計算每個股票的價值
stock_values = data.mul(shares, axis=1)

# 計算月度回報率
monthly_returns = portfolio_value.pct_change().dropna()

# 計算年化回報率和波動率
years = (portfolio_value.index[-1] - portfolio_value.index[0]).days / 365.25
cumulative_return = (portfolio_value[-1] / portfolio_value[0]) - 1
annual_return = (1 + cumulative_return) ** (1 / years) - 1
annual_volatility = monthly_returns.std() * np.sqrt(12)

print(f"Portfolio Annual Return: {annual_return:.4f}")
print(f"Portfolio Annual Volatility: {annual_volatility:.4f}")

# 繪製資產隨時間變化的圖表
plt.figure(figsize=(12, 6))
plt.plot(portfolio_value.index, portfolio_value, label='Total Portfolio Value')
plt.plot(stock_values.index, stock_values['QQQ'], label='QQQ Value')
plt.plot(stock_values.index, stock_values['XLU'], label='XLU Value')
plt.plot(stock_values.index, stock_values['BRK-A'], label='BRK-A Value')
plt.title('Portfolio and Individual Stock Values Over Time')
plt.xlabel('Date')
plt.ylabel('Value ($)')
plt.legend()
plt.grid(True)
plt.show()

# 分別計算每個個股的年化回報率和波動率
for ticker in tickers:
    stock_return = stock_values[ticker].pct_change().dropna()
    cumulative_return_stock = (stock_values[ticker][-1] / stock_values[ticker][0]) - 1
    annual_return_stock = (1 + cumulative_return_stock) ** (1 / years) - 1
    annual_volatility_stock = stock_return.std() * np.sqrt(12)
    
    print(f"{ticker} Annual Return: {annual_return_stock:.4f}")
    print(f"{ticker} Annual Volatility: {annual_volatility_stock:.4f}")
