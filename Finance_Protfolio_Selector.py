import yfinance as yf
import pandas as pd
import numpy as np
import cvxpy as cp
import statsmodels.api as sm

# 設定股票代碼
raw_tickers = [
    'SPY', 'IVV', 'VOO', 'QQQ', 'VTI', 'IWM', 'DIA', 'EFA', 'VEU', 'VWO',
    'EEM', 'EWJ', 'FXI', 'EWG', 'EWZ', 'RSX', 'EWT', 'EWA', 'EWQ', 'EWU',
    'SCZ', 'SCHD', 'VIG', 'DVY', 'HDV', 'AGG', 'BND', 'TLT', 'IEF', 'SHY',
    'LQD', 'HYG', 'TIP', 'BNDX', 'EMB', 'VWOB', 'MUB', 'VCIT', 'VCSH', 'BSV',
    'BIV', 'TLH', 'MBB', 'GOVT', 'VTIP', 'BWX', 'CWB', 'PFF', 'PGX', 'ANGL',
    'XLE', 'XLU', 'KRE', 'XLP', 'ARKK', 'VEA', 'XBI', 'IEMG', 'SMH', 'XLV',
    'IAU', 'SPLG', 'BKLN', 'XLI', 'XLRE', 'XLF', 'XLY', 'XLB', 'XLK', 'XLC',
    'EWW', 'EWL', 'EWD', 'EWP', 'EIS', 'EZA', 'TUR', 'THD', 'EPOL', 'NORW',
    'ARGT', 'GXG', 'EPU', 'FM', 'FRN', 'VNM', 'KSA', 'QAT', 'EGPT', 'TAN',
    'ICLN', 'PBW', 'PBD', 'SUSA', 'KLD', 'DGT', 'SDG', 'NXTG', 'SNSR', 'MILN',
    'EFV', 'IVE', 'VLUE', 'IVW', 'MGK', 'MGV', 'SPYG', 'SPYD', 'SCHX', 'SCHA',
    'BRK-A'
]

data = yf.download(raw_tickers, start='2000-01-01', end='2024-01-01', auto_adjust=True)

# 儲存起始和終止日期
dates = []

for ticker in raw_tickers:
    ticker_data = data['Close'][ticker].dropna()
    start_date = ticker_data.index[0]
    end_date = ticker_data.index[-1]
    dates.append((ticker, start_date, end_date))

# 按起始日期排序
sorted_dates = sorted(dates, key=lambda x: x[1])

# 過濾出End Date = 2022-12-30 且 Start Date < 2005-01-01 的股票
tickers = [raw_tickers for raw_tickers, start_date, end_date in sorted_dates if end_date == pd.Timestamp('2023-12-29') and start_date < pd.Timestamp('2003-01-01')]


# 下載包含股息的每月數據
data = yf.download(tickers, start='2003-01-01', end='2024-01-01', interval='1mo', auto_adjust=True)

# 刪除缺失值
data = data['Close'].dropna()

# 計算每月回報率
monthly_returns = data.pct_change().dropna()

# 計算年化回報率
years = (data.index[-1] - data.index[0]).days / 365.25
mu = ((data.iloc[-1] / data.iloc[0]) ** (1 / years) - 1)

# 計算年化波動率
Sigma = monthly_returns.cov() * 12

# 只考慮成分股，不包括BRK-A
mu_stocks = mu.loc[tickers[:-1]]
Sigma_stocks = Sigma.loc[tickers[:-1], tickers[:-1]]

# BRK-A的年化波動率
brk_a_volatility = np.sqrt(Sigma.loc['BRK-A', 'BRK-A'])

# 定義投資組合優化問題
w = cp.Variable(len(tickers) - 1)
portfolio_return = mu_stocks.values @ w
portfolio_volatility = cp.quad_form(w, Sigma_stocks)

# 目標函數：最大化回報率，同時限制波動率
constraints = [cp.sum(w) == 1, w >= 0, portfolio_volatility <= ((brk_a_volatility-0.02)**2)]
problem = cp.Problem(cp.Maximize(portfolio_return), constraints)

# 嘗試使用不同的求解器
try:
    problem.solve(solver=cp.SCS)
    print("Solved with SCS")
except cp.error.SolverError:
    try:
        problem.solve(solver=cp.OSQP)
        print("Solved with OSQP")
    except cp.error.SolverError:
        try:
            problem.solve(solver=cp.ECOS)
            print("Solved with ECOS")
        except cp.error.SolverError as e:
            print("All solvers failed.")
            raise e

# 獲得最優權重
optimal_weights = w.value
print(f'Optimized Weights: {optimal_weights}')
print(f'Sum of weights: {sum(optimal_weights)}')

# 計算投資組合的預期回報率、波動率和Sharpe比率
portfolio_return_value = portfolio_return.value
portfolio_volatility_value = np.sqrt(portfolio_volatility.value)
sharpe_ratio = portfolio_return_value / portfolio_volatility_value

print(f"Expected annual return: {portfolio_return_value:.2f}")
print(f"Annual volatility: {portfolio_volatility_value:.2f}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# 計算投資組合的beta
brk_a_returns = monthly_returns['BRK-A']
portfolio_returns = monthly_returns[tickers[:-1]].dot(optimal_weights)

X = sm.add_constant(brk_a_returns)
model = sm.OLS(portfolio_returns, X).fit()
portfolio_beta = model.params[1]

print(f"Portfolio Beta: {portfolio_beta:.2f}")

# 與BRK-A對比
brk_a_return = mu['BRK-A']
print(f"BRK-A Return: {brk_a_return:.2f}")
print(f"BRK-A Volatility: {brk_a_volatility:.2f}")

# 選出占比超過1%的股票並打印
selected_stocks = []
for i, weight in enumerate(optimal_weights):
    if weight > 0.01:
        ticker = tickers[i]
        selected_stocks.append((ticker, weight, mu[ticker], np.sqrt(Sigma[ticker][ticker])))

# 打印結果
print("\nSelected stocks with weight > 1%:")
for stock in selected_stocks:
    print(f"Ticker: {stock[0]}, Weight: {stock[1]:.4f}, Annual Return: {stock[2]:.4f}, Annual Volatility: {stock[3]:.4f}")
