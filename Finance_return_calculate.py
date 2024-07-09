import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf

stock_name = [
    'BRK-A', 'SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'AGG', 'VNQ', 'VTI', 'IVV', 'BND',
    'VOO', 'VEU', 'VIG', 'LQD', 'TIP', 'HYG', 'SHY', 'IEFA', 'IEMG', 'ITOT',
    'IUSB', 'BNDX', 'SCHB', 'SCHD', 'SCHF', 'SCHZ', 'IYR', 'XLE', 'XLF', 'XLK',
    'XLV', 'XLY', 'XLP', 'XLU', 'XLI', 'XLB', 'XLC', 'XBI', 'XOP', 'ICLN',
    'ARKK', 'VYM', 'VTV', 'VB', 'VBR', 'VO', 'VOT', 'VUG', 'VEA', 'BSV'
]
results = pd.DataFrame(columns=['Stock', 'Initial Price', 'Final Price', 'Annual Return', 'Annual Volatility', 'Beta'])

market_ticker = 'SPY'
market_hist = yf.Ticker(market_ticker).history(start="2010-01-01", end="2023-01-01")
market_returns = market_hist['Close'].pct_change().dropna()

for stock in stock_name:
    # load stock price
    sp500 = yf.Ticker(stock)
    hist = sp500.history(start="2010-01-01", end="2023-01-01")
    
    # 获取股息数据
    dividends = hist['Dividends'].fillna(0)

    # consistent with market data date
    combinde_data = pd.DataFrame({'Market':market_hist['Close'], 'Stock': hist['Close'], 'Dividends': dividends})
    combinde_data.dropna(inplace=True)
    
    # Calculate return including dividends
    monthly_close_prices = combinde_data['Stock'].resample('M').last()
    monthly_dividends = combinde_data['Dividends'].resample('M').sum()
    monthly_returns = (monthly_close_prices + monthly_dividends).pct_change().dropna()
    market_monthly_close_price = combinde_data['Market'].resample('M').last()
    market_monthly_returns = market_monthly_close_price.pct_change().dropna()

    annual_returns = (1 + monthly_returns) ** 12 - 1
    n_years = (monthly_close_prices.index[-1] - monthly_close_prices.index[0]).days / 365.25
    expected_return = (monthly_close_prices[-1] / monthly_close_prices[0]) ** (1 / n_years) - 1

    SD = monthly_returns.std() * (12 ** 0.5)

    print(f"{stock} average return: {expected_return}")
    print(f"{stock} risk: {SD}")

    corariance_matrix = np.cov(monthly_returns, market_monthly_returns)
    beta = corariance_matrix[0, 1] / corariance_matrix[1, 1]

    new_row = pd.DataFrame({'Stock': [stock], 'Initial Price': [monthly_close_prices[0]], 'Final Price': [monthly_close_prices[-1]], 
                            'Annual Return': [expected_return], 'Annual Volatility': [SD], 'Length': [n_years], 'Beta': [beta]})
    results = pd.concat([results, new_row], ignore_index=True)


    plt.plot(annual_returns)
    plt.axhline(y=expected_return, color='r', linestyle='--', label='Expected Return')
    plt.title(f'{stock} Daily Returns Over Time')
    plt.xlabel('Date')
    plt.ylabel('Daily Returns')
    #plt.show()

print(results)

# OLS regression to test CAPM
# import statsmodels.api as sm
# X = results.loc[:,'Beta']
# X = sm.add_constant(X)
# Y = results.loc[:,'Annual Return']
# model = sm.OLS(Y, X).fit()

# print(model.summary())
