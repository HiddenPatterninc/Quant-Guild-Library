# Top 19 Notebooks for a Strategy / Trading Engine

Code cells only (no outputs), in notebook order. Source paths are given for each.

**Requirements:** `pip install numpy pandas scipy plotly matplotlib scikit-learn statsmodels yfinance ipywidgets qfin openpyxl ipython`. No Interactive Brokers account is needed. Lectures 47, 88, 100, 113, 119 and 138 originally read CSVs produced by IB scripts (or missing from the repo), so their sections now begin with a `yfinance` data-setup block that writes the same files.

## Index

1. **How to Trade with the Black-Scholes Model (Lecture 6)** — Trade options against Black-Scholes model prices and simulate the equity curve.
2. **Delta Hedging and Black-Scholes Prices (Lecture 9)** — Delta-hedge a short call under GBM; Monte Carlo P/L converges to the BS price.
3. **Martingale Volatility Trading (Lecture 7)** — Martingale betting and Ornstein-Uhlenbeck volatility mean reversion.
4. **How to Trade Implied Volatility Crush (Lecture 43)** — Model and trade the post-event implied-volatility crush.
5. **Trading Mean Reversion with Kalman Filters (Lecture 95)** — Mean-reversion strategy using Kalman filters.
6. **How to Trade the Covered Call (Lecture 106)** — Covered call payoff and risk analysis.
7. **How to Trade the Cash Secured Puts (Lecture 113)** — Cash-secured put / volatility risk premium analysis. The original IB data download is replaced with yfinance (SPY prices + VIX as the implied-vol proxy).
8. **Volatility Risk Premium How Options Markets Price Fear (Lecture 119)** — Volatility risk premium: realized vs implied vol.
9. **How to Protect your Stock Portfolio Against a Market Crash (Lecture 138)** — Hedging a stock portfolio against a crash.
10. **Master Volatility with ARCH & GARCH Models (Lecture 47)** — ARCH/GARCH volatility fitting and forecasting.
11. **Kalman Filter for Quant Finance (Lecture 92)** — Kalman filter for state estimation.
12. **Hidden Markov Models for Quant Finance (Lecture 51)** — Hidden Markov Models for market regimes.
13. **Black-Scholes Implied Volatility in 3 Minutes (Lecture 89)** — Solve Black-Scholes implied volatility numerically.
14. **Trading with the Black-Scholes Implied Volatility Surface (Lecture 30)** — Build and explore the implied volatility surface.
15. **How to Price Options with Monte Carlo Simulation (Lecture 73)** — Price options with Monte Carlo simulation.
16. **How to Trade with the Kelly Criterion (Lecture 36)** — Kelly criterion bet sizing.
17. **Analyzing Stock Returns with Principal Component Analysis in Python (Lecture 17)** — PCA of stock returns (SVD and covariance eigendecomposition) for risk factors and decorrelation.
18. **Black-Litterman vs. Mean-Variance Portfolio Optimization in Python (Lecture 100)** — Black-Litterman vs mean-variance optimization.
19. **How a Quant Manages a Portfolio (Lecture 88)** — Factor-model portfolio management.

---

## 1. How to Trade with the Black-Scholes Model (Lecture 6)

`2025 Video Lectures/6. How to Trade with the Black-Scholes Model/Black-ScholesTrading.ipynb` — Trade options against Black-Scholes model prices and simulate the equity curve.

```python
import numpy as np
from scipy.stats import norm

def black_scholes_call(S, K, sigma, r, t):
    d1 = (np.log(S/K) + (r + ((sigma**2)/2))*t) / (sigma * np.sqrt(t))
    d2 = d1 - (sigma * np.sqrt(t))
    C = S * norm.cdf(d1) - K * np.exp(-r*t) * norm.cdf(d2)
    return C

black_scholes_call(100, 100, .3, .05, 1)
```

```python
import matplotlib.pyplot as plt
import qfin as qf

# simulate dynamics of the underlying according to Geometric Brownian Motion
path = qf.simulations.GeometricBrownianMotion(100, 0.05, .3, 1/252, 1)


# create a chart of the price path and the strike price
plt.title("Terminal Value of an Option Contract")
plt.hlines(100, 0, 252, label='Strike', color='orange')
plt.plot(path.simulated_path, label='Price Path', color='white')
if max(path.simulated_path[-1] - 100, 0) == 0:
    plt.vlines(252, path.simulated_path[-1], 100, color='red', label="P/L")
else:
    plt.vlines(252, 100, path.simulated_path[-1], color='green', label="P/L")
plt.style.use('dark_background')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# print the premium and the resulting P/L
print("Premium at t=0:", black_scholes_call(100, 100, .3, .05, 1))
print("P/L:", max(path.simulated_path[-1] - 100, 0) - black_scholes_call(100, 100, .3, .05, 1))
```

```python
print("Call Price:", black_scholes_call(100, 100, .3, .05, 1))
print("Market Maker Quote:", "13.43 @ 14.10")
print("Trade Edge:", black_scholes_call(100, 100, .3, .05, 1) - 14.10)
```

```python
premium = 14.10 * 100
pls = []

for i in range(100000):
    path = qf.simulations.GeometricBrownianMotion(100, 0.05, .3, 1/252, 1)
    pls.append(max(path.simulated_path[-1] - 100, 0)*100 - premium)

np.mean(pls)
```

```python
premium = 14.10 * 100
pls = []

for i in range(100000):
    path = qf.simulations.GeometricBrownianMotion(100, -.5, .3, 1/252, 1)
    pls.append(max(path.simulated_path[-1] - 100, 0)*100 - premium)

plt.title("Trading this Edge Over Time")
plt.plot(np.cumsum(pls), label="Account Equity")
plt.style.use('dark_background')
plt.xlabel('Option Trade')
plt.ylabel('Portfolio Value')
plt.legend()
plt.show()
```


---

## 2. Delta Hedging and Black-Scholes Prices (Lecture 9)

`2025 Video Lectures/9. Delta Hedging and Black-Scholes Prices/Delta Hedging Trading Strategy.ipynb` — Delta-hedge a short call under GBM; Monte Carlo P/L converges to the BS price.

```python
import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt

# Black-Scholes pricing function for a European Call option
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
    delta = si.norm.cdf(d1)  # Delta of the call option
    return call_price, delta
```

```python
import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interactive

# Function to update the plot
def update(K=100, T=1, r=0.05, sigma=0.2):
    S_range = np.linspace(50, 150, 100)
    prices = []
    deltas = []
    
    for s in S_range:
        price, delta = black_scholes_call(s, K, T, r, sigma)
        prices.append(price)
        deltas.append(delta)
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    ax1.plot(S_range, prices, 'b-', label='Option Price')
    ax2.plot(S_range, deltas, 'r-', label='Delta')
    
    ax1.set_xlabel('Stock Price (S)')
    ax1.set_ylabel('Option Price', color='b')
    ax2.set_ylabel('Delta', color='r')
    
    ax1.tick_params(axis='y', colors='b')
    ax2.tick_params(axis='y', colors='r')
    
    plt.title('Black-Scholes Option Pricing & Delta')
    plt.show()

# Create interactive sliders
interactive_plot = interactive(update, 
                               K=widgets.FloatSlider(min=50, max=150, step=1, value=100, description='K'),
                               T=widgets.FloatSlider(min=0.1, max=5, step=0.1, value=1, description='T'),
                               r=widgets.FloatSlider(min=0, max=0.2, step=0.01, value=0.05, description='r'),
                               sigma=widgets.FloatSlider(min=0.05, max=1, step=0.05, value=0.2, description='sigma'))

interactive_plot
```

```python
def simulate_gbm(S0, mu, sigma, T, dt=1/252):
    """Simulates a Geometric Brownian Motion (GBM) price path."""
    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps)
    W = np.random.standard_normal(size=n_steps)
    W = np.cumsum(W) * np.sqrt(dt)  # Wiener process
    S_t = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)
    return S_t, t
```

```python
# Example Parameters
S0, K, T, r, sigma, mu = 100, 100, 1, 0.05, 0.2, 0.08

# Run single delta hedge simulation
S_t, t = simulate_gbm(S0, mu, sigma, T)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(t, S_t, label="Stock Price (GBM)")
plt.title("Stock Price Path")
plt.xlabel("Time (Years)")
plt.ylabel("Stock Price")
plt.legend()
plt.show()
```

```python
def delta_hedge_simulation(S0, K, T, r, sigma, mu, dt=1/252):
    """Simulates a delta hedging strategy for a short call position."""
    # Generate GBM path
    S_t, t = simulate_gbm(S0, mu, sigma, T, dt)
    
    # Initial Black-Scholes price
    call_price, delta = black_scholes_call(S0, K, T, r, sigma)
    
    # Short the call, buy delta shares
    initial_cash = call_price - delta * S0  # Initial cash position
    portfolio_value = initial_cash
    deltas = [delta]
    
    # Iterate over time steps to rebalance hedge
    for i in range(1, len(S_t)):
        tau = T - i * dt  # Time remaining
        call_price, delta = black_scholes_call(S_t[i], K, tau, r, sigma)
        deltas.append(delta)
        
        # Portfolio value at each step
        portfolio_value = delta * S_t[i] + initial_cash * np.exp(r * i * dt)

    # Final profit/loss
    final_pnl = portfolio_value - max(S_t[-1] - K, 0)  # Payoff of the short call
    return S_t, final_pnl, deltas, t
```

```python
# Example Parameters
S0, K, T, r, sigma, mu = 100, 100, 1, 0.05, 0.2, 0.08

ex = black_scholes_call(S0, K, T, r, sigma)

print('Option Price | ', ex[0], ' | Initial Delta | ', ex[1])

# Run single delta hedge simulation
S_t, final_pnl, deltas, t = delta_hedge_simulation(S0, K, T, r, sigma, mu)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(t, S_t, label="Stock Price (GBM)")
plt.axhline(K, color='r', linestyle="--", label="Strike Price")
plt.title("Stock Price Path and Delta Hedging")
plt.xlabel("Time (Years)")
plt.ylabel("Stock Price")
plt.legend()
plt.show()

print(f"Final P&L from Delta Hedging: {final_pnl:.2f}")
```

```python
n_simulations = 100
pnl_results = []

for _ in range(n_simulations):
    _, final_pnl, _, _ = delta_hedge_simulation(S0, K, T, r, sigma, mu)
    pnl_results.append(final_pnl)

# Plot histogram of outcomes
plt.figure(figsize=(10, 5))
plt.hist(pnl_results, bins=10, alpha=0.75, color="blue", edgecolor="black")
plt.axvline(np.mean(pnl_results), color='r', linestyle="--", label=f"Mean P/L: {np.mean(pnl_results):.2f}")
plt.title("Distribution of P&L from (Daily) Delta Hedging")
plt.xlabel("Final P&L")
plt.ylabel("Frequency")
plt.legend()
plt.show()

print(f"Average P&L over {n_simulations} simulations: {np.mean(pnl_results):.2f}")
```

```python
def delta_hedge_simulation(S0, K, T, r, sigma, mu, dt=1/1000):
    """Simulates a delta hedging strategy for a short call position."""
    # Generate GBM path
    S_t, t = simulate_gbm(S0, mu, sigma, T, dt)
    
    # Initial Black-Scholes price
    call_price, delta = black_scholes_call(S0, K, T, r, sigma)
    
    # Short the call, buy delta shares
    initial_cash = call_price - delta * S0  # Initial cash position
    portfolio_value = initial_cash
    deltas = [delta]
    
    # Iterate over time steps to rebalance hedge
    for i in range(1, len(S_t)):
        tau = T - i * dt  # Time remaining
        call_price, delta = black_scholes_call(S_t[i], K, tau, r, sigma)
        deltas.append(delta)
        
        # Portfolio value at each step
        portfolio_value = delta * S_t[i] + initial_cash * np.exp(r * i * dt)

    # Final profit/loss
    final_pnl = portfolio_value - max(S_t[-1] - K, 0)  # Payoff of the short call
    return S_t, final_pnl, deltas, t
```

```python
n_simulations = 100
pnl_results = []

for _ in range(n_simulations):
    _, final_pnl, _, _ = delta_hedge_simulation(S0, K, T, r, sigma, mu)
    pnl_results.append(final_pnl)

# Plot histogram of outcomes
plt.figure(figsize=(10, 5))
plt.hist(pnl_results, bins=10, alpha=0.75, color="blue", edgecolor="black")
plt.axvline(np.mean(pnl_results), color='r', linestyle="--", label=f"Mean P/L: {np.mean(pnl_results):.2f}")
plt.title("Distribution of P&L from (Closer to Continuous) Delta Hedging")
plt.xlabel("Final P&L")
plt.ylabel("Frequency")
plt.legend()
plt.show()

print(f"Average P&L over {n_simulations} simulations: {np.mean(pnl_results):.2f}")
```

```python
def delta_hedge_simulation(S0, K, T, r, sigma, mu, dt=1/500, vol_pos=None):
    """Simulates a delta hedging strategy for a short call position."""
    # Initial Black-Scholes price capturing market opinion
    call_price, delta = black_scholes_call(S0, K, T, r, sigma)
    
    if vol_pos != None:
        sigma = vol_pos
    
    # Generate GBM path according to our vol opinion
    S_t, t = simulate_gbm(S0, mu, sigma, T, dt)
    
    
    
    # Short the call, buy delta shares
    initial_cash = call_price - delta * S0  # Initial cash position
    portfolio_value = initial_cash
    deltas = [delta]
    
    # Iterate over time steps to rebalance hedge
    for i in range(1, len(S_t)):
        tau = T - i * dt  # Time remaining
        call_price, delta = black_scholes_call(S_t[i], K, tau, r, sigma)
        deltas.append(delta)
        
        # Portfolio value at each step
        portfolio_value = delta * S_t[i] + initial_cash * np.exp(r * i * dt)

    # Final profit/loss
    final_pnl = portfolio_value - max(S_t[-1] - K, 0)  # Payoff of the short call
    return S_t, final_pnl, deltas, t
```

```python
n_simulations = 100
pnl_results = []

for _ in range(n_simulations):
    _, final_pnl, _, _ = delta_hedge_simulation(S0, K, T, r, sigma, mu, vol_pos=.1)
    pnl_results.append(final_pnl)

# Plot histogram of outcomes
plt.figure(figsize=(10, 5))
plt.hist(pnl_results, bins=10, alpha=0.75, color="blue", edgecolor="black")
plt.axvline(np.mean(pnl_results), color='r', linestyle="--", label=f"Mean P/L: {np.mean(pnl_results):.2f}")
plt.title("Distribution of P&L from (Bi-Daily) Delta Hedging")
plt.xlabel("Final P&L")
plt.ylabel("Frequency")
plt.legend()
plt.show()

print(f"Average P&L over {n_simulations} simulations: {np.mean(pnl_results):.2f}")
```


---

## 3. Martingale Volatility Trading (Lecture 7)

`2025 Video Lectures/7. Martingale Volatility Trading/Martingale Volatility Trading.ipynb` — Martingale betting and Ornstein-Uhlenbeck volatility mean reversion.

```python
import random

def roulette_martingale(initial_bet, max_rounds, bankroll):
    current_bet = initial_bet
    total_bet = 0
    rounds = 0

    while rounds < max_rounds and bankroll > 0:
        total_bet += current_bet
        outcome = random.choice(["win", "lose"])
        if outcome == "win":
            bankroll += current_bet
            print(f"Round {rounds + 1}: Win! Bankroll = ${bankroll}")
            break
        else:
            bankroll -= current_bet
            current_bet *= 2
            print(f"Round {rounds + 1}: Loss. Bankroll = ${bankroll}")
        rounds += 1

    if bankroll <= 0:
        print("Bankrupt! The Martingale strategy failed.")
    else:
        print(f"Strategy succeeded in {rounds + 1} rounds.")
    
    return bankroll

# Parameters
initial_bet = 10
max_rounds = 10
bankroll = 1000

roulette_martingale(initial_bet, max_rounds, bankroll)
```

```python
import matplotlib.pyplot as plt
import numpy as np

def roulette_martingale_simulation_with_bets(initial_bet, bankroll, max_rounds, num_games):
    cumulative_wealth = []
    bets_over_time = []
    total_wealth = 0  # Tracks accumulated wealth across games

    for _ in range(num_games):
        current_bet = initial_bet
        game_bankroll = bankroll
        total_wagered = 0
        game_result = None

        for round_number in range(max_rounds):
            total_wagered += current_bet
            outcome = np.random.choice(["win", "lose"])

            if outcome == "win":
                game_bankroll += current_bet
                total_wealth += current_bet  # Profit is the initial bet amount
                game_result = "win"
                break
            else:
                game_bankroll -= current_bet
                current_bet *= 2

            # If bankroll is depleted, stop the game
            if game_bankroll <= 0:
                game_result = "lose"
                break

        # If the game ends in a loss, subtract the total wagered
        if game_result == "lose":
            total_wealth -= total_wagered

        # Append the cumulative wealth and the last bet for the game
        cumulative_wealth.append(total_wealth)
        bets_over_time.append(current_bet)

    return cumulative_wealth, bets_over_time

# Parameters
initial_bet = 10
bankroll = 1000
max_rounds = 10
num_games = 1000

# Run the simulation
cumulative_wealth, bets_over_time = roulette_martingale_simulation_with_bets(
    initial_bet, bankroll, max_rounds, num_games
)

# Plot the cumulative equity curve
plt.figure(figsize=(12, 6))
plt.plot(cumulative_wealth, label="Cumulative Wealth", color="green")
plt.title("Cumulative Equity Curve: Martingale Strategy in Roulette")
plt.xlabel("Game Number")
plt.ylabel("Cumulative Wealth ($)")
plt.axhline(0, color="red", linestyle="--", label="Break-Even Line")
plt.legend()
plt.grid()
plt.show()

# Plot the bets over time
plt.figure(figsize=(12, 6))
plt.plot(bets_over_time, label="Bet Size Over Time", color="blue")
plt.title("Bet Size Over Time: Martingale Strategy")
plt.xlabel("Game Number")
plt.ylabel("Bet Size ($)")
plt.legend()
plt.grid()
plt.show()
```

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_ornstein_uhlenbeck(mu, theta, sigma, x0, T, dt):
    n = int(T / dt)
    times = np.linspace(0, T, n)
    x = np.zeros(n)
    x[0] = x0

    for t in range(1, n):
        dx = theta * (mu - x[t-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
        x[t] = x[t-1] + dx

    return times, x

# Parameters
mu = 30       # Mean reversion level
theta = 0.8   # Mean reversion speed
sigma = 5     # Volatility
x0 = 30       # Initial value
T = 3       # Time in years
dt = 0.01     # Time step

times, ou_process = simulate_ornstein_uhlenbeck(mu, theta, sigma, x0, T, dt)

# Plot the simulation
plt.figure(figsize=(12, 6))
plt.plot(times, ou_process, label="Ornstein-Uhlenbeck Process")
plt.axhline(mu, color="red", linestyle="--", label="Mean Reversion Level")
plt.title("Simulated Ornstein-Uhlenbeck Process")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()
```

```python
import yfinance as yf
import pandas as pd

# Fetch VIX data
vix_data = yf.download("^VIX", start="2010-01-01", end="2023-01-01")
vix_data["Mean"] = vix_data["Close"].mean()

# Plot VIX data with mean
plt.figure(figsize=(12, 6))
plt.plot(vix_data["Close"], label="VIX Closing Prices")
plt.axhline(float(np.squeeze(vix_data["Close"].mean())), color="red", linestyle="--", label="Mean Level")
plt.title("VIX Historical Data with Mean Reversion")
plt.xlabel("Date")
plt.ylabel("VIX Closing Price")
plt.legend()
plt.show()
```

```python
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

# Download VIX data from Yahoo Finance
data = yf.download("^VIX", start="2010-01-01", end="2025-01-01")
prices = data['Close'].dropna().values

# Martingale strategy simulation
initial_capital = 100000  # Initial capital
position = 0
capital = initial_capital
max_exposure = 0
profit_loss = 0

equity_curve = []
positions = []

for i in range(1, len(prices)):
    previous_price = prices[i - 1][0]
    current_price = prices[i][0]

    # Calculate P&L for the current position
    profit_loss = position * (previous_price - current_price)
    capital += profit_loss

    if profit_loss >= 0:  # Close profitable position and restart
        position = -100  # Start with a short position of 100 contracts
    else:  # Double down to cover losses
        position *= 2

    # Update maximum exposure
    max_exposure = max(max_exposure, abs(position * current_price))

    equity_curve.append(capital)
    positions.append(position)

# Plot the equity curve
plt.figure(figsize=(12, 6))
plt.plot(equity_curve, label='Equity Curve')
plt.axhline(initial_capital, color='r', linestyle='--', label='Initial Capital')
plt.title('Martingale Strategy Equity Curve')
plt.xlabel('Weeks')
plt.ylabel('Equity ($)')
plt.legend()
plt.show()

# Results
print("Final Capital:", capital)
print("Max Exposure:", max_exposure)
```


---

## 4. How to Trade Implied Volatility Crush (Lecture 43)

`2025 Video Lectures/43. How to Trade Implied Volatility Crush/how_to_trade_volatility_crush_option_strategy.ipynb` — Model and trade the post-event implied-volatility crush.

```python
%%html
<style>
/* Overwrite the hard-coded white background for ipywidgets */
.cell-output-ipywidget-background {
    background-color: transparent !important;
}
/* Set widget foreground text and color to match the VS Code dark theme */
:root {
    --jp-widgets-color: var(--vscode-editor-foreground);
    --jp-widgets-font-size: var(--vscode-editor-font-size);
}
</style>
```

```python
import numpy as np
import plotly.graph_objects as go
import scipy.stats as stats
from scipy.stats import norm

# Black-Scholes formula for option pricing
def black_scholes(S, K, T, r, sigma, option_type):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == "call":
        price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:  # put
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    return price

# Fixed parameters
S = 100  # Stock price
K = 100  # Strike price
T = 30/365  # Time to expiry (30 days)
r = 0.05  # Risk-free rate

# Different implied volatilities
implied_vols = [0.2, 0.3, 0.4, 0.5, 0.6]  # 20% to 60% volatility

# Calculate straddle prices (call + put) for each volatility
straddle_prices = []
for vol in implied_vols:
    call = S * 0.1 * vol  # Simplified pricing for demonstration
    put = S * 0.1 * vol   # Simplified pricing for demonstration
    straddle_prices.append(call + put)

# Create bar chart
fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=[f"{vol*100}% Vol" for vol in implied_vols],
        y=straddle_prices,
        marker_color='rgba(0, 191, 255, 0.6)',
        name='Straddle Price'
    )
)

# Update layout
fig.update_layout(
    title='Straddle Option Prices at Different Implied Volatilities',
    xaxis_title='Implied Volatility',
    yaxis_title='Straddle Price ($)',
    width=800,
    height=500,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=False
)

# Update axes
fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.show()
```

```python
# Fixed parameters
S = 100  # Initial stock price
K = 100  # Strike price
T = 30/365  # Time to expiry (30 days)
r = 0.05  # Risk-free rate

# Different implied volatilities and corresponding spot prices
implied_vols = [0.2, 0.3, 0.4, 0.5, 0.6]  # 20% to 60% volatility
spot_prices = [S - (i * 15) for i in range(len(implied_vols))]  # Decrease spot by 10 each time

# Calculate straddle prices (call + put) for each volatility and spot price
straddle_prices = []
for vol, spot in zip(implied_vols, spot_prices):
    call = spot * 0.1 * vol  # Simplified pricing for demonstration
    put = spot * 0.1 * vol   # Simplified pricing for demonstration
    straddle_prices.append(call + put)

# Create bar chart
fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=[f"{vol*100}% Vol (S={spot})" for vol, spot in zip(implied_vols, spot_prices)],
        y=straddle_prices,
        marker_color='rgba(0, 191, 255, 0.6)',
        name='Straddle Price'
    )
)

# Update layout
fig.update_layout(
    title='Straddle Option Prices at Different Implied Volatilities and Spot Prices',
    xaxis_title='Implied Volatility (Spot Price)',
    yaxis_title='Straddle Price ($)',
    width=800,
    height=500,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=False
)

# Update axes
fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.show()
```


---

## 5. Trading Mean Reversion with Kalman Filters (Lecture 95)

`2026 Video Lectures/95. Trading Mean Reversion with Kalman Filters/ktsmr.ipynb` — Mean-reversion strategy using Kalman filters.

```python
%%html
<style>
/* Overwrite the hard-coded white background for ipywidgets */
.cell-output-ipywidget-background {
    background-color: transparent !important;
}
/* Set widget foreground text and color to match the VS Code dark theme */
:root {
    --jp-widgets-color: var(--vscode-editor-foreground);
    --jp-widgets-font-size: var(--vscode-editor-font-size);
}
</style>
```

```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# --- SIMULATION PARAMETERS ---
# ==========================================
n_samples = 30     # LOW SAMPLE SIZE: Forces a bad estimate of the mean
n_steps = 400      # EXTENDED HORIZON: Shows the full bleed-out effect
total_steps = n_samples + n_steps

# OU Process Parameters
true_mu = 100.0    # True long-run mean
theta = 0.05       # Mean reversion speed
sigma = 1.2        # Volatility
dt = 1.0

# Calculate theoretical stationary standard deviation
stat_dev = sigma / np.sqrt(2 * theta)

np.random.seed(42) # Yields an initial sample mean significantly above 100

# ==========================================
# --- PHASE 1 & 2: DATA GENERATION ---
# ==========================================
X = np.zeros(total_steps)
X[0] = 106.0  # Start artificially high to skew the small-sample estimate

# Generate the full Ornstein-Uhlenbeck path
for t in range(1, total_steps):
    dW = np.random.randn() * np.sqrt(dt)
    X[t] = X[t-1] + theta * (true_mu - X[t-1]) * dt + sigma * dW

# Phase 1: Estimation
phase1_data = X[:n_samples]
rolling_means = np.cumsum(phase1_data) / np.arange(1, n_samples + 1)
final_estimated_mu = rolling_means[-1]

# Define Trading Bands (1 Stationary Standard Deviation)
band_width = stat_dev * 0.8
upper_band = final_estimated_mu + band_width
lower_band = final_estimated_mu - band_width

# Generate Theoretical Distribution for the Order Book / Profiler
all_prices = np.round(np.arange(true_mu - 4*stat_dev, true_mu + 6*stat_dev, 0.5), 1)
all_vols = np.exp(-0.5 * ((all_prices - true_mu) / stat_dev)**2)

base_colors = []
for p in all_prices:
    if p < true_mu:
        base_colors.append('rgba(0, 150, 0, 0.6)')  # True Undervalued
    elif p > true_mu:
        base_colors.append('rgba(150, 0, 0, 0.6)')  # True Overvalued
    else:
        base_colors.append('rgba(150, 150, 150, 0.6)')

# ==========================================
# --- PHASE 2: TRADING SIMULATION ---
# ==========================================
positions = np.zeros(n_steps)
profits = np.zeros(n_steps)
current_pos = 0  # 1 for Long, -1 for Short, 0 for Flat

trade_markers_x = []
trade_markers_y = []
trade_markers_color = []

for i in range(n_steps):
    t = n_samples + i
    price = X[t]
    prev_price = X[t-1]
    
    # Calculate Mark-to-Market Profit from the previous position
    if i > 0:
        profits[i] = current_pos * (price - prev_price)
    
    # Trading Logic
    if current_pos == 0:
        if price > upper_band:
            current_pos = -1  # Enter Short
            trade_markers_x.append(i)
            trade_markers_y.append(price)
            trade_markers_color.append('red')
        elif price < lower_band:
            current_pos = 1   # Enter Long
            trade_markers_x.append(i)
            trade_markers_y.append(price)
            trade_markers_color.append('green')
    elif current_pos == 1:
        if price >= final_estimated_mu:
            current_pos = 0   # Close Long
            trade_markers_x.append(i)
            trade_markers_y.append(price)
            trade_markers_color.append('gray')
    elif current_pos == -1:
        if price <= final_estimated_mu:
            current_pos = 0   # Close Short
            trade_markers_x.append(i)
            trade_markers_y.append(price)
            trade_markers_color.append('gray')
            
    positions[i] = current_pos

cum_profits = np.cumsum(profits)

# ==========================================
# --- ANIMATION FRAMES ---
# ==========================================
frames = []

for k in range(total_frames := total_steps):
    if k <= n_samples:
        t1_x = np.arange(k)
        t1_y = X[:k]
        roll_y = rolling_means[:k]
        current_est = rolling_means[k-1] if k > 0 else X[0]
        
        # Keep Phase 2 plots empty during Phase 1
        t2_x, t2_y = [0], [current_est]
        eq_x, eq_y = [0], [0]
        mark_x, mark_y, mark_c = [0], [current_est], ['rgba(0,0,0,0)']
        
        plot_upper, plot_lower = current_est, current_est
        frame_name = f"est{k}"
    else:
        t_trad = k - n_samples
        t1_x = np.arange(n_samples)
        t1_y = phase1_data
        roll_y = rolling_means
        current_est = final_estimated_mu
        
        t2_x = np.arange(t_trad)
        t2_y = X[n_samples : k]
        eq_x = np.arange(t_trad)
        eq_y = cum_profits[:t_trad]
        
        valid_marks = [idx for idx, mx in enumerate(trade_markers_x) if mx < t_trad]
        mark_x = [trade_markers_x[idx] for idx in valid_marks] if valid_marks else [0]
        mark_y = [trade_markers_y[idx] for idx in valid_marks] if valid_marks else [current_est]
        mark_c = [trade_markers_color[idx] for idx in valid_marks] if valid_marks else ['rgba(0,0,0,0)']
        
        plot_upper, plot_lower = upper_band, lower_band
        frame_name = f"trade{t_trad}"

    current_price = X[k-1] if k > 0 else X[0]

    # --- Trace Definitions ---
    # Phase 1
    tr0 = go.Scatter(x=t1_x, y=t1_y, mode='lines', line=dict(color='#00ffff', width=2))
    tr1 = go.Scatter(x=t1_x, y=roll_y, mode='lines', line=dict(color='yellow', width=3))
    
    # PDF
    tr2 = go.Bar(x=all_prices, y=all_vols, marker=dict(color=base_colors))
    tr3 = go.Scatter(x=[true_mu, true_mu], y=[0, max(all_vols)], mode='lines', line=dict(color='rgba(255,255,255,0.4)', width=2, dash='dash'))
    tr4 = go.Scatter(x=[current_est, current_est], y=[0, max(all_vols)], mode='lines', line=dict(color='yellow', width=3, dash='dash'))
    tr5 = go.Scatter(x=[plot_upper, plot_upper], y=[0, max(all_vols)], mode='lines', line=dict(color='red', width=2))
    tr6 = go.Scatter(x=[plot_lower, plot_lower], y=[0, max(all_vols)], mode='lines', line=dict(color='green', width=2))
    tr7 = go.Scatter(x=[current_price], y=[max(all_vols)/2], mode='markers', marker=dict(color='white', size=12, symbol='diamond'))
    
    # Phase 2
    tr8 = go.Scatter(x=t2_x, y=t2_y, mode='lines', line=dict(color='#00ffff', width=2))
    tr9 = go.Scatter(x=[0, n_steps], y=[true_mu, true_mu], mode='lines', line=dict(color='rgba(255,255,255,0.4)', width=2, dash='dash'))
    tr10 = go.Scatter(x=[0, n_steps], y=[current_est, current_est], mode='lines', line=dict(color='yellow', width=2, dash='dash'))
    tr11 = go.Scatter(x=[0, n_steps], y=[plot_upper, plot_upper], mode='lines', line=dict(color='red', width=1, dash='dot'))
    tr12 = go.Scatter(x=[0, n_steps], y=[plot_lower, plot_lower], mode='lines', line=dict(color='green', width=1, dash='dot'))
    tr13 = go.Scatter(x=mark_x, y=mark_y, mode='markers', marker=dict(color=mark_c, size=10))
    
    # Equity
    tr14 = go.Scatter(x=eq_x, y=eq_y, mode='lines', line=dict(color='magenta', width=3, shape='hv'))

    frames.append(go.Frame(data=[tr0, tr1, tr2, tr3, tr4, tr5, tr6, tr7, tr8, tr9, tr10, tr11, tr12, tr13, tr14], name=frame_name))

# ==========================================
# --- FIGURE INITIALIZATION ---
# ==========================================
fig = make_subplots(
    rows=2, cols=2, 
    subplot_titles=[
        "Phase 1: 30-Point Sample & Estimated Mean", "Phase 2: Mean Reversion Trading", 
        "True OU PDF vs Estimated Trading Bands", "Phase 2: Equity Curve (PnL)"
    ],
    horizontal_spacing=0.10, vertical_spacing=0.15
)

# Phase 1
fig.add_trace(go.Scatter(x=[0], y=[0], line=dict(color='#00ffff')), row=1, col=1)
fig.add_trace(go.Scatter(x=[0], y=[0], line=dict(color='yellow')), row=1, col=1)

# PDF
fig.add_trace(go.Bar(x=all_prices, y=all_vols, marker=dict(color=base_colors)), row=2, col=1)
fig.add_trace(go.Scatter(x=[true_mu, true_mu], y=[0, 1], line=dict(color='rgba(255,255,255,0.4)', dash='dash')), row=2, col=1)
fig.add_trace(go.Scatter(x=[0], y=[0], line=dict(color='yellow', dash='dash')), row=2, col=1)
fig.add_trace(go.Scatter(x=[0], y=[0], line=dict(color='red')), row=2, col=1)
fig.add_trace(go.Scatter(x=[0], y=[0], line=dict(color='green')), row=2, col=1)
fig.add_trace(go.Scatter(x=[0], y=[0], marker=dict(color='white')), row=2, col=1)

# Phase 2 (Note the newly added lines)
fig.add_trace(go.Scatter(x=[0], y=[0], line=dict(color='#00ffff')), row=1, col=2)
fig.add_trace(go.Scatter(x=[0, n_steps], y=[true_mu, true_mu], line=dict(color='rgba(255,255,255,0.4)', dash='dash'), name="True Mean"), row=1, col=2)
fig.add_trace(go.Scatter(x=[0, n_steps], y=[0, 0], line=dict(color='yellow', dash='dash'), name="Est Mean"), row=1, col=2)
fig.add_trace(go.Scatter(x=[0, n_steps], y=[0, 0], line=dict(color='red', dash='dot'), name="Upper Band"), row=1, col=2)
fig.add_trace(go.Scatter(x=[0, n_steps], y=[0, 0], line=dict(color='green', dash='dot'), name="Lower Band"), row=1, col=2)
fig.add_trace(go.Scatter(x=[0], y=[0], marker=dict(color='rgba(0,0,0,0)')), row=1, col=2)

# Equity
fig.add_trace(go.Scatter(x=[0], y=[0], line=dict(color='magenta', shape='hv')), row=2, col=2)

fig.frames = frames

# ==========================================
# --- SLIDER & LAYOUT ---
# ==========================================
sliders = [dict(
    active=0, currentvalue={"prefix": "Step: "}, pad={"t": 0},
    x=0.15, len=0.85, y=-0.1,
    steps=[
        dict(
            method="animate",
            args=[[frames[k].name], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
            label=f"Est {k}" if k <= n_samples else f"Trade {k - n_samples}"
        ) for k in range(total_frames)
    ]
)]

fig.update_layout(
    height=750, width=1100,
    title_text="Mean Reversion Trap: Long-Term Bleed from Structural Bias",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'), showlegend=False, sliders=sliders, margin=dict(b=100),
    updatemenus=[{
        'type': 'buttons', 'x': 0.0, 'y': -0.15, 'xanchor': 'left', 'yanchor': 'top', 'direction': 'left', 'showactive': False,
        'buttons': [{'label': '▶ Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 20, 'redraw': True}, 'fromcurrent': True}]}]
    }]
)

# Axes Styling
fig.update_xaxes(title_text='Sample Step', range=[0, n_samples], row=1, col=1, showgrid=True, gridcolor='rgba(128,128,128,0.3)')
fig.update_yaxes(title_text='Price', range=[92, 110], row=1, col=1, showgrid=True, gridcolor='rgba(128,128,128,0.3)')

fig.update_xaxes(title_text='Price', range=[92, 110], row=2, col=1, showgrid=False)
fig.update_yaxes(title_text='PDF Mass', range=[0, max(all_vols) * 1.1], row=2, col=1, showgrid=False)

fig.update_xaxes(title_text='Trade Step', range=[0, n_steps], row=1, col=2, showgrid=True, gridcolor='rgba(128,128,128,0.3)')
fig.update_yaxes(title_text='Price', range=[92, 110], row=1, col=2, showgrid=True, gridcolor='rgba(128,128,128,0.3)')

min_pnl, max_pnl = min(-10, np.min(cum_profits) * 1.1), max(5, np.max(cum_profits) * 1.1)
fig.update_xaxes(title_text='Trade Step', range=[0, n_steps], row=2, col=2, showgrid=True, gridcolor='rgba(128,128,128,0.3)')
fig.update_yaxes(title_text='Cumulative PnL', range=[min_pnl, max_pnl], row=2, col=2, showgrid=True, gridcolor='rgba(128,128,128,0.3)')

fig.show()
```


---

## 6. How to Trade the Covered Call (Lecture 106)

`2026 Video Lectures/106. How to Trade the Covered Call/httcc.ipynb` — Covered call payoff and risk analysis.

```python
%%html
<style>
/* Overwrite the hard-coded white background for ipywidgets */
.cell-output-ipywidget-background {
    background-color: transparent !important;
}
/* Set widget foreground text and color to match the VS Code dark theme */
:root {
    --jp-widgets-color: var(--vscode-editor-foreground);
    --jp-widgets-font-size: var(--vscode-editor-font-size);
}
</style>
```

```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# --- 1. DATA GENERATION & MATH SETUP ---
# ==========================================
K = 100  # Strike Price
S = np.linspace(50, 150, 500)  # Range of Underlying Asset Prices

# Payoff Functions
call_payoff = np.maximum(S - K, 0)
put_payoff = np.maximum(K - S, 0)

# Initial states for the sliders
initial_S_T_call = 125  # ITM Call
initial_S_T_put = 75    # ITM Put

initial_call_p = max(initial_S_T_call - K, 0)
initial_put_p = max(K - initial_S_T_put, 0)

# Determine initial line colors
init_call_color = "#00ff00" if initial_S_T_call > K else "#ff0000"
init_put_color = "#00ff00" if initial_S_T_put < K else "#ff0000"

# ==========================================
# --- 2. FIGURE & TRACE INITIALIZATION ---
# ==========================================
fig = make_subplots(
    rows=1, cols=2, 
    horizontal_spacing=0.10,
    subplot_titles=[
        f"<b>Call Option Payoff: ${initial_call_p}</b>", 
        f"<b>Put Option Payoff: ${initial_put_p}</b>"
    ]
)

# Left Plot: Call Option Payoff (Cyan)
fig.add_trace(go.Scatter(x=S, y=call_payoff, mode='lines', 
                         line=dict(color='#00ffff', width=3)), row=1, col=1)

# Right Plot: Put Option Payoff (Magenta)
fig.add_trace(go.Scatter(x=S, y=put_payoff, mode='lines', 
                         line=dict(color='#ff00ff', width=3)), row=1, col=2)

# ==========================================
# --- 3. SHAPES (STATIC & DYNAMIC LINES) ---
# ==========================================
# Shape 0: Strike Price Line (Call)
fig.add_shape(type="line", x0=K, y0=0, x1=K, y1=50, 
              line=dict(color="white", width=2, dash="dot"), row=1, col=1)
# Shape 1: Strike Price Line (Put)
fig.add_shape(type="line", x0=K, y0=0, x1=K, y1=50, 
              line=dict(color="white", width=2, dash="dot"), row=1, col=2)

# Shape 2: Underlying at Expiration (Call) - Dynamic
fig.add_shape(type="line", x0=initial_S_T_call, y0=0, x1=initial_S_T_call, y1=50, 
              line=dict(color=init_call_color, width=2, dash="dash"), row=1, col=1)
# Shape 3: Underlying at Expiration (Put) - Dynamic
fig.add_shape(type="line", x0=initial_S_T_put, y0=0, x1=initial_S_T_put, y1=50, 
              line=dict(color=init_put_color, width=2, dash="dash"), row=1, col=2)


# ==========================================
# --- 4. SLIDER LOGIC ---
# ==========================================
call_steps = []
put_steps = []

for st in range(50, 151):
    # Call Logic (ITM if S_T > K)
    call_p = max(st - K, 0)
    call_color = "#00ff00" if st > K else "#ff0000"
    call_steps.append(dict(
        method="relayout",
        args=[{
            "shapes[2].x0": st,
            "shapes[2].x1": st,
            "shapes[2].line.color": call_color,
            "annotations[0].text": f"<b>Call Option Payoff: ${call_p}</b>"
        }],
        label=str(st)
    ))

    # Put Logic (ITM if S_T < K)
    put_p = max(K - st, 0)
    put_color = "#00ff00" if st < K else "#ff0000"
    put_steps.append(dict(
        method="relayout",
        args=[{
            "shapes[3].x0": st,
            "shapes[3].x1": st,
            "shapes[3].line.color": put_color,
            "annotations[1].text": f"<b>Put Option Payoff: ${put_p}</b>"
        }],
        label=str(st)
    ))

sliders = [
    # Call Slider (Left Side)
    dict(
        active=initial_S_T_call - 50,
        currentvalue={"prefix": "Call Underlying ($): ", "font": {"color": "white"}},
        pad={"t": 30},
        x=0.0, y=-0.15, len=0.45, xanchor="left", yanchor="top",
        steps=call_steps,
        font=dict(color="white")
    ),
    # Put Slider (Right Side)
    dict(
        active=initial_S_T_put - 50,
        currentvalue={"prefix": "Put Underlying ($): ", "font": {"color": "white"}},
        pad={"t": 30},
        x=0.55, y=-0.15, len=0.45, xanchor="left", yanchor="top",
        steps=put_steps,
        font=dict(color="white")
    )
]

# ==========================================
# --- 5. LAYOUT & ANNOTATIONS ---
# ==========================================
fig.update_layout(
    height=650, width=1100,
    margin=dict(b=150), # Add bottom margin to fit sliders
    template="plotly_dark",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    showlegend=False,
    sliders=sliders
)

# Update Axes
fig.update_xaxes(title_text='Underlying Price', range=[50, 150], row=1, col=1)
fig.update_yaxes(title_text='Payoff ($)', range=[-5, 55], row=1, col=1)

fig.update_xaxes(title_text='Underlying Price', range=[50, 150], row=1, col=2)
fig.update_yaxes(title_text='Payoff ($)', range=[-5, 55], row=1, col=2)

# Strike Annotations
fig.add_annotation(text="<b>Strike (K)</b><br>100", row=1, col=1, 
                   x=K-2, y=45, showarrow=False, xanchor="right",
                   font=dict(color="white"), bgcolor="rgba(0,0,0,0.6)", bordercolor="white", borderwidth=1)

fig.add_annotation(text="<b>Strike (K)</b><br>100", row=1, col=2, 
                   x=K+2, y=45, showarrow=False, xanchor="left",
                   font=dict(color="white"), bgcolor="rgba(0,0,0,0.6)", bordercolor="white", borderwidth=1)

fig.show()
```

```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# --- 1. DATA GENERATION & MATH SETUP ---
# ==========================================
K = 100  # Strike Price
S = np.linspace(50, 150, 500)  # Range of Underlying Asset Prices

# Payoff Functions
long_call_payoff = np.maximum(S - K, 0)
long_put_payoff = np.maximum(K - S, 0)
short_call_payoff = -long_call_payoff
short_put_payoff = -long_put_payoff

# Initial states for the sliders
initial_S_T_call = 125  
initial_S_T_put = 75    

# Initial Payoffs
init_lc_p = max(initial_S_T_call - K, 0)
init_lp_p = max(K - initial_S_T_put, 0)
init_sc_p = -init_lc_p
init_sp_p = -init_lp_p

# Determine initial line colors (Green if profitable/neutral, Red if losing)
init_lc_color = "#00ff00" if initial_S_T_call > K else "#ff0000"
init_lp_color = "#00ff00" if initial_S_T_put < K else "#ff0000"
init_sc_color = "#ff0000" if initial_S_T_call > K else "#00ff00" 
init_sp_color = "#ff0000" if initial_S_T_put < K else "#00ff00"

# ==========================================
# --- 2. FIGURE & TRACE INITIALIZATION ---
# ==========================================
fig = make_subplots(
    rows=2, cols=2, 
    horizontal_spacing=0.10, vertical_spacing=0.15,
    subplot_titles=[
        f"<b>Long Call Payoff: ${init_lc_p}</b>",   # annotations[0]
        f"<b>Long Put Payoff: ${init_lp_p}</b>",    # annotations[1]
        f"<b>Short Call Payoff: ${init_sc_p}</b>",  # annotations[2]
        f"<b>Short Put Payoff: ${init_sp_p}</b>"    # annotations[3]
    ]
)

# Top Left: Long Call (Cyan)
fig.add_trace(go.Scatter(x=S, y=long_call_payoff, mode='lines', 
                         line=dict(color='#00ffff', width=3)), row=1, col=1)

# Top Right: Long Put (Magenta)
fig.add_trace(go.Scatter(x=S, y=long_put_payoff, mode='lines', 
                         line=dict(color='#ff00ff', width=3)), row=1, col=2)

# Bottom Left: Short Call (Cyan)
fig.add_trace(go.Scatter(x=S, y=short_call_payoff, mode='lines', 
                         line=dict(color='#00ffff', width=3)), row=2, col=1)

# Bottom Right: Short Put (Magenta)
fig.add_trace(go.Scatter(x=S, y=short_put_payoff, mode='lines', 
                         line=dict(color='#ff00ff', width=3)), row=2, col=2)

# ==========================================
# --- 3. SHAPES (STATIC & DYNAMIC LINES) ---
# ==========================================
# Static Strike Lines (Shapes 0-3)
fig.add_shape(type="line", x0=K, y0=0, x1=K, y1=50, line=dict(color="white", width=2, dash="dot"), row=1, col=1)
fig.add_shape(type="line", x0=K, y0=0, x1=K, y1=50, line=dict(color="white", width=2, dash="dot"), row=1, col=2)
fig.add_shape(type="line", x0=K, y0=0, x1=K, y1=-50, line=dict(color="white", width=2, dash="dot"), row=2, col=1)
fig.add_shape(type="line", x0=K, y0=0, x1=K, y1=-50, line=dict(color="white", width=2, dash="dot"), row=2, col=2)

# Dynamic Underlying Lines (Shapes 4-7)
fig.add_shape(type="line", x0=initial_S_T_call, y0=0, x1=initial_S_T_call, y1=50, line=dict(color=init_lc_color, width=2, dash="dash"), row=1, col=1) # 4
fig.add_shape(type="line", x0=initial_S_T_put, y0=0, x1=initial_S_T_put, y1=50, line=dict(color=init_lp_color, width=2, dash="dash"), row=1, col=2)   # 5
fig.add_shape(type="line", x0=initial_S_T_call, y0=0, x1=initial_S_T_call, y1=-50, line=dict(color=init_sc_color, width=2, dash="dash"), row=2, col=1) # 6
fig.add_shape(type="line", x0=initial_S_T_put, y0=0, x1=initial_S_T_put, y1=-50, line=dict(color=init_sp_color, width=2, dash="dash"), row=2, col=2)   # 7

# ==========================================
# --- 4. SLIDER LOGIC ---
# ==========================================
call_steps = []
put_steps = []

for st in range(50, 151):
    # --- Call Logic ---
    lc_p = max(st - K, 0)
    sc_p = -lc_p
    lc_color = "#00ff00" if st > K else "#ff0000"
    sc_color = "#ff0000" if st > K else "#00ff00"
    
    call_steps.append(dict(
        method="relayout",
        args=[{
            "shapes[4].x0": st, "shapes[4].x1": st, "shapes[4].line.color": lc_color, # Long Call shape
            "shapes[6].x0": st, "shapes[6].x1": st, "shapes[6].line.color": sc_color, # Short Call shape
            "annotations[0].text": f"<b>Long Call Payoff: ${lc_p}</b>",
            "annotations[2].text": f"<b>Short Call Payoff: ${sc_p}</b>"
        }],
        label=str(st)
    ))

    # --- Put Logic ---
    lp_p = max(K - st, 0)
    sp_p = -lp_p
    lp_color = "#00ff00" if st < K else "#ff0000"
    sp_color = "#ff0000" if st < K else "#00ff00"
    
    put_steps.append(dict(
        method="relayout",
        args=[{
            "shapes[5].x0": st, "shapes[5].x1": st, "shapes[5].line.color": lp_color, # Long Put shape
            "shapes[7].x0": st, "shapes[7].x1": st, "shapes[7].line.color": sp_color, # Short Put shape
            "annotations[1].text": f"<b>Long Put Payoff: ${lp_p}</b>",
            "annotations[3].text": f"<b>Short Put Payoff: ${sp_p}</b>"
        }],
        label=str(st)
    ))

sliders = [
    # Call Slider
    dict(
        active=initial_S_T_call - 50,
        currentvalue={"prefix": "Call Underlying ($): ", "font": {"color": "white"}},
        pad={"t": 30},
        x=0.0, y=-0.12, len=0.45, xanchor="left", yanchor="top",
        steps=call_steps, font=dict(color="white")
    ),
    # Put Slider
    dict(
        active=initial_S_T_put - 50,
        currentvalue={"prefix": "Put Underlying ($): ", "font": {"color": "white"}},
        pad={"t": 30},
        x=0.55, y=-0.12, len=0.45, xanchor="left", yanchor="top",
        steps=put_steps, font=dict(color="white")
    )
]

# ==========================================
# --- 5. LAYOUT & ANNOTATIONS ---
# ==========================================
fig.update_layout(
    height=850, width=1100,
    margin=dict(t=80, b=150),
    template="plotly_dark",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    showlegend=False,
    sliders=sliders
)

# Update Axes Ranges and Titles
for r in [1, 2]:
    for c in [1, 2]:
        fig.update_xaxes(title_text='Underlying Price', range=[50, 150], row=r, col=c)
        if r == 1:
            fig.update_yaxes(title_text='Payoff ($)', range=[-5, 55], row=r, col=c)
        else:
            fig.update_yaxes(title_text='Payoff ($)', range=[-55, 5], row=r, col=c)

# Strike Annotations
annotation_style = dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0.6)", bordercolor="white", borderwidth=1, showarrow=False)

fig.add_annotation(text="<b>Strike (K)</b><br>100", row=1, col=1, x=K-2, y=45, xanchor="right", **annotation_style)
fig.add_annotation(text="<b>Strike (K)</b><br>100", row=1, col=2, x=K+2, y=45, xanchor="left", **annotation_style)
fig.add_annotation(text="<b>Strike (K)</b><br>100", row=2, col=1, x=K-2, y=-45, xanchor="right", **annotation_style)
fig.add_annotation(text="<b>Strike (K)</b><br>100", row=2, col=2, x=K+2, y=-45, xanchor="left", **annotation_style)

fig.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. Parameters & Data Generation ---
S0 = 100        # Initial price
K = 105         # Strike price
T = 1.0         # Time to maturity (1 year)
r = 0.05        # Risk-free rate
vol = 0.25      # Volatility
n_steps = 100   # Number of simulation steps
dt = T / n_steps

def call_payoff(S, K):
    return np.maximum(S - K, 0)

# Generate a Price Path
np.random.seed(np.random.randint(0, 1000))
rets = np.random.normal((r - 0.5 * vol**2) * dt, vol * np.sqrt(dt), n_steps)
path = S0 * np.exp(np.cumsum(rets))
path = np.insert(path, 0, S0)
dates = pd.date_range(start='2025-01-01', periods=n_steps+1, freq='B')

S_range = np.linspace(60, 160, 200)
payoff_vals = call_payoff(S_range, K)

# Colors
off_white = "#e0e0e0"
strike_color = "#ff4b4b"
payoff_color = "#00ff88"
path_color = "#00d1ff"

# --- 2. Figure Setup ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("European Call Payoff Diagram", "Underlying Price Simulation"),
    horizontal_spacing=0.12
)

# Base traces
fig.add_trace(go.Scatter(x=S_range, y=payoff_vals, line=dict(color=payoff_color, width=3)), row=1, col=1)
fig.add_trace(go.Scatter(x=[K, K], y=[0, 60], line=dict(color=strike_color, width=1, dash='dash')), row=1, col=1)
fig.add_trace(go.Scatter(x=[S0, S0], y=[0, 60], line=dict(color=off_white, width=2), mode='lines'), row=1, col=1)
fig.add_trace(go.Scatter(x=[dates[0], dates[-1]], y=[K, K], line=dict(color=strike_color, width=1, dash='dash')), row=1, col=2)
fig.add_trace(go.Scatter(x=[dates[0]], y=[S0], line=dict(color=path_color, width=2.5), mode='lines'), row=1, col=2)

# --- 3. Animation Frames & Slider Steps ---
frames = []
slider_steps = []

for i in range(n_steps + 1):
    current_price = path[i]
    frame_name = f"f{i}"
    
    # Create Frame
    frames.append(go.Frame(
        data=[
            go.Scatter(x=S_range, y=payoff_vals),
            go.Scatter(x=[K, K], y=[0, 60]),
            go.Scatter(x=[current_price, current_price], y=[0, 60]),
            go.Scatter(x=[dates[0], dates[-1]], y=[K, K]),
            go.Scatter(x=dates[:i+1], y=path[:i+1])
        ],
        name=frame_name
    ))
    
    # Create Slider Step
    step = {
        "args": [
            [frame_name],
            {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "fromcurrent": True}
        ],
        "label": str(i),
        "method": "animate"
    }
    slider_steps.append(step)

fig.frames = frames

# --- 4. Layout with Buttons and Slider ---
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=650, width=1200,
    margin=dict(t=100, b=150),
    showlegend=False,
    updatemenus=[{
        "type": "buttons",
        "buttons": [
            {
                "label": "▶ Play",
                "method": "animate",
                "args": [None, {"frame": {"duration": 20, "redraw": False}, "fromcurrent": True}]
            },
            {
                "label": "⏸ Pause",
                "method": "animate",
                "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "fromcurrent": True}]
            }
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "showactive": False,
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top"
    }],
    sliders=[{
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 14, "color": off_white},
            "prefix": "Step: ",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 0},
        "pad": {"b": 10, "t": 50},
        "len": 0.85,
        "x": 0.15,
        "y": 0,
        "steps": slider_steps
    }]
)

axis_style = dict(
    showgrid=True, gridcolor='rgba(255,255,255,0.1)',
    tickfont=dict(color=off_white), linecolor=off_white,
    zeroline=False, title_font=dict(color=off_white)
)

fig.update_xaxes(axis_style, range=[60, 160], title_text="Asset Price", row=1, col=1)
fig.update_yaxes(axis_style, range=[-2, 60], title_text="Payoff ($)", row=1, col=1)
fig.update_xaxes(axis_style, range=[dates[0], dates[-1]], title_text="Date", row=1, col=2)
fig.update_yaxes(axis_style, range=[60, 160], title_text="Asset Price ($)", row=1, col=2)

fig.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# --- 1. Parameters & Data Generation ---
S0 = 100        # Initial price
K = 105         # Strike price
T = 1.0         # Time to maturity (1 year)
r = 0.05        # Risk-free rate
vol = 0.25      # Volatility
n_steps = 100   # Number of simulation steps
dt = T / n_steps

def call_payoff(S, K):
    return np.maximum(S - K, 0)

# Black-Scholes European Call Price
def bs_call_price(S, K, tau, r, vol):
    if tau <= 1e-5:
        return call_payoff(S, K)
    d1 = (np.log(S / K) + (r + 0.5 * vol**2) * tau) / (vol * np.sqrt(tau))
    d2 = d1 - vol * np.sqrt(tau)
    return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)

# Generate a Price Path
np.random.seed(np.random.randint(0, 1000))
rets = np.random.normal((r - 0.5 * vol**2) * dt, vol * np.sqrt(dt), n_steps)
path = S0 * np.exp(np.cumsum(rets))
path = np.insert(path, 0, S0)
dates = pd.date_range(start='2025-01-01', periods=n_steps+1, freq='B')

S_range = np.linspace(60, 160, 200)
payoff_vals = call_payoff(S_range, K)

# Colors
off_white = "#e0e0e0"
strike_color = "#ff4b4b"
payoff_color = "#00ff88"
path_color = "#00d1ff"
bs_color = "#ff9900"  

# --- 2. Figure Setup ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("European Call Payoff & BS Price", "Underlying Price Simulation"),
    horizontal_spacing=0.12
)

bs_vals_initial = bs_call_price(S_range, K, T, r, vol)

# Base traces (Only Payoff and BS Price have showlegend=True)
fig.add_trace(go.Scatter(x=S_range, y=payoff_vals, line=dict(color=payoff_color, width=3), 
                         name='Intrinsic Payoff', showlegend=True), row=1, col=1)
fig.add_trace(go.Scatter(x=[K, K], y=[0, 60], line=dict(color=strike_color, width=1, dash='dash'), 
                         name='Strike', showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=[S0, S0], y=[0, 60], line=dict(color=off_white, width=2), mode='lines', 
                         name='Current S', showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=S_range, y=bs_vals_initial, line=dict(color=bs_color, width=2), 
                         name='BS Price', showlegend=True), row=1, col=1) 

fig.add_trace(go.Scatter(x=[dates[0], dates[-1]], y=[K, K], line=dict(color=strike_color, width=1, dash='dash'), 
                         showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=[dates[0]], y=[S0], line=dict(color=path_color, width=2.5), mode='lines', 
                         showlegend=False), row=1, col=2)

# --- 3. Animation Frames & Slider Steps ---
frames = []
slider_steps = []

for i in range(n_steps + 1):
    current_price = path[i]
    tau = max(T - i * dt, 0.0) 
    current_bs_vals = bs_call_price(S_range, K, tau, r, vol)
    
    frame_name = f"f{i}"
    
    frames.append(go.Frame(
        data=[
            go.Scatter(x=S_range, y=payoff_vals),                            
            go.Scatter(x=[K, K], y=[0, 60]),                                 
            go.Scatter(x=[current_price, current_price], y=[0, 60]),         
            go.Scatter(x=S_range, y=current_bs_vals),                        
            go.Scatter(x=[dates[0], dates[-1]], y=[K, K]),                   
            go.Scatter(x=dates[:i+1], y=path[:i+1])                          
        ],
        name=frame_name
    ))
    
    step = {
        "args": [
            [frame_name],
            {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "fromcurrent": True}
        ],
        "label": str(i),
        "method": "animate"
    }
    slider_steps.append(step)

fig.frames = frames

# --- 4. Layout with Legend, Buttons and Slider ---
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=650, width=1200,
    margin=dict(t=100, b=150),
    showlegend=True,
    legend=dict(
        yanchor="top", y=0.95,
        xanchor="left", x=0.05,
        bgcolor="rgba(0,0,0,0.6)",
        bordercolor=off_white,
        borderwidth=1,
        font=dict(color="white")
    ),
    updatemenus=[{
        "type": "buttons",
        "buttons": [
            {
                "label": "▶ Play",
                "method": "animate",
                "args": [None, {"frame": {"duration": 40, "redraw": False}, "fromcurrent": True}]
            },
            {
                "label": "⏸ Pause",
                "method": "animate",
                "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "fromcurrent": True}]
            }
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "showactive": False,
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top"
    }],
    sliders=[{
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 14, "color": off_white},
            "prefix": "Step: ",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 0},
        "pad": {"b": 10, "t": 50},
        "len": 0.85,
        "x": 0.15,
        "y": 0,
        "steps": slider_steps
    }]
)

axis_style = dict(
    showgrid=True, gridcolor='rgba(255,255,255,0.1)',
    tickfont=dict(color=off_white), linecolor=off_white,
    zeroline=False, title_font=dict(color=off_white)
)

fig.update_xaxes(axis_style, range=[60, 160], title_text="Asset Price", row=1, col=1)
fig.update_yaxes(axis_style, range=[-2, 60], title_text="Payoff / Option Price ($)", row=1, col=1)
fig.update_xaxes(axis_style, range=[dates[0], dates[-1]], title_text="Date", row=1, col=2)
fig.update_yaxes(axis_style, range=[60, 160], title_text="Asset Price ($)", row=1, col=2)

fig.show()
```

```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# --- 1. DATA GENERATION & MATH SETUP ---
# ==========================================
K = 100       # Strike Price of Short Call
S_basis = 90  # Cost basis of Long Underlying
S = np.linspace(50, 150, 500)  # Range of Underlying Asset Prices

# Payoff Functions
short_call_payoff = -np.maximum(S - K, 0)
long_stock_payoff = S - S_basis
combined_payoff = short_call_payoff + long_stock_payoff

# Initial state for the slider
initial_S_T = 105  

# Initial Payoffs
init_sc_p = -max(initial_S_T - K, 0)
init_ls_p = initial_S_T - S_basis
init_cb_p = init_sc_p + init_ls_p

# Determine initial line colors (Green if profitable/neutral, Red if losing)
init_sc_color = "#00ff00" if init_sc_p >= 0 else "#ff0000"
init_ls_color = "#00ff00" if init_ls_p >= 0 else "#ff0000"
init_cb_color = "#00ff00" if init_cb_p >= 0 else "#ff0000"

# ==========================================
# --- 2. FIGURE & TRACE INITIALIZATION ---
# ==========================================
fig = make_subplots(
    rows=2, cols=2, 
    specs=[[{"type": "xy"}, {"type": "xy"}],
           [{"colspan": 2, "type": "xy"}, None]], # Bottom plot spans both columns
    horizontal_spacing=0.10, vertical_spacing=0.15,
    subplot_titles=[
        f"<b>Short Call Payoff (K={K}): ${init_sc_p}</b>",     # annotations[0]
        f"<b>Long Stock Payoff (Basis={S_basis}): ${init_ls_p}</b>",  # annotations[1]
        f"<b>Combined Covered Call Payoff: ${init_cb_p}</b>"          # annotations[2]
    ]
)

# Top Left: Short Call (Cyan)
fig.add_trace(go.Scatter(x=S, y=short_call_payoff, mode='lines', 
                         line=dict(color='#00ffff', width=3)), row=1, col=1)

# Top Right: Long Stock (Magenta)
fig.add_trace(go.Scatter(x=S, y=long_stock_payoff, mode='lines', 
                         line=dict(color='#ff00ff', width=3)), row=1, col=2)

# Bottom: Combined Covered Call (Yellow)
fig.add_trace(go.Scatter(x=S, y=combined_payoff, mode='lines', 
                         line=dict(color='#ffff00', width=3)), row=2, col=1)

# ==========================================
# --- 3. SHAPES (STATIC & DYNAMIC LINES) ---
# ==========================================
# Static Reference Lines
# Shape 0: Strike Price Line (Top Left - Short Call)
fig.add_shape(type="line", x0=K, y0=0, x1=K, y1=-55, line=dict(color="white", width=2, dash="dot"), row=1, col=1)
# Shape 1: Cost Basis Line (Top Right - Long Stock)
fig.add_shape(type="line", x0=S_basis, y0=-45, x1=S_basis, y1=65, line=dict(color="white", width=2, dash="dot"), row=1, col=2)

# Shape 2: Strike Price Line (Bottom - Combined)
fig.add_shape(type="line", x0=K, y0=-45, x1=K, y1=15, line=dict(color="white", width=2, dash="dot"), row=2, col=1)
# Shape 3: Stock Cost Basis Line (Bottom - Combined)
fig.add_shape(type="line", x0=S_basis, y0=-45, x1=S_basis, y1=15, line=dict(color="white", width=2, dash="dot"), row=2, col=1)
# Shape 4: Horizontal Zero Line (Bottom - Combined)
fig.add_shape(type="line", x0=50, y0=0, x1=150, y1=0, line=dict(color="rgba(255,255,255,0.4)", width=2, dash="dash"), row=2, col=1)

# Dynamic Underlying Lines (Shapes 5-7)
# Shape 5: S_T (Top Left)
fig.add_shape(type="line", x0=initial_S_T, y0=0, x1=initial_S_T, y1=-55, line=dict(color=init_sc_color, width=2, dash="dash"), row=1, col=1) 
# Shape 6: S_T (Top Right)
fig.add_shape(type="line", x0=initial_S_T, y0=-45, x1=initial_S_T, y1=65, line=dict(color=init_ls_color, width=2, dash="dash"), row=1, col=2)   
# Shape 7: S_T (Bottom Combined)
fig.add_shape(type="line", x0=initial_S_T, y0=-45, x1=initial_S_T, y1=15, line=dict(color=init_cb_color, width=2, dash="dash"), row=2, col=1) 

# ==========================================
# --- 4. SLIDER LOGIC ---
# ==========================================
slider_steps = []

for st in range(50, 151):
    # Calculations
    sc_p = -max(st - K, 0)
    ls_p = st - S_basis
    cb_p = sc_p + ls_p
    
    # Color mapping
    sc_color = "#00ff00" if sc_p >= 0 else "#ff0000"
    ls_color = "#00ff00" if ls_p >= 0 else "#ff0000"
    cb_color = "#00ff00" if cb_p >= 0 else "#ff0000"
    
    slider_steps.append(dict(
        method="relayout",
        args=[{
            # Update Dynamic Line Positions and Colors (Shifted indices +2 due to new static shapes)
            "shapes[5].x0": st, "shapes[5].x1": st, "shapes[5].line.color": sc_color,
            "shapes[6].x0": st, "shapes[6].x1": st, "shapes[6].line.color": ls_color,
            "shapes[7].x0": st, "shapes[7].x1": st, "shapes[7].line.color": cb_color,
            
            # Update Titles
            "annotations[0].text": f"<b>Short Call Payoff (K={K}): ${sc_p}</b>",
            "annotations[1].text": f"<b>Long Stock Payoff (Basis={S_basis}): ${ls_p}</b>",
            "annotations[2].text": f"<b>Combined Covered Call Payoff: ${cb_p}</b>"
        }],
        label=str(st)
    ))

sliders = [dict(
    active=initial_S_T - 50,
    currentvalue={"prefix": "Underlying Price at Expiration ($): ", "font": {"color": "white"}},
    pad={"t": 30},
    x=0.1, y=-0.12, len=0.8, xanchor="left", yanchor="top",
    steps=slider_steps, font=dict(color="white")
)]

# ==========================================
# --- 5. LAYOUT & ANNOTATIONS ---
# ==========================================
fig.update_layout(
    height=800, width=1100,
    margin=dict(t=80, b=150),
    template="plotly_dark",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    showlegend=False,
    sliders=sliders
)

# Update Axes Ranges and Titles
fig.update_xaxes(title_text='Underlying Price', range=[50, 150])

# Custom Y-axis limits depending on the plot profile
fig.update_yaxes(title_text='Payoff ($)', range=[-55, 5], row=1, col=1)  # Short Call
fig.update_yaxes(title_text='Payoff ($)', range=[-45, 65], row=1, col=2) # Long Stock
fig.update_yaxes(title_text='Payoff ($)', range=[-45, 20], row=2, col=1) # Combined

# Informational Annotations
annotation_style = dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0.6)", bordercolor="white", borderwidth=1, showarrow=False)

fig.add_annotation(text="<b>Strike (K)</b><br>100", row=1, col=1, x=K-2, y=-45, xanchor="right", **annotation_style)
fig.add_annotation(text="<b>Basis</b><br>90", row=1, col=2, x=S_basis+2, y=-35, xanchor="left", **annotation_style)

# Bottom Chart Annotations
fig.add_annotation(text="<b>Strike (K)</b><br>100", row=2, col=1, x=K+2, y=-35, xanchor="left", **annotation_style)
fig.add_annotation(text="<b>Basis</b><br>90", row=2, col=1, x=S_basis-2, y=-35, xanchor="right", **annotation_style)
fig.add_annotation(text="<b>Max Profit Cap</b>", row=2, col=1, x=130, y=12, xanchor="center", font=dict(color="yellow"), showarrow=False)

fig.show()
```

```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# --- 1. DATA GENERATION & MATH SETUP ---
# ==========================================
K = 100       # Strike Price of Short Call
S_basis = 90  # Cost basis of Long Underlying
S = np.linspace(50, 150, 500)  # Range of Underlying Asset Prices

# Payoff Functions
short_call_payoff = -np.maximum(S - K, 0)
long_stock_payoff = S - S_basis
combined_payoff = short_call_payoff + long_stock_payoff

# Initial state for the slider
initial_S_T = 105  

# Initial Payoffs
init_sc_p = -max(initial_S_T - K, 0)
init_ls_p = initial_S_T - S_basis
init_cb_p = init_sc_p + init_ls_p

# Determine initial line colors (Green if profitable/neutral, Red if losing)
init_sc_color = "#00ff00" if init_sc_p >= 0 else "#ff0000"
init_ls_color = "#00ff00" if init_ls_p >= 0 else "#ff0000"
init_cb_color = "#00ff00" if init_cb_p >= 0 else "#ff0000"

# ==========================================
# --- 2. FIGURE & TRACE INITIALIZATION ---
# ==========================================
fig = make_subplots(
    rows=2, cols=2, 
    specs=[[{"type": "xy"}, {"type": "xy"}],
           [{"colspan": 2, "type": "xy"}, None]], # Bottom plot spans both columns
    horizontal_spacing=0.10, vertical_spacing=0.15,
    subplot_titles=[
        f"<b>Short Call Payoff (K={K}): ${init_sc_p}</b>",     # annotations[0]
        f"<b>Long Stock Payoff (Basis={S_basis}): ${init_ls_p}</b>",  # annotations[1]
        f"<b>Combined Covered Call Payoff: ${init_cb_p}</b>"          # annotations[2]
    ]
)

# Top Left: Short Call (Cyan)
fig.add_trace(go.Scatter(x=S, y=short_call_payoff, mode='lines', 
                         line=dict(color='#00ffff', width=3)), row=1, col=1)

# Top Right: Long Stock (Magenta)
fig.add_trace(go.Scatter(x=S, y=long_stock_payoff, mode='lines', 
                         line=dict(color='#ff00ff', width=3)), row=1, col=2)

# Bottom: Combined Covered Call (Yellow)
fig.add_trace(go.Scatter(x=S, y=combined_payoff, mode='lines', 
                         line=dict(color='#ffff00', width=3)), row=2, col=1)

# ==========================================
# --- 3. SHAPES (STATIC & DYNAMIC LINES) ---
# ==========================================
# Static Reference Lines
# Shape 0: Strike Price Line (Top Left - Short Call)
fig.add_shape(type="line", x0=K, y0=0, x1=K, y1=-55, line=dict(color="white", width=2, dash="dot"), row=1, col=1)
# Shape 1: Cost Basis Line (Top Right - Long Stock)
fig.add_shape(type="line", x0=S_basis, y0=-45, x1=S_basis, y1=65, line=dict(color="white", width=2, dash="dot"), row=1, col=2)

# Shape 2: Strike Price Line (Bottom - Combined)
fig.add_shape(type="line", x0=K, y0=-45, x1=K, y1=15, line=dict(color="white", width=2, dash="dot"), row=2, col=1)
# Shape 3: Stock Cost Basis Line (Bottom - Combined)
fig.add_shape(type="line", x0=S_basis, y0=-45, x1=S_basis, y1=15, line=dict(color="white", width=2, dash="dot"), row=2, col=1)
# Shape 4: Horizontal Zero Line (Bottom - Combined)
fig.add_shape(type="line", x0=50, y0=0, x1=150, y1=0, line=dict(color="rgba(255,255,255,0.4)", width=2, dash="dash"), row=2, col=1)

# Dynamic Underlying Lines (Shapes 5-7)
# Shape 5: S_T (Top Left)
fig.add_shape(type="line", x0=initial_S_T, y0=0, x1=initial_S_T, y1=-55, line=dict(color=init_sc_color, width=2, dash="dash"), row=1, col=1) 
# Shape 6: S_T (Top Right)
fig.add_shape(type="line", x0=initial_S_T, y0=-45, x1=initial_S_T, y1=65, line=dict(color=init_ls_color, width=2, dash="dash"), row=1, col=2)   
# Shape 7: S_T (Bottom Combined)
fig.add_shape(type="line", x0=initial_S_T, y0=-45, x1=initial_S_T, y1=15, line=dict(color=init_cb_color, width=2, dash="dash"), row=2, col=1) 

# ==========================================
# --- 4. SLIDER LOGIC ---
# ==========================================
slider_steps = []

for st in range(50, 151):
    # Calculations
    sc_p = -max(st - K, 0)
    ls_p = st - S_basis
    cb_p = sc_p + ls_p
    
    # Color mapping
    sc_color = "#00ff00" if sc_p >= 0 else "#ff0000"
    ls_color = "#00ff00" if ls_p >= 0 else "#ff0000"
    cb_color = "#00ff00" if cb_p >= 0 else "#ff0000"
    
    slider_steps.append(dict(
        method="relayout",
        args=[{
            # Update Dynamic Line Positions and Colors (Shifted indices +2 due to new static shapes)
            "shapes[5].x0": st, "shapes[5].x1": st, "shapes[5].line.color": sc_color,
            "shapes[6].x0": st, "shapes[6].x1": st, "shapes[6].line.color": ls_color,
            "shapes[7].x0": st, "shapes[7].x1": st, "shapes[7].line.color": cb_color,
            
            # Update Titles
            "annotations[0].text": f"<b>Short Call Payoff (K={K}): ${sc_p}</b>",
            "annotations[1].text": f"<b>Long Stock Payoff (Basis={S_basis}): ${ls_p}</b>",
            "annotations[2].text": f"<b>Combined Covered Call Payoff: ${cb_p}</b>"
        }],
        label=str(st)
    ))

sliders = [dict(
    active=initial_S_T - 50,
    currentvalue={"prefix": "Underlying Price at Expiration ($): ", "font": {"color": "white"}},
    pad={"t": 30},
    x=0.1, y=-0.12, len=0.8, xanchor="left", yanchor="top",
    steps=slider_steps, font=dict(color="white")
)]

# ==========================================
# --- 5. LAYOUT & ANNOTATIONS ---
# ==========================================
fig.update_layout(
    height=800, width=1100,
    margin=dict(t=80, b=150),
    template="plotly_dark",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    showlegend=False,
    sliders=sliders
)

# Update Axes Ranges and Titles
fig.update_xaxes(title_text='Underlying Price', range=[50, 150])

# Custom Y-axis limits depending on the plot profile
fig.update_yaxes(title_text='Payoff ($)', range=[-55, 5], row=1, col=1)  # Short Call
fig.update_yaxes(title_text='Payoff ($)', range=[-45, 65], row=1, col=2) # Long Stock
fig.update_yaxes(title_text='Payoff ($)', range=[-45, 20], row=2, col=1) # Combined

# Informational Annotations
annotation_style = dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0.6)", bordercolor="white", borderwidth=1, showarrow=False)

fig.add_annotation(text="<b>Strike (K)</b><br>100", row=1, col=1, x=K-2, y=-45, xanchor="right", **annotation_style)
fig.add_annotation(text="<b>Basis</b><br>90", row=1, col=2, x=S_basis+2, y=-35, xanchor="left", **annotation_style)

# Bottom Chart Annotations
fig.add_annotation(text="<b>Strike (K)</b><br>100", row=2, col=1, x=K+2, y=-35, xanchor="left", **annotation_style)
fig.add_annotation(text="<b>Basis</b><br>90", row=2, col=1, x=S_basis-2, y=-35, xanchor="right", **annotation_style)
fig.add_annotation(text="<b>Max Profit Cap</b>", row=2, col=1, x=130, y=12, xanchor="center", font=dict(color="yellow"), showarrow=False)

fig.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# --- 1. Parameters & Data Generation ---
S_basis = 90.0  # Cost basis of the stock
S0 = 95.0       # Initial spot price (when selling the call)
K = 100.0       # Strike price of the short call
T = 1.0         # Time to maturity (1 year)
r = 0.05        # Risk-free rate
vol = 0.45      # Volatility
n_steps = 100   # Number of simulation steps
dt = T / n_steps

def call_payoff(S, K):
    return np.maximum(S - K, 0)

# Black-Scholes European Call Price
def bs_call_price(S, K, tau, r, vol):
    if tau <= 1e-5:
        return call_payoff(S, K)
    d1 = (np.log(S / K) + (r + 0.5 * vol**2) * tau) / (vol * np.sqrt(tau))
    d2 = d1 - vol * np.sqrt(tau)
    return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)

# Calculate Premium Received at t=0
C0 = bs_call_price(S0, K, T, r, vol)

# Generate a Price Path
np.random.seed(np.random.randint(0, 1000))
rets = np.random.normal((r - 0.5 * vol**2) * dt, vol * np.sqrt(dt), n_steps)
path = S0 * np.exp(np.cumsum(rets))
path = np.insert(path, 0, S0)
dates = pd.date_range(start='2025-01-01', periods=n_steps+1, freq='B')

# Payoff ranges
S_range = np.linspace(40, 180, 200)

# Total P/L (Stock + Option Premium - Call Liability)
intrinsic_pl = (S_range - S_basis) + C0 - call_payoff(S_range, K)
bs_pl_initial = (S_range - S_basis) + C0 - bs_call_price(S_range, K, T, r, vol)

# Colors
off_white = "#e0e0e0"
strike_color = "#ff4b4b"
payoff_color = "#00ff88"
path_color = "#00d1ff"
bs_color = "#ff9900"  
missed_color = "#ffdd00"

# --- 2. Figure Setup ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Covered Call Total P/L & BS Price", "Underlying Price Simulation"),
    horizontal_spacing=0.12
)

# Base traces (Order matters for animation frames)
# 0: Total Intrinsic
fig.add_trace(go.Scatter(x=S_range, y=intrinsic_pl, line=dict(color=payoff_color, width=3), 
                         name='Total Intrinsic P/L', showlegend=True), row=1, col=1)
# 1: Strike Line
fig.add_trace(go.Scatter(x=[K, K], y=[-60, 80], line=dict(color=strike_color, width=1, dash='dash'), 
                         name='Strike', showlegend=False), row=1, col=1)
# 2: Current Underlying Price
fig.add_trace(go.Scatter(x=[S0, S0], y=[-60, 80], line=dict(color=off_white, width=2), mode='lines', 
                         name='Current S', showlegend=False), row=1, col=1)
# 3: BS P/L Curve
fig.add_trace(go.Scatter(x=S_range, y=bs_pl_initial, line=dict(color=bs_color, width=2), 
                         name='Current BS P/L', showlegend=True), row=1, col=1) 
# 4: Zero Line for P/L
fig.add_trace(go.Scatter(x=[40, 180], y=[0, 0], line=dict(color="rgba(255,255,255,0.3)", width=2, dash='dash'), 
                         showlegend=False), row=1, col=1)
# 5: Path Strike Line
fig.add_trace(go.Scatter(x=[dates[0], dates[-1]], y=[K, K], line=dict(color=strike_color, width=1, dash='dash'), 
                         showlegend=False), row=1, col=2)
# 6: Price Path
fig.add_trace(go.Scatter(x=[dates[0]], y=[S0], line=dict(color=path_color, width=2.5), mode='lines', 
                         showlegend=False), row=1, col=2)

# --- 3. Animation Frames & Slider Steps ---
frames = []
slider_steps = []

# Capture the base annotations created by make_subplots (the titles)
base_annotations = list(fig.layout.annotations)

# Calculate Final P/L for the Text Box
final_S = path[-1]
is_assigned = final_S > K

if is_assigned:
    stock_pl = K - S_basis
    option_pl = C0
    total_pl = stock_pl + option_pl
    missed_upside = final_S - K
    cf_text = (
        f"<b>Expiration: ASSIGNED</b><br><br>"
        f"<b><span style='color:{off_white}'>--- Stock Leg ---</span></b><br>"
        f"Bought at: ${S_basis:.2f}<br>"
        f"Sold at Strike: ${K:.2f}<br>"
        f"Stock P/L: {'+' if stock_pl >= 0 else '-'}${abs(stock_pl):.2f}<br><br>"
        f"<b><span style='color:{off_white}'>--- Option Leg ---</span></b><br>"
        f"Premium Recv: +${C0:.2f}<br>"
        f"Option P/L: +${option_pl:.2f}<br><br>"
        f"<b><span style='color:{missed_color}'>--- Opportunity Cost ---</span></b><br>"
        f"<span style='color:{missed_color}'>Missed Upside: -${missed_upside:.2f}</span><br>"
        f"--------------------------------<br>"
        f"<b><span style='color:{payoff_color}'>Total Realized P/L: {'+' if total_pl >= 0 else '-'}${abs(total_pl):.2f}</span></b>"
    )
else:
    stock_pl = final_S - S_basis
    option_pl = C0
    total_pl = stock_pl + option_pl
    cf_text = (
        f"<b>Expiration: UNASSIGNED</b><br><br>"
        f"<b><span style='color:{off_white}'>--- Stock Leg ---</span></b><br>"
        f"Cost Basis: ${S_basis:.2f}<br>"
        f"Final Value: ${final_S:.2f}<br>"
        f"Stock Unrealized P/L: {'+' if stock_pl >= 0 else '-'}${abs(stock_pl):.2f}<br><br>"
        f"<b><span style='color:{off_white}'>--- Option Leg ---</span></b><br>"
        f"Premium Recv: +${C0:.2f}<br>"
        f"Expires Worthless: $0.00<br>"
        f"Option Realized P/L: +${option_pl:.2f}<br>"
        f"--------------------------------<br>"
        f"<b><span style='color:{payoff_color}'>Total Net P/L: {'+' if total_pl >= 0 else '-'}${abs(total_pl):.2f}</span></b>"
    )

for i in range(n_steps + 1):
    current_price = path[i]
    tau = max(T - i * dt, 0.0) 
    current_bs_pl = (S_range - S_basis) + C0 - bs_call_price(S_range, K, tau, r, vol)
    
    frame_name = f"f{i}"
    frame_ann = list(base_annotations)
    
    # On the very last frame, append the cashflow breakdown box to the right chart
    if i == n_steps:
        frame_ann.append(dict(
            x=0.05, y=0.95, xref="x2 domain", yref="y2 domain",
            text=cf_text, showarrow=False, align="left",
            bgcolor="rgba(0,0,0,0.85)", bordercolor=payoff_color, borderwidth=1,
            font=dict(color="white", size=13)
        ))
    
    frames.append(go.Frame(
        data=[
            go.Scatter(x=S_range, y=intrinsic_pl),                             
            go.Scatter(x=[K, K], y=[-60, 80]),                                 
            go.Scatter(x=[current_price, current_price], y=[-60, 80]),         
            go.Scatter(x=S_range, y=current_bs_pl),                        
            go.Scatter(x=[40, 180], y=[0, 0]),                   
            go.Scatter(x=[dates[0], dates[-1]], y=[K, K]),                   
            go.Scatter(x=dates[:i+1], y=path[:i+1])                          
        ],
        layout=go.Layout(annotations=frame_ann),
        name=frame_name
    ))
    
    step = {
        "args": [
            [frame_name],
            {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "fromcurrent": True}
        ],
        "label": str(i),
        "method": "animate"
    }
    slider_steps.append(step)

fig.frames = frames

# --- 4. Layout with Legend, Buttons and Slider ---
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=650, width=1200,
    margin=dict(t=100, b=150),
    showlegend=True,
    legend=dict(
        yanchor="top", y=0.95,
        xanchor="left", x=0.02,
        bgcolor="rgba(0,0,0,0.6)",
        bordercolor=off_white,
        borderwidth=1,
        font=dict(color="white")
    ),
    updatemenus=[{
        "type": "buttons",
        "buttons": [
            {
                "label": "▶ Play",
                "method": "animate",
                "args": [None, {"frame": {"duration": 40, "redraw": False}, "fromcurrent": True}]
            },
            {
                "label": "⏸ Pause",
                "method": "animate",
                "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "fromcurrent": True}]
            }
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "showactive": False,
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top"
    }],
    sliders=[{
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 14, "color": off_white},
            "prefix": "Step: ",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 0},
        "pad": {"b": 10, "t": 50},
        "len": 0.85,
        "x": 0.15,
        "y": 0,
        "steps": slider_steps
    }]
)

axis_style = dict(
    showgrid=True, gridcolor='rgba(255,255,255,0.1)',
    tickfont=dict(color=off_white), linecolor=off_white,
    zeroline=False, title_font=dict(color=off_white)
)

fig.update_xaxes(axis_style, range=[40, 180], title_text="Asset Price", row=1, col=1)
fig.update_yaxes(axis_style, range=[-60, 80], title_text="Total P/L ($)", row=1, col=1)
fig.update_xaxes(axis_style, range=[dates[0], dates[-1]], title_text="Date", row=1, col=2)
fig.update_yaxes(axis_style, range=[40, 180], title_text="Asset Price ($)", row=1, col=2)

fig.show()
```


---

## 7. How to Trade the Cash Secured Puts (Lecture 113)

`2026 Video Lectures/113. How to Trade the Cash Secured Puts/httcc.ipynb` — Cash-secured put / volatility risk premium analysis. The original IB data download is replaced with yfinance (SPY prices + VIX as the implied-vol proxy).

```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# --- 1. DATA GENERATION & MATH SETUP ---
# ==========================================
K = 100
S = np.linspace(50, 150, 500)

# Payoff Functions
long_put_payoff = np.maximum(K - S, 0)
short_put_payoff = -np.maximum(K - S, 0)

# Initial states for sliders
initial_S_T_long_put = 75
initial_S_T_short_put = 75

initial_long_put_p = max(K - initial_S_T_long_put, 0)
initial_short_put_p = -max(K - initial_S_T_short_put, 0)

# Initial line colors
init_long_put_color = "#00ff00" if initial_S_T_long_put < K else "#ff0000"
init_short_put_color = "#ff0000" if initial_S_T_short_put < K else "#00ff00"

# ==========================================
# --- 2. FIGURE & TRACE INITIALIZATION ---
# ==========================================
fig = make_subplots(
    rows=1, cols=2,
    horizontal_spacing=0.10,
    subplot_titles=[
        f"<b>Long Put Payoff: ${initial_long_put_p}</b>",
        f"<b>Short Put Payoff: ${initial_short_put_p}</b>"
    ]
)

# Left Plot: Long Put
fig.add_trace(
    go.Scatter(
        x=S,
        y=long_put_payoff,
        mode="lines",
        line=dict(color="#ff00ff", width=3)
    ),
    row=1, col=1
)

# Right Plot: Short Put
fig.add_trace(
    go.Scatter(
        x=S,
        y=short_put_payoff,
        mode="lines",
        line=dict(color="#ffaa00", width=3)
    ),
    row=1, col=2
)

# ==========================================
# --- 3. SHAPES ---
# ==========================================
# Strike Price Line - Long Put
fig.add_shape(
    type="line",
    x0=K, y0=-50,
    x1=K, y1=50,
    line=dict(color="white", width=2, dash="dot"),
    row=1, col=1
)

# Strike Price Line - Short Put
fig.add_shape(
    type="line",
    x0=K, y0=-50,
    x1=K, y1=50,
    line=dict(color="white", width=2, dash="dot"),
    row=1, col=2
)

# Underlying at Expiration - Long Put
fig.add_shape(
    type="line",
    x0=initial_S_T_long_put, y0=-50,
    x1=initial_S_T_long_put, y1=50,
    line=dict(color=init_long_put_color, width=2, dash="dash"),
    row=1, col=1
)

# Underlying at Expiration - Short Put
fig.add_shape(
    type="line",
    x0=initial_S_T_short_put, y0=-50,
    x1=initial_S_T_short_put, y1=50,
    line=dict(color=init_short_put_color, width=2, dash="dash"),
    row=1, col=2
)

# ==========================================
# --- 4. SLIDER LOGIC ---
# ==========================================
long_put_steps = []
short_put_steps = []

for st in range(50, 151):
    # Long Put Logic
    long_put_p = max(K - st, 0)
    long_put_color = "#00ff00" if st < K else "#ff0000"

    long_put_steps.append(dict(
        method="relayout",
        args=[{
            "shapes[2].x0": st,
            "shapes[2].x1": st,
            "shapes[2].line.color": long_put_color,
            "annotations[0].text": f"<b>Long Put Payoff: ${long_put_p}</b>"
        }],
        label=str(st)
    ))

    # Short Put Logic
    short_put_p = -max(K - st, 0)
    short_put_color = "#ff0000" if st < K else "#00ff00"

    short_put_steps.append(dict(
        method="relayout",
        args=[{
            "shapes[3].x0": st,
            "shapes[3].x1": st,
            "shapes[3].line.color": short_put_color,
            "annotations[1].text": f"<b>Short Put Payoff: ${short_put_p}</b>"
        }],
        label=str(st)
    ))

sliders = [
    dict(
        active=initial_S_T_long_put - 50,
        currentvalue={
            "prefix": "Long Put Underlying ($): ",
            "font": {"color": "white"}
        },
        pad={"t": 30},
        x=0.0,
        y=-0.15,
        len=0.45,
        xanchor="left",
        yanchor="top",
        steps=long_put_steps,
        font=dict(color="white")
    ),
    dict(
        active=initial_S_T_short_put - 50,
        currentvalue={
            "prefix": "Short Put Underlying ($): ",
            "font": {"color": "white"}
        },
        pad={"t": 30},
        x=0.55,
        y=-0.15,
        len=0.45,
        xanchor="left",
        yanchor="top",
        steps=short_put_steps,
        font=dict(color="white")
    )
]

# ==========================================
# --- 5. LAYOUT & ANNOTATIONS ---
# ==========================================
fig.update_layout(
    height=650,
    width=1100,
    margin=dict(b=150),
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    showlegend=False,
    sliders=sliders
)

# Axes
fig.update_xaxes(title_text="Underlying Price", range=[50, 150], row=1, col=1)
fig.update_yaxes(title_text="Payoff ($)", range=[-55, 55], row=1, col=1)

fig.update_xaxes(title_text="Underlying Price", range=[50, 150], row=1, col=2)
fig.update_yaxes(title_text="Payoff ($)", range=[-55, 55], row=1, col=2)

# Strike Annotations
fig.add_annotation(
    text="<b>Strike (K)</b><br>100",
    row=1, col=1,
    x=K - 2,
    y=45,
    showarrow=False,
    xanchor="right",
    font=dict(color="white"),
    bgcolor="rgba(0,0,0,0.6)",
    bordercolor="white",
    borderwidth=1
)

fig.add_annotation(
    text="<b>Strike (K)</b><br>100",
    row=1, col=2,
    x=K + 2,
    y=45,
    showarrow=False,
    xanchor="left",
    font=dict(color="white"),
    bgcolor="rgba(0,0,0,0.6)",
    bordercolor="white",
    borderwidth=1
)

fig.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# --- 1. PARAMETERS ---
# ============================================================
S0 = 100
K = 95
premium = 4.00

contracts = 1
shares_per_contract = 100

purchase_price = K - premium

early_close_price = 1.25
early_close_pl = (
    (premium - early_close_price)
    * shares_per_contract
    * contracts
)

n_steps = 80

dates = pd.date_range(
    start="2025-01-01",
    periods=n_steps + 1,
    freq="B"
)

# ============================================================
# --- 2. SCENARIO PATHS ---
# ============================================================

# OTM expiration
path_otm = (
    np.linspace(S0, 112, n_steps + 1)
    + 2 * np.sin(np.linspace(0, 3 * np.pi, n_steps + 1))
)

# Assigned shares
path_assigned = (
    np.linspace(S0, 82, n_steps + 1)
    + 2.5 * np.sin(np.linspace(0, 3 * np.pi, n_steps + 1))
)

# Close early
close_idx = 44

path_close_visible = (
    np.linspace(S0, 101, close_idx + 1)
    + 2 * np.sin(np.linspace(0, 2 * np.pi, close_idx + 1))
)

path_close = np.r_[
    path_close_visible,
    np.full(n_steps + 1 - len(path_close_visible), np.nan)
]

close_date = dates[close_idx]
close_stock_price = path_close[close_idx]

# ============================================================
# --- 3. P/L CALCULATIONS ---
# ============================================================

otm_pl = premium * shares_per_contract * contracts

assigned_final_price = path_assigned[-1]

assigned_total_pl = (
    (assigned_final_price - K + premium)
    * shares_per_contract
    * contracts
)

# ============================================================
# --- 4. COLORS ---
# ============================================================

off_white = "#e0e0e0"

path_color = "#00d1ff"

strike_color = "#ff4b4b"

purchase_color = "#ff9966"

profit_color = "#00ff88"

loss_color = "#ff4b4b"

close_color = "#ffcc00"

# ============================================================
# --- 5. FIGURE SETUP ---
# ============================================================

fig = make_subplots(
    rows=3,
    cols=1,
    subplot_titles=(
        f"1) OTM Expiration  |  Keep Premium: ${otm_pl:,.0f}",

        f"2) Assignment  |  Net Purchase Price: ${purchase_price:.2f}",

        f"3) Early Close  |  Buyback ${early_close_price:.2f}  |  P/L ${early_close_pl:,.0f}",
    ),
    vertical_spacing=0.10
)

# ============================================================
# --- 6. SCENARIO DEFINITIONS ---
# ============================================================

scenarios = [
    {
        "row": 1,
        "path": path_otm,
        "event_x": dates[-1],
        "event_y": path_otm[-1],
        "color": profit_color,
        "label": "Worthless"
    },

    {
        "row": 2,
        "path": path_assigned,
        "event_x": dates[-1],
        "event_y": assigned_final_price,
        "color": loss_color,
        "label": f"Assigned @ ${K}"
    },

    {
        "row": 3,
        "path": path_close,
        "event_x": close_date,
        "event_y": close_stock_price,
        "color": close_color,
        "label": f"Closed @ ${early_close_price}"
    }
]

# ============================================================
# --- 7. PLOT SCENARIOS ---
# ============================================================

for s in scenarios:

    row = s["row"]

    # Underlying path
    fig.add_trace(

        go.Scatter(

            x=dates,
            y=s["path"],

            mode="lines",

            line=dict(
                color=path_color,
                width=3
            ),

            hovertemplate=
            "Date: %{x|%Y-%m-%d}<br>"
            "Stock Price: $%{y:.2f}"
            "<extra></extra>"
        ),

        row=row,
        col=1
    )

    # Strike line
    fig.add_trace(

        go.Scatter(

            x=[dates[0], dates[-1]],
            y=[K, K],

            mode="lines",

            line=dict(
                color=strike_color,
                width=1.5,
                dash="dash"
            ),

            hoverinfo="skip"
        ),

        row=row,
        col=1
    )

    # Effective purchase price
    fig.add_trace(

        go.Scatter(

            x=[dates[0], dates[-1]],
            y=[purchase_price, purchase_price],

            mode="lines",

            line=dict(
                color=purchase_color,
                width=1.5,
                dash="dot"
            ),

            hoverinfo="skip"
        ),

        row=row,
        col=1
    )

    # Event marker
    fig.add_trace(

        go.Scatter(

            x=[s["event_x"]],
            y=[s["event_y"]],

            mode="markers+text",

            marker=dict(
                size=13,
                color=s["color"],
                line=dict(
                    color=off_white,
                    width=1
                )
            ),

            text=[s["label"]],

            textposition="top center",

            textfont=dict(
                color=off_white,
                size=12
            ),

            hovertemplate=
            "Event Price: $%{y:.2f}"
            "<extra></extra>"
        ),

        row=row,
        col=1
    )

# ============================================================
# --- 8. LINE LABELS ---
# ============================================================

for row in [1, 2, 3]:

    # Strike label
    fig.add_annotation(

        x=dates[4],
        y=K,

        text=f"Strike ${K}",

        showarrow=False,

        xanchor="left",
        yanchor="bottom",

        font=dict(
            color=strike_color,
            size=10
        ),

        row=row,
        col=1
    )

    # Purchase price label
    fig.add_annotation(

        x=dates[4],
        y=purchase_price,

        text=f"Net Cost ${purchase_price:.2f}",

        showarrow=False,

        xanchor="left",
        yanchor="top",

        font=dict(
            color=purchase_color,
            size=10
        ),

        row=row,
        col=1
    )

# ============================================================
# --- 9. DYNAMIC AXIS PADDING ---
# ============================================================

all_prices = np.concatenate([
    path_otm,
    path_assigned,
    path_close_visible
])

y_min = np.nanmin(all_prices)

y_max = np.nanmax(all_prices)

y_pad = (y_max - y_min) * 0.18

x_pad_days = 6

x_start = dates[0] - pd.Timedelta(days=x_pad_days)

x_end = dates[-1] + pd.Timedelta(days=x_pad_days)

# ============================================================
# --- 10. LAYOUT ---
# ============================================================

fig.update_layout(

    template="plotly_dark",

    paper_bgcolor="rgba(0,0,0,0)",

    plot_bgcolor="rgba(0,0,0,0)",

    height=850,

    width=1100,

    margin=dict(
        t=95,
        b=70,
        l=80,
        r=50
    ),

    showlegend=False,

    title=(
        "<b>Short Put: Three States of the World</b><br>"

        f"<sup>"
        f"Strike ${K}  |  "
        f"Premium ${premium:.2f}  |  "
        f"Net Purchase Price ${purchase_price:.2f}"
        f"</sup>"
    )
)

# ============================================================
# --- 11. AXIS STYLE ---
# ============================================================

axis_style = dict(

    showgrid=True,

    gridcolor="rgba(255,255,255,0.1)",

    tickfont=dict(
        color=off_white
    ),

    linecolor=off_white,

    zeroline=False,

    title_font=dict(
        color=off_white
    )
)

# ============================================================
# --- 12. APPLY AXES ---
# ============================================================

for row in [1, 2, 3]:

    fig.update_xaxes(

        axis_style,

        title_text="Date" if row == 3 else "",

        range=[x_start, x_end],

        row=row,
        col=1
    )

    fig.update_yaxes(

        axis_style,

        title_text="Stock Price",

        range=[
            y_min - y_pad,
            y_max + y_pad
        ],

        row=row,
        col=1
    )

fig.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# --- 1. Parameters & Data Generation ---
S0 = 100        # Initial price
K = 105         # Strike price
T = 1.0         # Time to maturity (1 year)
r = 0.05        # Risk-free rate
vol = 0.25      # Volatility
n_steps = 100   # Number of simulation steps
dt = T / n_steps

def put_payoff(S, K):
    return np.maximum(K - S, 0)

def short_put_payoff(S, K):
    return -put_payoff(S, K)

# Black-Scholes European Put Price
def bs_put_price(S, K, tau, r, vol):
    if tau <= 1e-5:
        return put_payoff(S, K)

    d1 = (np.log(S / K) + (r + 0.5 * vol**2) * tau) / (vol * np.sqrt(tau))
    d2 = d1 - vol * np.sqrt(tau)

    return K * np.exp(-r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Short Put Value = Negative Long Put Value
def bs_short_put_price(S, K, tau, r, vol):
    return -bs_put_price(S, K, tau, r, vol)

# Generate a Price Path
np.random.seed(np.random.randint(0, 1000))
rets = np.random.normal(
    (r - 0.5 * vol**2) * dt,
    vol * np.sqrt(dt),
    n_steps
)

path = S0 * np.exp(np.cumsum(rets))
path = np.insert(path, 0, S0)

dates = pd.date_range(
    start="2025-01-01",
    periods=n_steps + 1,
    freq="B"
)

S_range = np.linspace(60, 160, 200)
payoff_vals = short_put_payoff(S_range, K)

# Colors
off_white = "#e0e0e0"
strike_color = "#ff4b4b"
payoff_color = "#ff9966"
path_color = "#00d1ff"
bs_color = "#ff0000"

# --- 2. Figure Setup ---
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=(
        "Short Put Payoff & BS Value",
        "Underlying Price Simulation"
    ),
    horizontal_spacing=0.12
)

bs_vals_initial = bs_short_put_price(S_range, K, T, r, vol)

# Short Put Payoff
fig.add_trace(
    go.Scatter(
        x=S_range,
        y=payoff_vals,
        line=dict(color=payoff_color, width=3)
    ),
    row=1,
    col=1
)

# Strike Line on Payoff Chart
fig.add_trace(
    go.Scatter(
        x=[K, K],
        y=[-60, 5],
        line=dict(color=strike_color, width=1, dash="dash")
    ),
    row=1,
    col=1
)

# Current Price Line on Payoff Chart
fig.add_trace(
    go.Scatter(
        x=[S0, S0],
        y=[-60, 5],
        line=dict(color=off_white, width=2),
        mode="lines"
    ),
    row=1,
    col=1
)

# Black-Scholes Short Put Value
fig.add_trace(
    go.Scatter(
        x=S_range,
        y=bs_vals_initial,
        line=dict(color=bs_color, width=2)
    ),
    row=1,
    col=1
)

# Strike Line on Price Simulation
fig.add_trace(
    go.Scatter(
        x=[dates[0], dates[-1]],
        y=[K, K],
        line=dict(color=strike_color, width=1, dash="dash")
    ),
    row=1,
    col=2
)

# Simulated Price Path
fig.add_trace(
    go.Scatter(
        x=[dates[0]],
        y=[S0],
        line=dict(color=path_color, width=2.5),
        mode="lines"
    ),
    row=1,
    col=2
)

# --- 3. Animation Frames & Slider Steps ---
frames = []
slider_steps = []

for i in range(n_steps + 1):
    current_price = path[i]
    tau = max(T - i * dt, 0.0)
    current_bs_vals = bs_short_put_price(S_range, K, tau, r, vol)

    frame_name = f"f{i}"

    frames.append(
        go.Frame(
            data=[
                go.Scatter(x=S_range, y=payoff_vals),
                go.Scatter(x=[K, K], y=[-60, 5]),
                go.Scatter(x=[current_price, current_price], y=[-60, 5]),
                go.Scatter(x=S_range, y=current_bs_vals),
                go.Scatter(x=[dates[0], dates[-1]], y=[K, K]),
                go.Scatter(x=dates[:i + 1], y=path[:i + 1])
            ],
            name=frame_name
        )
    )

    slider_steps.append(
        {
            "args": [
                [frame_name],
                {
                    "frame": {"duration": 0, "redraw": False},
                    "mode": "immediate",
                    "fromcurrent": True
                }
            ],
            "label": str(i),
            "method": "animate"
        }
    )

fig.frames = frames

# --- 4. Layout with Buttons and Slider ---
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=650,
    width=1200,
    margin=dict(t=100, b=150),
    showlegend=False,
    updatemenus=[
        {
            "type": "buttons",
            "buttons": [
                {
                    "label": "▶ Play",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "frame": {
                                "duration": 40,
                                "redraw": False
                            },
                            "fromcurrent": True
                        }
                    ]
                },
                {
                    "label": "⏸ Pause",
                    "method": "animate",
                    "args": [
                        [None],
                        {
                            "frame": {
                                "duration": 0,
                                "redraw": False
                            },
                            "mode": "immediate",
                            "fromcurrent": True
                        }
                    ]
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ],
    sliders=[
        {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 14, "color": off_white},
                "prefix": "Step: ",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 0},
            "pad": {"b": 10, "t": 50},
            "len": 0.85,
            "x": 0.15,
            "y": 0,
            "steps": slider_steps
        }
    ]
)

axis_style = dict(
    showgrid=True,
    gridcolor="rgba(255,255,255,0.1)",
    tickfont=dict(color=off_white),
    linecolor=off_white,
    zeroline=True,
    zerolinecolor="rgba(255,255,255,0.45)",
    title_font=dict(color=off_white)
)

fig.update_xaxes(
    axis_style,
    range=[60, 160],
    title_text="Asset Price",
    row=1,
    col=1
)

fig.update_yaxes(
    axis_style,
    range=[-60, 5],
    title_text="Payoff / Option Value ($)",
    row=1,
    col=1
)

fig.update_xaxes(
    axis_style,
    range=[dates[0], dates[-1]],
    title_text="Date",
    row=1,
    col=2
)

fig.update_yaxes(
    axis_style,
    range=[60, 160],
    title_text="Asset Price ($)",
    row=1,
    col=2
)

fig.show()
```

```python
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf  # replaces the Interactive Brokers (ibapi) data download

# --- DATA: SPY prices and VIX (30-day implied vol proxy) from Yahoo Finance ---
raw = yf.download(["SPY", "^VIX"], period="5y", interval="1d", auto_adjust=True, progress=False)
price_df = pd.DataFrame({"spy_close": raw["Close"]["SPY"]})
iv_df = pd.DataFrame({"implied_vol": raw["Close"]["^VIX"] / 100.0})  # decimals, like IBKR's IV series
price_df.index = pd.to_datetime(price_df.index).tz_localize(None)
iv_df.index = pd.to_datetime(iv_df.index).tz_localize(None)


# ============================================================
# --- 8. CALCULATE FORWARD 30-DAY REALIZED VOLATILITY ---
# ============================================================

df = price_df[["spy_close"]].join(iv_df[["implied_vol"]], how="inner")

df["log_return"] = np.log(df["spy_close"] / df["spy_close"].shift(1))

forward_window = 30
annualization = np.sqrt(252)

df["forward_30d_realized_vol"] = (
    df["log_return"]
    .rolling(forward_window)
    .std()
    .shift(-forward_window)
    * annualization
)

df = df.dropna()

# Convert to percentages if IBKR returns decimals
if df["implied_vol"].median() < 2:
    df["implied_vol_pct"] = df["implied_vol"] * 100
else:
    df["implied_vol_pct"] = df["implied_vol"]

df["forward_30d_realized_vol_pct"] = df["forward_30d_realized_vol"] * 100

df["vrp"] = df["implied_vol_pct"] - df["forward_30d_realized_vol_pct"]


# ============================================================
# --- 9. REGRESSION ---
# ============================================================

x = df["implied_vol_pct"].values
y = df["forward_30d_realized_vol_pct"].values

slope, intercept = np.polyfit(x, y, 1)

x_line = np.linspace(x.min(), x.max(), 100)
y_line = slope * x_line + intercept

r_squared = np.corrcoef(x, y)[0, 1] ** 2

avg_iv = df["implied_vol_pct"].mean()
avg_fwd_rv = df["forward_30d_realized_vol_pct"].mean()
avg_vrp = df["vrp"].mean()


# ============================================================
# --- 10. PLOTLY DARK STYLE VRP REGRESSION PLOT ---
# ============================================================

off_white = "#e0e0e0"
iv_color = "#00d1ff"
regression_color = "#ff0000"
avg_line_color = "#ff4b4b"

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df["implied_vol_pct"],
        y=df["forward_30d_realized_vol_pct"],
        mode="markers",
        marker=dict(
            color=iv_color,
            size=7,
            opacity=0.55
        ),
        hovertemplate=
        "Date: %{customdata}<br>" +
        "Implied Vol: %{x:.2f}%<br>" +
        "Forward 30D Realized Vol: %{y:.2f}%<br>" +
        "<extra></extra>",
        customdata=df.index.strftime("%Y-%m-%d")
    )
)

fig.add_trace(
    go.Scatter(
        x=x_line,
        y=y_line,
        mode="lines",
        line=dict(color=regression_color, width=3)
    )
)

fig.add_trace(
    go.Scatter(
        x=[x.min(), x.max()],
        y=[x.min(), x.max()],
        mode="lines",
        line=dict(color=off_white, width=1, dash="dash")
    )
)

fig.add_vline(
    x=avg_iv,
    line=dict(color=avg_line_color, width=1, dash="dot")
)

fig.add_hline(
    y=avg_fwd_rv,
    line=dict(color=avg_line_color, width=1, dash="dot")
)

fig.update_layout(
    title=(
        "<b>SPY Volatility Risk Premium Regression</b><br>"
        f"<sup>Implied Vol vs Forward 30-Day Realized Vol | "
        f"Avg VRP: {avg_vrp:.2f}% | R²: {r_squared:.3f}</sup>"
    ),
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=700,
    width=1000,
    margin=dict(t=100, b=80, l=90, r=50),
    showlegend=False
)

axis_style = dict(
    showgrid=True,
    gridcolor="rgba(255,255,255,0.1)",
    tickfont=dict(color=off_white),
    linecolor=off_white,
    zeroline=True,
    zerolinecolor="rgba(255,255,255,0.45)",
    title_font=dict(color=off_white)
)

fig.update_xaxes(
    axis_style,
    title_text="SPY Implied Volatility (%)"
)

fig.update_yaxes(
    axis_style,
    title_text="Average Forward 30-Day Realized Volatility (%)"
)

fig.add_annotation(
    x=0.02,
    y=0.98,
    xref="paper",
    yref="paper",
    showarrow=False,
    align="left",
    text=(
        f"<b>Regression:</b> RV = {slope:.2f} × IV + {intercept:.2f}<br>"
        f"<b>Average IV:</b> {avg_iv:.2f}%<br>"
        f"<b>Average Forward RV:</b> {avg_fwd_rv:.2f}%<br>"
        f"<b>Average VRP:</b> {avg_vrp:.2f}%"
    ),
    font=dict(color=off_white),
    bgcolor="rgba(0,0,0,0.65)",
    bordercolor=off_white,
    borderwidth=1
)

fig.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ============================================================
# --- 1. PARAMETERS ---
# ============================================================
starting_equity = 100_000
n_trades = 140

win_rate = 0.92
premium_income = 750
small_loss = -1_250

# Tail-event settings
tail_event_probability = 0.0025   # 2.5% chance per trade
tail_loss_mean = -142_500
tail_loss_std = 8_000

dates = pd.date_range(start="2025-01-01", periods=n_trades + 1, freq="W")

# ============================================================
# --- 2. SIMULATION FUNCTION ---
# ============================================================
def simulate_cash_secured_put_equity(seed=None):
    if seed is not None:
        np.random.seed(seed)

    pnl = []
    tail_events = []

    for i in range(n_trades):
        tail_event = np.random.rand() < tail_event_probability

        if tail_event:
            trade_pnl = np.random.normal(tail_loss_mean, tail_loss_std)
            tail_events.append(i + 1)
        else:
            if np.random.rand() < win_rate:
                trade_pnl = premium_income + np.random.normal(0, 75)
            else:
                trade_pnl = small_loss + np.random.normal(0, 200)

        pnl.append(trade_pnl)

    pnl = np.array(pnl)

    equity_curve = np.r_[
        starting_equity,
        starting_equity + np.cumsum(pnl)
    ]

    return equity_curve, pnl, tail_events


equity_curve, trade_pnl, tail_events = simulate_cash_secured_put_equity()

running_max = np.maximum.accumulate(equity_curve)

# ============================================================
# --- 3. COLORS ---
# ============================================================
off_white = "#e0e0e0"
equity_color = "#00d1ff"
premium_color = "#00ff88"
tail_color = "#ff4b4b"

# ============================================================
# --- 4. BASE FIGURE ---
# ============================================================
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=[dates[0]],
        y=[equity_curve[0]],
        mode="lines",
        line=dict(color=equity_color, width=3),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Equity: $%{y:,.0f}<extra></extra>"
    )
)

fig.add_trace(
    go.Scatter(
        x=[dates[0]],
        y=[running_max[0]],
        mode="lines",
        line=dict(color="rgba(255,255,255,0)", width=0),
        hoverinfo="skip"
    )
)

fig.add_trace(
    go.Scatter(
        x=[dates[0]],
        y=[equity_curve[0]],
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(255,75,75,0.18)",
        line=dict(color="rgba(255,255,255,0)", width=0),
        hoverinfo="skip"
    )
)

# Tail event markers
fig.add_trace(
    go.Scatter(
        x=[],
        y=[],
        mode="markers+text",
        marker=dict(size=13, color=tail_color, line=dict(color=off_white, width=1)),
        text=[],
        textposition="bottom center",
        textfont=dict(color=off_white, size=12),
        hovertemplate="Tail Event<br>Equity: $%{y:,.0f}<extra></extra>"
    )
)

# ============================================================
# --- 5. ANIMATION FRAMES ---
# ============================================================
frames = []
slider_steps = []

for i in range(n_trades + 1):
    visible_tail_events = [t for t in tail_events if t <= i]

    frames.append(
        go.Frame(
            data=[
                go.Scatter(
                    x=dates[:i + 1],
                    y=equity_curve[:i + 1],
                    mode="lines",
                    line=dict(color=equity_color, width=3)
                ),
                go.Scatter(
                    x=dates[:i + 1],
                    y=running_max[:i + 1],
                    mode="lines",
                    line=dict(color="rgba(255,255,255,0)", width=0)
                ),
                go.Scatter(
                    x=dates[:i + 1],
                    y=equity_curve[:i + 1],
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(255,75,75,0.18)",
                    line=dict(color="rgba(255,255,255,0)", width=0)
                ),
                go.Scatter(
                    x=[dates[t] for t in visible_tail_events],
                    y=[equity_curve[t] for t in visible_tail_events],
                    mode="markers+text",
                    marker=dict(size=13, color=tail_color, line=dict(color=off_white, width=1)),
                    text=["Tail"] * len(visible_tail_events),
                    textposition="bottom center",
                    textfont=dict(color=off_white, size=12)
                )
            ],
            name=f"f{i}"
        )
    )

    slider_steps.append(
        {
            "args": [
                [f"f{i}"],
                {
                    "frame": {"duration": 0, "redraw": False},
                    "mode": "immediate",
                    "fromcurrent": True
                }
            ],
            "label": str(i),
            "method": "animate"
        }
    )

fig.frames = frames

# ============================================================
# --- 6. AXIS PADDING ---
# ============================================================
y_min = equity_curve.min()
y_max = equity_curve.max()
y_pad = (y_max - y_min) * 0.18

x_start = dates[0] - pd.Timedelta(days=30)
x_end = dates[-1] + pd.Timedelta(days=30)

# ============================================================
# --- 7. LAYOUT ---
# ============================================================
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=720,
    width=1150,
    margin=dict(t=100, b=130, l=90, r=60),
    showlegend=False,
    title=(
        "<b>Cash-Secured Put Simulation: High Win Rate with Random Tail Risk</b><br>"
        f"<sup>Win Rate {win_rate:.0%} | Tail Probability {tail_event_probability:.1%} per trade | "
        f"Premium ${premium_income:,.0f}</sup>"
    ),
    updatemenus=[
        {
            "type": "buttons",
            "buttons": [
                {
                    "label": "▶ Play",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "frame": {"duration": 45, "redraw": False},
                            "fromcurrent": True
                        }
                    ]
                },
                {
                    "label": "⏸ Pause",
                    "method": "animate",
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "fromcurrent": True
                        }
                    ]
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 80},
            "showactive": False,
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ],
    sliders=[
        {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 14, "color": off_white},
                "prefix": "Trade: ",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 0},
            "pad": {"b": 10, "t": 50},
            "len": 0.85,
            "x": 0.15,
            "y": 0,
            "steps": slider_steps
        }
    ]
)

axis_style = dict(
    showgrid=True,
    gridcolor="rgba(255,255,255,0.1)",
    tickfont=dict(color=off_white),
    linecolor=off_white,
    zeroline=False,
    title_font=dict(color=off_white)
)

fig.update_xaxes(
    axis_style,
    range=[x_start, x_end],
    title_text="Date"
)

fig.update_yaxes(
    axis_style,
    range=[y_min - y_pad, y_max + y_pad],
    title_text="Account Equity ($)",
    tickprefix="$",
    separatethousands=True
)

fig.show()
```


---

## 8. Volatility Risk Premium How Options Markets Price Fear (Lecture 119)

`2026 Video Lectures/119. Volatility Risk Premium How Options Markets Price Fear/riveot.ipynb` — Volatility risk premium: realized vs implied vol.

```python
import numpy as np
import pandas as pd
import yfinance as yf  # replaces the Interactive Brokers (ibapi) data scripts


def _yf_close(ticker, start, end=None):
    """Daily adjusted close for one ticker from Yahoo Finance, as a Series indexed by date."""
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    s = raw["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = s.dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.name = ticker
    return s


# Writes spx_option_implied_realized_vol_10y.csv (columns: date, option_implied_volatility, realized_volatility)
_px = _yf_close("^GSPC", "2015-06-01")
_iv = _yf_close("^VIX", "2015-06-01") / 100.0          # VIX = 30-day implied vol proxy, in decimals
_rv = np.log(_px).diff().rolling(21).std() * np.sqrt(252)  # 21-day realized vol, annualized
_df = pd.concat([_iv.rename("option_implied_volatility"), _rv.rename("realized_volatility")], axis=1).dropna()
_df = _df[_df.index >= _df.index.max() - pd.DateOffset(years=10)]
_df.rename_axis("date").to_csv("spx_option_implied_realized_vol_10y.csv")

```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- Load data ---
path = "spx_option_implied_realized_vol_10y.csv"
df = pd.read_csv(path)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Use the supplied 21-day realized vol series and implied vol series.
# Drop the initial rows where the 21-day realized-vol window is not available.
df = df.dropna(subset=["realized_volatility", "option_implied_volatility"]).reset_index(drop=True)

dates = df["date"]
realized_vol = df["realized_volatility"]
implied_vol = df["option_implied_volatility"]

# --- Styling ---
off_white = "#e0e0e0"
realized_color = "#00ff88"
implied_color = "#33aaff"
baseline_color = "#777777"

# --- Summary stats ---
latest_date = dates.iloc[-1]
latest_rv = realized_vol.iloc[-1]
latest_iv = implied_vol.iloc[-1]
avg_rv = realized_vol.mean()
avg_iv = implied_vol.mean()
spread_latest = latest_iv - latest_rv

# --- Figure ---
fig = go.Figure()

# Long-run average realized vol baseline
fig.add_trace(go.Scatter(
    x=[dates.iloc[0], dates.iloc[-1]],
    y=[avg_rv, avg_rv],
    mode="lines",
    line=dict(color=baseline_color, width=1, dash="dash"),
    opacity=0.65,
    name=f"Avg Realized Vol · {avg_rv:.1%}",
    hoverinfo="skip"
))

# Long-run average implied vol baseline
fig.add_trace(go.Scatter(
    x=[dates.iloc[0], dates.iloc[-1]],
    y=[avg_iv, avg_iv],
    mode="lines",
    line=dict(color=baseline_color, width=1, dash="dot"),
    opacity=0.65,
    name=f"Avg Implied Vol · {avg_iv:.1%}",
    hoverinfo="skip"
))

# Animated realized volatility
fig.add_trace(go.Scatter(
    x=[dates.iloc[0]],
    y=[realized_vol.iloc[0]],
    mode="lines",
    line=dict(color=realized_color, width=3),
    name="21D Realized Vol",
    hovertemplate="Date: %{x|%Y-%m-%d}<br>21D Realized Vol: %{y:.2%}<extra></extra>"
))

# Animated implied volatility
fig.add_trace(go.Scatter(
    x=[dates.iloc[0]],
    y=[implied_vol.iloc[0]],
    mode="lines",
    line=dict(color=implied_color, width=3),
    name="Option-Implied Vol",
    hovertemplate="Date: %{x|%Y-%m-%d}<br>Implied Vol: %{y:.2%}<extra></extra>"
))

# --- Animation frames ---
frames = []
slider_steps = []

# 5-trading-day stride keeps the HTML responsive.
# Set frame_stride = 1 for a frame on every trading day.
frame_stride = 5
frame_indices = list(range(0, len(df), frame_stride))
if frame_indices[-1] != len(df) - 1:
    frame_indices.append(len(df) - 1)

for i in frame_indices:
    frame_name = f"f{i}"
    frames.append(go.Frame(
        data=[
            go.Scatter(x=[dates.iloc[0], dates.iloc[-1]], y=[avg_rv, avg_rv]),
            go.Scatter(x=[dates.iloc[0], dates.iloc[-1]], y=[avg_iv, avg_iv]),
            go.Scatter(x=dates.iloc[:i + 1], y=realized_vol.iloc[:i + 1]),
            go.Scatter(x=dates.iloc[:i + 1], y=implied_vol.iloc[:i + 1]),
        ],
        name=frame_name
    ))
    slider_steps.append({
        "args": [
            [frame_name],
            {
                "frame": {"duration": 0, "redraw": False},
                "mode": "immediate",
                "fromcurrent": True
            }
        ],
        "label": dates.iloc[i].strftime("%Y"),
        "method": "animate"
    })

fig.frames = frames

axis_style = dict(
    showgrid=True,
    gridcolor="rgba(255,255,255,0.1)",
    tickfont=dict(color=off_white),
    linecolor=off_white,
    zeroline=False,
    title_font=dict(color=off_white)
)

y_min = min(realized_vol.min(), implied_vol.min(), avg_rv, avg_iv)
y_max = max(realized_vol.max(), implied_vol.max(), avg_rv, avg_iv)
y_pad = (y_max - y_min) * 0.08

fig.update_layout(
    title=dict(
        text=(
            "SPX 21-Day Realized Volatility vs Option-Implied Volatility"
            f"<br><sup>Latest: {latest_date:%Y-%m-%d} · "
            f"RV: {latest_rv:.1%} · IV: {latest_iv:.1%} · "
            f"IV − RV: {spread_latest:.1%}</sup>"
        ),
        x=0.5,
        font=dict(color=off_white)
    ),
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=650,
    width=1100,
    margin=dict(t=100, b=150, r=50, l=75),
    legend=dict(
        x=0.98,
        y=0.96,
        xanchor="right",
        yanchor="top",
        bgcolor="rgba(30,30,30,0.8)",
        bordercolor="rgba(255,255,255,0.25)",
        borderwidth=1,
        font=dict(color=off_white)
    ),
    hovermode="x unified",
    updatemenus=[{
        "type": "buttons",
        "buttons": [
            {
                "label": "▶ Play",
                "method": "animate",
                "args": [
                    None,
                    {
                        "frame": {"duration": 20, "redraw": False},
                        "transition": {"duration": 0},
                        "fromcurrent": True
                    }
                ]
            },
            {
                "label": "⏸ Pause",
                "method": "animate",
                "args": [
                    [None],
                    {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "fromcurrent": True
                    }
                ]
            }
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "showactive": False,
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top"
    }],
    sliders=[{
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 14, "color": off_white},
            "prefix": "Through: ",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 0},
        "pad": {"b": 10, "t": 50},
        "len": 0.85,
        "x": 0.15,
        "y": 0,
        "steps": slider_steps
    }]
)

fig.update_xaxes(
    axis_style,
    range=[dates.iloc[0], dates.iloc[-1]],
    title_text="Date"
)

fig.update_yaxes(
    axis_style,
    range=[max(0, y_min - y_pad), y_max + y_pad],
    title_text="Annualized Volatility",
    tickformat=".0%"
)
fig.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# Load data
# =============================================================================

path = "spx_option_implied_realized_vol_10y.csv"
df = pd.read_csv(path)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

df = df.dropna(
    subset=["realized_volatility", "option_implied_volatility"]
).reset_index(drop=True)

dates_all = df["date"]
iv_all = df["option_implied_volatility"].astype(float)
rv21_base = df["realized_volatility"].astype(float)

# =============================================================================
# Expectation / smoothing windows
# =============================================================================
# The supplied realized_volatility is already a 21D realized-vol estimate.
# These windows do NOT recompute true realized vol from returns.
# Instead, they create different expectation processes over the already-smoothed RV21 series:
#
# expectation_window = 1   -> raw RV21(t)
# expectation_window = 5   -> trailing 5D average of RV21(t)
# expectation_window = 21  -> trailing 21D average of RV21(t)
# expectation_window = 63  -> trailing 63D average of RV21(t)
#
# Effective interpretation:
# RV21 is the base realized-vol process.
# Applying another rolling expectation window makes the target smoother/slower.

expectation_windows = [1, 5, 10, 21, 30, 42, 63, 84, 126]

for w in expectation_windows:
    col = f"rv21_expectation_{w}d"

    if w == 1:
        df[col] = rv21_base
    else:
        df[col] = (
            rv21_base
            .rolling(window=w, min_periods=w)
            .mean()
        )

# Keep only rows where the largest expectation window is available.
required_cols = ["option_implied_volatility"] + [
    f"rv21_expectation_{w}d" for w in expectation_windows
]

df = df.dropna(subset=required_cols).reset_index(drop=True)

dates_all = df["date"]
iv_all = df["option_implied_volatility"].astype(float)

# =============================================================================
# Styling
# =============================================================================

off_white = "#e0e0e0"
realized_color = "#00ff88"
implied_color = "#33aaff"
regression_color = "#ff4fd8"
baseline_color = "#777777"
current_color = "#ff3e3e"

# =============================================================================
# Helper functions
# =============================================================================

def regression_stats(xv, yv):
    xv = np.asarray(xv, dtype=float)
    yv = np.asarray(yv, dtype=float)

    if len(xv) < 3 or np.std(xv, ddof=1) == 0:
        return np.nan, np.nan, np.nan, np.nan

    slope, intercept = np.polyfit(xv, yv, 1)
    y_hat = intercept + slope * xv

    ss_res = np.sum((yv - y_hat) ** 2)
    ss_tot = np.sum((yv - np.mean(yv)) ** 2)

    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    corr = np.corrcoef(xv, yv)[0, 1]

    return slope, intercept, r2, corr


def expectation_frame_data(w):
    rv_col = f"rv21_expectation_{w}d"

    temp = df.dropna(subset=["option_implied_volatility", rv_col]).copy()

    d = temp["date"]
    iv = temp["option_implied_volatility"].astype(float).to_numpy()
    rv_expectation = temp[rv_col].astype(float).to_numpy()

    return d, iv, rv_expectation


def expectation_stats(w):
    d, iv, rv_expectation = expectation_frame_data(w)

    slope, intercept, r2, corr = regression_stats(iv, rv_expectation)

    premium = np.mean(iv - rv_expectation)
    hit_rate = np.mean(rv_expectation > iv)

    return {
        "dates": d,
        "iv": iv,
        "rv": rv_expectation,
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "corr": corr,
        "premium": premium,
        "hit_rate": hit_rate,
        "mean_iv": float(np.mean(iv)),
        "mean_rv": float(np.mean(rv_expectation)),
        "n": len(iv)
    }


def title_text(w, stats):
    if w == 1:
        window_label = "Raw supplied RV21(t)"
        effective_note = "base 21D realized-vol process"
    else:
        window_label = f"{w}D trailing expectation of supplied RV21(t)"
        effective_note = f"21D RV process smoothed with an additional {w}D expectation window"

    return (
        "Expectation Changes Everything: IV(t) vs Realized-Vol Expectation Process"
        f"<br><sup>{window_label} · "
        f"Mean IV − RV expectation: {stats['premium']:.2%} · "
        f"RV expectation > IV share: {stats['hit_rate']:.1%} · "
        f"ρ={stats['corr']:.2f} · R²={stats['r2']:.2f} · "
        f"{effective_note}</sup>"
    )

# =============================================================================
# Global axis bounds
# =============================================================================

all_rv_expectations = []

for w in expectation_windows:
    _, _, rv_w = expectation_frame_data(w)
    all_rv_expectations.append(rv_w)

combined_vols = np.concatenate([iv_all.to_numpy()] + all_rv_expectations)

vol_min = max(0, float(np.nanmin(combined_vols) * 0.92))
vol_max = float(np.nanmax(combined_vols) * 1.08)

line_x = np.linspace(vol_min, vol_max, 250)

# =============================================================================
# Initial expectation window
# =============================================================================

w0 = 1
stats0 = expectation_stats(w0)

slope0, intercept0 = stats0["slope"], stats0["intercept"]

reg_y0 = (
    intercept0 + slope0 * line_x
    if np.isfinite(slope0)
    else np.full_like(line_x, np.nan)
)

# =============================================================================
# Figure
# =============================================================================

fig = make_subplots(
    rows=1,
    cols=2,
    column_widths=[0.62, 0.38],
    horizontal_spacing=0.08,
    subplot_titles=(
        "Time-series comparison at date t",
        "IV(t) vs RV expectation at date t"
    )
)

# Left panel: IV(t)
fig.add_trace(
    go.Scatter(
        x=stats0["dates"],
        y=stats0["iv"],
        mode="lines",
        line=dict(color=implied_color, width=3),
        name="IV(t)",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>IV(t): %{y:.2%}<extra></extra>"
    ),
    row=1,
    col=1
)

# Left panel: RV expectation
fig.add_trace(
    go.Scatter(
        x=stats0["dates"],
        y=stats0["rv"],
        mode="lines",
        line=dict(color=realized_color, width=3),
        name=f"RV21 expectation {w0}D",
        hovertemplate=(
            "Date: %{x|%Y-%m-%d}<br>"
            "RV expectation: %{y:.2%}<extra></extra>"
        )
    ),
    row=1,
    col=1
)

# Mean IV baseline
fig.add_trace(
    go.Scatter(
        x=[stats0["dates"].iloc[0], stats0["dates"].iloc[-1]],
        y=[stats0["mean_iv"], stats0["mean_iv"]],
        mode="lines",
        line=dict(color=baseline_color, width=1, dash="dot"),
        opacity=0.65,
        name="Mean IV",
        hoverinfo="skip"
    ),
    row=1,
    col=1
)

# Mean RV expectation baseline
fig.add_trace(
    go.Scatter(
        x=[stats0["dates"].iloc[0], stats0["dates"].iloc[-1]],
        y=[stats0["mean_rv"], stats0["mean_rv"]],
        mode="lines",
        line=dict(color=baseline_color, width=1, dash="dash"),
        opacity=0.65,
        name="Mean RV expectation",
        hoverinfo="skip"
    ),
    row=1,
    col=1
)

# Right panel: y = x
fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_x,
        mode="lines",
        line=dict(color=baseline_color, width=2, dash="dash"),
        opacity=0.85,
        name="y = x",
        hovertemplate="Equal IV/RV expectation: %{x:.2%}<extra></extra>"
    ),
    row=1,
    col=2
)

# Right panel: scatter
fig.add_trace(
    go.Scatter(
        x=stats0["iv"],
        y=stats0["rv"],
        mode="markers",
        marker=dict(
            color=implied_color,
            size=7,
            opacity=0.55,
            line=dict(color="rgba(255,255,255,0.22)", width=0.5)
        ),
        customdata=np.column_stack([
            stats0["dates"].dt.strftime("%Y-%m-%d"),
            stats0["iv"],
            stats0["rv"],
            stats0["iv"] - stats0["rv"]
        ]),
        name="IV vs RV expectation",
        hovertemplate=(
            "Date: %{customdata[0]}<br>"
            "IV(t): %{customdata[1]:.2%}<br>"
            "RV expectation(t): %{customdata[2]:.2%}<br>"
            "IV − RV expectation: %{customdata[3]:.2%}<extra></extra>"
        )
    ),
    row=1,
    col=2
)

# Right panel: regression line
fig.add_trace(
    go.Scatter(
        x=line_x,
        y=reg_y0,
        mode="lines",
        line=dict(color=regression_color, width=4),
        name="OLS fit",
        hovertemplate=(
            "OLS fit<br>"
            "IV: %{x:.2%}<br>"
            "Fitted RV expectation: %{y:.2%}<extra></extra>"
        )
    ),
    row=1,
    col=2
)

# Right panel: mean point
fig.add_trace(
    go.Scatter(
        x=[stats0["mean_iv"]],
        y=[stats0["mean_rv"]],
        mode="markers",
        marker=dict(
            color=current_color,
            size=15,
            opacity=1,
            symbol="diamond",
            line=dict(color="rgba(255,255,255,0.95)", width=1)
        ),
        name="Mean point",
        hovertemplate=(
            "Mean IV: %{x:.2%}<br>"
            "Mean RV expectation: %{y:.2%}<extra></extra>"
        )
    ),
    row=1,
    col=2
)

# =============================================================================
# Animation frames
# =============================================================================

frames = []
slider_steps = []

for w in expectation_windows:
    stats = expectation_stats(w)

    slope, intercept = stats["slope"], stats["intercept"]

    reg_y = (
        intercept + slope * line_x
        if np.isfinite(slope)
        else np.full_like(line_x, np.nan)
    )

    frame_name = f"w{w}"

    frames.append(go.Frame(
        data=[
            # 0 IV(t)
            go.Scatter(
                x=stats["dates"],
                y=stats["iv"]
            ),

            # 1 RV expectation
            go.Scatter(
                x=stats["dates"],
                y=stats["rv"],
                name=f"RV21 expectation {w}D"
            ),

            # 2 Mean IV
            go.Scatter(
                x=[stats["dates"].iloc[0], stats["dates"].iloc[-1]],
                y=[stats["mean_iv"], stats["mean_iv"]]
            ),

            # 3 Mean RV expectation
            go.Scatter(
                x=[stats["dates"].iloc[0], stats["dates"].iloc[-1]],
                y=[stats["mean_rv"], stats["mean_rv"]]
            ),

            # 5 Scatter
            go.Scatter(
                x=stats["iv"],
                y=stats["rv"],
                customdata=np.column_stack([
                    stats["dates"].dt.strftime("%Y-%m-%d"),
                    stats["iv"],
                    stats["rv"],
                    stats["iv"] - stats["rv"]
                ])
            ),

            # 6 Regression
            go.Scatter(
                x=line_x,
                y=reg_y
            ),

            # 7 Mean point
            go.Scatter(
                x=[stats["mean_iv"]],
                y=[stats["mean_rv"]]
            )
        ],
        traces=[0, 1, 2, 3, 5, 6, 7],
        layout=go.Layout(
            title=dict(
                text=title_text(w, stats),
                x=0.5,
                font=dict(color=off_white)
            )
        ),
        name=frame_name
    ))

    slider_steps.append({
        "args": [
            [frame_name],
            {
                "frame": {"duration": 0, "redraw": False},
                "mode": "immediate",
                "fromcurrent": True
            }
        ],
        "label": f"{w}D",
        "method": "animate"
    })

fig.frames = frames

# =============================================================================
# Layout
# =============================================================================

axis_style = dict(
    showgrid=True,
    gridcolor="rgba(255,255,255,0.1)",
    tickfont=dict(color=off_white),
    linecolor=off_white,
    zeroline=False,
    title_font=dict(color=off_white)
)

fig.update_layout(
    title=dict(
        text=title_text(w0, stats0),
        x=0.5,
        font=dict(color=off_white)
    ),
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=720,
    width=1300,
    margin=dict(t=115, b=150, r=50, l=75),
    legend=dict(
        x=0.98,
        y=0.04,
        xanchor="right",
        yanchor="bottom",
        bgcolor="rgba(30,30,30,0.8)",
        bordercolor="rgba(255,255,255,0.25)",
        borderwidth=1,
        font=dict(color=off_white)
    ),
    hovermode="closest",
    updatemenus=[{
        "type": "buttons",
        "buttons": [
            {
                "label": "▶ Play",
                "method": "animate",
                "args": [
                    None,
                    {
                        "frame": {"duration": 750, "redraw": False},
                        "transition": {"duration": 250},
                        "fromcurrent": True
                    }
                ]
            },
            {
                "label": "⏸ Pause",
                "method": "animate",
                "args": [
                    [None],
                    {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "fromcurrent": True
                    }
                ]
            }
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "showactive": False,
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top"
    }],
    sliders=[{
        "active": expectation_windows.index(w0),
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 14, "color": off_white},
            "prefix": "RV expectation window: ",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 0},
        "pad": {"b": 10, "t": 50},
        "len": 0.85,
        "x": 0.15,
        "y": 0,
        "steps": slider_steps
    }]
)

fig.update_xaxes(
    axis_style,
    range=[dates_all.iloc[0], dates_all.iloc[-1]],
    title_text="Date t",
    row=1,
    col=1
)

fig.update_yaxes(
    axis_style,
    range=[vol_min, vol_max],
    title_text="Annualized Volatility",
    tickformat=".0%",
    row=1,
    col=1
)

fig.update_xaxes(
    axis_style,
    range=[vol_min, vol_max],
    title_text="IV(t)",
    tickformat=".0%",
    row=1,
    col=2
)

fig.update_yaxes(
    axis_style,
    range=[vol_min, vol_max],
    title_text="RV21 Expectation Process at t",
    tickformat=".0%",
    scaleanchor="x2",
    scaleratio=1,
    row=1,
    col=2
)

for annotation in fig.layout.annotations:
    annotation.font = dict(color=off_white, size=15)

fig.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- Load local vol-history data for calibration ---
path = "spx_option_implied_realized_vol_10y.csv"
df = pd.read_csv(path)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
df = df.dropna(subset=["realized_volatility", "option_implied_volatility"]).reset_index(drop=True)

iv_hist = df["option_implied_volatility"].to_numpy()
rv_hist = df["realized_volatility"].to_numpy()

latest_date = df["date"].iloc[-1]
latest_iv = float(df["option_implied_volatility"].iloc[-1])
avg_iv = float(np.mean(iv_hist))
p75_iv = float(np.percentile(iv_hist, 75))
p90_iv = float(np.percentile(iv_hist, 90))

# Current SPY reference price from finance quote pulled in the prior step.
spot = 754.83
symbol = "SPY"

# --- Build a realistic non-live IV surface ---
dte = np.array([7, 14, 21, 30, 45, 60, 90, 120, 180], dtype=float)
moneyness = np.linspace(0.82, 1.18, 45)
strikes = spot * moneyness

K, T = np.meshgrid(strikes, dte)
M = K / spot
tau = T / 365.0

# Calibrate surface
atm_anchor = max(0.08, latest_iv)
front_premium = 0.055 * np.exp(-T / 38.0)
term_slope = 0.018 * np.log1p(T / 45.0)
left_skew = 0.22 * np.maximum(1 - M, 0) ** 1.25 * np.exp(-T / 260.0)
right_wing = 0.055 * np.maximum(M - 1, 0) ** 1.55
smile = 0.30 * (np.log(M) ** 2)
ripple = 0.004 * np.sin(12 * (M - 1)) * np.exp(-T / 150.0)

Z = atm_anchor + front_premium + term_slope + left_skew + right_wing + smile + ripple

Z = np.clip(Z, max(0.05, np.percentile(iv_hist, 2) * 0.75), max(0.80, p90_iv * 2.2))

# --- Styling ---
off_white = "#e0e0e0"
paper = "rgba(0,0,0,0)"
plot_bg = "rgba(0,0,0,0)"

# --- Figure ---
fig = go.Figure()

# Only the clean implied volatility surface, no lines or labels
fig.add_trace(go.Surface(
    x=K,
    y=T,
    z=Z,
    colorscale="Magma",
    opacity=0.92,
    name="Implied Vol Surface",
    colorbar=dict(
        title=dict(text="IV", font=dict(color=off_white)),
        tickfont=dict(color=off_white),
        tickformat=".0%",
        len=0.72,
        x=0.94
    ),
    contours=dict(
        z=dict(show=False),
        x=dict(show=False),
        y=dict(show=False)
    ),
    hovertemplate=(
        "Strike: %{x:.2f}<br>"
        "DTE: %{y:.0f}<br>"
        "Implied Vol: %{z:.2%}<extra></extra>"
    )
))

z_floor = float(np.min(Z) * 0.985)

# Camera/view buttons
camera_default = dict(eye=dict(x=1.65, y=-1.85, z=1.15), center=dict(x=0, y=0, z=-0.05))
camera_top = dict(eye=dict(x=0.01, y=0.01, z=2.65), center=dict(x=0, y=0, z=0))
camera_skew = dict(eye=dict(x=0.0, y=-2.45, z=0.55), center=dict(x=0, y=0, z=-0.08))
camera_term = dict(eye=dict(x=2.45, y=0.0, z=0.55), center=dict(x=0, y=0, z=-0.08))

fig.update_layout(
    title=dict(
        text=(
            f"{symbol} Static Implied Volatility Surface"
            f"<br><sup>Transparent 3D Plotly view · Spot ref: {spot:.2f} · "
            f"Calibrated ATM IV: {latest_iv:.1%} · Latest vol-history date: {latest_date:%Y-%m-%d}</sup>"
        ),
        x=0.5,
        font=dict(color=off_white)
    ),
    template="plotly_dark",
    paper_bgcolor=paper,
    plot_bgcolor=plot_bg,
    height=760,
    width=1180,
    margin=dict(t=105, b=55, r=25, l=25),
    legend=dict(
        x=0.02,
        y=0.98,
        xanchor="left",
        yanchor="top",
        bgcolor="rgba(30,30,30,0.8)",
        bordercolor="rgba(255,255,255,0.25)",
        borderwidth=1,
        font=dict(color=off_white)
    ),
    scene=dict(
        bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=dict(text="Strike", font=dict(color=off_white)),
            tickfont=dict(color=off_white),
            gridcolor="rgba(255,255,255,0.10)",
            zerolinecolor="rgba(255,255,255,0.12)",
            showbackground=False,
            range=[strikes.min(), strikes.max()]
        ),
        yaxis=dict(
            title=dict(text="Days to Expiry", font=dict(color=off_white)),
            tickfont=dict(color=off_white),
            gridcolor="rgba(255,255,255,0.10)",
            zerolinecolor="rgba(255,255,255,0.12)",
            showbackground=False,
            range=[dte.min(), dte.max()]
        ),
        zaxis=dict(
            title=dict(text="Implied Volatility", font=dict(color=off_white)),
            tickfont=dict(color=off_white),
            tickformat=".0%",
            gridcolor="rgba(255,255,255,0.10)",
            zerolinecolor="rgba(255,255,255,0.12)",
            showbackground=False,
            range=[z_floor, float(np.max(Z) * 1.05)]
        ),
        camera=camera_default,
        aspectmode="manual",
        aspectratio=dict(x=1.55, y=1.05, z=0.65)
    ),
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            x=0.5,
            y=0.02,
            xanchor="center",
            yanchor="bottom",
            showactive=True,
            bgcolor="rgba(30,30,30,0.8)",
            bordercolor="rgba(255,255,255,0.25)",
            borderwidth=1,
            font=dict(color=off_white),
            buttons=[
                dict(
                    label="Default 3D",
                    method="relayout",
                    args=[{"scene.camera": camera_default}]
                ),
                dict(
                    label="Top Surface",
                    method="relayout",
                    args=[{"scene.camera": camera_top}]
                ),
                dict(
                    label="Skew View",
                    method="relayout",
                    args=[{"scene.camera": camera_skew}]
                ),
                dict(
                    label="Term View",
                    method="relayout",
                    args=[{"scene.camera": camera_term}]
                )
            ]
        )
    ]
)

fig.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- Load data ---
path = "spx_option_implied_realized_vol_10y.csv"
df = pd.read_csv(path)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

df = df.dropna(subset=["realized_volatility", "option_implied_volatility"]).reset_index(drop=True)

# --- Forward 30-trading-day realized volatility ---
# Since realized_volatility is already a rolling realized-vol estimate,
# for date t, y = realized_volatility at t + 30 trading days.
forward_window = 30

df["forward_30d_realized_volatility"] = (
    df["realized_volatility"]
    .astype(float)
    .shift(-forward_window)
)

scatter_df = df.dropna(
    subset=["option_implied_volatility", "forward_30d_realized_volatility"]
).copy()

x = scatter_df["option_implied_volatility"].to_numpy(dtype=float)
y = scatter_df["forward_30d_realized_volatility"].to_numpy(dtype=float)
dates = scatter_df["date"].reset_index(drop=True)

# --- Styling ---
off_white = "#e0e0e0"
point_color = "#33aaff"
current_color = "#00ff88"
regression_color = "#ff4fd8"
baseline_color = "#777777"
zero_color = "#ff3e3e"

# --- Helper functions ---
def regression_stats(xv, yv):
    xv = np.asarray(xv, dtype=float)
    yv = np.asarray(yv, dtype=float)

    if len(xv) < 3 or np.std(xv, ddof=1) == 0:
        return np.nan, np.nan, np.nan, np.nan

    slope, intercept = np.polyfit(xv, yv, 1)
    y_hat = intercept + slope * xv

    ss_res = np.sum((yv - y_hat) ** 2)
    ss_tot = np.sum((yv - np.mean(yv)) ** 2)

    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    corr = np.corrcoef(xv, yv)[0, 1]

    return slope, intercept, r2, corr


def title_text(frame_date, n_points, slope, intercept, r2, corr):
    if np.isfinite(slope):
        stats = (
            f"Through {frame_date:%Y-%m-%d} · n={n_points:,} · "
            f"Regression: y = {slope:.2f}x + {intercept:.1%} · "
            f"R²={r2:.2f} · ρ={corr:.2f}"
        )
    else:
        stats = f"Through {frame_date:%Y-%m-%d} · n={n_points:,}"

    return (
        "Current Implied Vol vs Realized Vol 30 Trading Days Forward"
        f"<br><sup>{stats}</sup>"
    )


# --- Axis bounds ---
combined = np.concatenate([x, y])
axis_min = max(0, float(np.nanmin(combined) * 0.90))
axis_max = float(np.nanmax(combined) * 1.10)

line_x = np.linspace(axis_min, axis_max, 200)
identity_y = line_x

# --- Initial frame index ---
min_points_for_first_frame = 40
initial_i = min(min_points_for_first_frame, len(scatter_df) - 1)

slope0, intercept0, r20, corr0 = regression_stats(
    x[:initial_i + 1],
    y[:initial_i + 1]
)

reg_y0 = (
    intercept0 + slope0 * line_x
    if np.isfinite(slope0)
    else np.full_like(line_x, np.nan)
)

# --- Figure ---
fig = go.Figure()

# y = x line
fig.add_trace(go.Scatter(
    x=line_x,
    y=identity_y,
    mode="lines",
    line=dict(color=baseline_color, width=2, dash="dash"),
    opacity=0.85,
    name="y = x",
    hovertemplate="Equal IV / forward RV: %{x:.2%}<extra></extra>"
))

# Scatter points through time
fig.add_trace(go.Scatter(
    x=x[:initial_i + 1],
    y=y[:initial_i + 1],
    mode="markers",
    marker=dict(
        color=point_color,
        size=7,
        opacity=0.58,
        line=dict(color="rgba(255,255,255,0.25)", width=0.5)
    ),
    customdata=np.column_stack([
        dates.iloc[:initial_i + 1].dt.strftime("%Y-%m-%d"),
        x[:initial_i + 1],
        y[:initial_i + 1],
        y[:initial_i + 1] - x[:initial_i + 1]
    ]),
    name="Historical observations",
    hovertemplate=(
        "Date: %{customdata[0]}<br>"
        "Current IV: %{customdata[1]:.2%}<br>"
        "Realized Vol t+30: %{customdata[2]:.2%}<br>"
        "Forward RV − IV: %{customdata[3]:.2%}<extra></extra>"
    )
))

# Updating regression line
fig.add_trace(go.Scatter(
    x=line_x,
    y=reg_y0,
    mode="lines",
    line=dict(color=regression_color, width=4),
    name="Updating OLS fit",
    hovertemplate=(
        "OLS fit<br>"
        "Current IV: %{x:.2%}<br>"
        "Fitted t+30 RV: %{y:.2%}<extra></extra>"
    )
))

# Current latest point
fig.add_trace(go.Scatter(
    x=[x[initial_i]],
    y=[y[initial_i]],
    mode="markers",
    marker=dict(
        color=current_color,
        size=13,
        opacity=1.0,
        line=dict(color="rgba(255,255,255,0.95)", width=1.25)
    ),
    customdata=[[
        dates.iloc[initial_i].strftime("%Y-%m-%d"),
        x[initial_i],
        y[initial_i],
        y[initial_i] - x[initial_i]
    ]],
    name="Current frame point",
    hovertemplate=(
        "Frame date: %{customdata[0]}<br>"
        "Current IV: %{customdata[1]:.2%}<br>"
        "Realized Vol t+30: %{customdata[2]:.2%}<br>"
        "Forward RV − IV: %{customdata[3]:.2%}<extra></extra>"
    )
))

# Visual pricing gap labels
fig.add_annotation(
    x=axis_min + 0.78 * (axis_max - axis_min),
    y=axis_min + 0.90 * (axis_max - axis_min),
    text="t+30 RV > IV",
    showarrow=False,
    font=dict(color=current_color, size=13),
    opacity=0.8
)

fig.add_annotation(
    x=axis_min + 0.83 * (axis_max - axis_min),
    y=axis_min + 0.63 * (axis_max - axis_min),
    text="t+30 RV < IV",
    showarrow=False,
    font=dict(color=off_white, size=13),
    opacity=0.7
)

# --- Animation frames ---
frames = []
slider_steps = []

frame_stride = 10
frame_indices = list(range(initial_i, len(scatter_df), frame_stride))

if frame_indices[-1] != len(scatter_df) - 1:
    frame_indices.append(len(scatter_df) - 1)

for i in frame_indices:
    slope, intercept, r2, corr = regression_stats(x[:i + 1], y[:i + 1])

    reg_y = (
        intercept + slope * line_x
        if np.isfinite(slope)
        else np.full_like(line_x, np.nan)
    )

    frame_name = f"f{i}"

    frames.append(go.Frame(
        data=[
            go.Scatter(
                x=x[:i + 1],
                y=y[:i + 1],
                customdata=np.column_stack([
                    dates.iloc[:i + 1].dt.strftime("%Y-%m-%d"),
                    x[:i + 1],
                    y[:i + 1],
                    y[:i + 1] - x[:i + 1]
                ])
            ),
            go.Scatter(
                x=line_x,
                y=reg_y
            ),
            go.Scatter(
                x=[x[i]],
                y=[y[i]],
                customdata=[[
                    dates.iloc[i].strftime("%Y-%m-%d"),
                    x[i],
                    y[i],
                    y[i] - x[i]
                ]]
            )
        ],
        traces=[1, 2, 3],
        layout=go.Layout(
            title=dict(
                text=title_text(dates.iloc[i], i + 1, slope, intercept, r2, corr),
                x=0.5,
                font=dict(color=off_white)
            )
        ),
        name=frame_name
    ))

    slider_steps.append({
        "args": [
            [frame_name],
            {
                "frame": {"duration": 0, "redraw": False},
                "mode": "immediate",
                "fromcurrent": True
            }
        ],
        "label": dates.iloc[i].strftime("%Y"),
        "method": "animate"
    })

fig.frames = frames

axis_style = dict(
    showgrid=True,
    gridcolor="rgba(255,255,255,0.1)",
    tickfont=dict(color=off_white),
    linecolor=off_white,
    zeroline=False,
    title_font=dict(color=off_white)
)

# --- Layout ---
fig.update_layout(
    title=dict(
        text=title_text(dates.iloc[initial_i], initial_i + 1, slope0, intercept0, r20, corr0),
        x=0.5,
        font=dict(color=off_white)
    ),
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=700,
    width=1100,
    margin=dict(t=105, b=150, r=50, l=85),
    legend=dict(
        x=0.98,
        y=0.04,
        xanchor="right",
        yanchor="bottom",
        bgcolor="rgba(30,30,30,0.8)",
        bordercolor="rgba(255,255,255,0.25)",
        borderwidth=1,
        font=dict(color=off_white)
    ),
    hovermode="closest",
    updatemenus=[{
        "type": "buttons",
        "buttons": [
            {
                "label": "▶ Play",
                "method": "animate",
                "args": [
                    None,
                    {
                        "frame": {"duration": 28, "redraw": False},
                        "transition": {"duration": 18},
                        "fromcurrent": True
                    }
                ]
            },
            {
                "label": "⏸ Pause",
                "method": "animate",
                "args": [
                    [None],
                    {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "fromcurrent": True
                    }
                ]
            }
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "showactive": False,
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top"
    }],
    sliders=[{
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 14, "color": off_white},
            "prefix": "Through: ",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 0},
        "pad": {"b": 10, "t": 50},
        "len": 0.85,
        "x": 0.15,
        "y": 0,
        "steps": slider_steps
    }]
)

fig.update_xaxes(
    axis_style,
    range=[axis_min, axis_max],
    title_text="Current Option-Implied Volatility",
    tickformat=".0%"
)

fig.update_yaxes(
    axis_style,
    range=[axis_min, axis_max],
    title_text="Realized Volatility 30 Trading Days Forward",
    tickformat=".0%",
    scaleanchor="x",
    scaleratio=1
)

fig.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Load data ---
path = "spx_option_implied_realized_vol_10y.csv"
df = pd.read_csv(path)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Use the supplied 21-day realized vol series and implied vol series.
# Drop the initial rows where the 21-day realized-vol window is not available.
df = df.dropna(subset=["realized_volatility", "option_implied_volatility"]).reset_index(drop=True)

realized_vol = df["realized_volatility"].to_numpy()
implied_vol = df["option_implied_volatility"].to_numpy()

# --- Styling ---
off_white = "#e0e0e0"
realized_color = "#00ff88"
implied_color = "#33aaff"
baseline_color = "#777777"

# --- Helper functions ---
def freedman_diaconis_bins(values, min_bins=25, max_bins=70):
    """Robust empirical bin count."""
    values = np.asarray(values)
    q25, q75 = np.percentile(values, [25, 75])
    iqr = q75 - q25
    n = len(values)
    if iqr <= 0:
        return min_bins
    width = 2 * iqr / np.cbrt(n)
    if width <= 0:
        return min_bins
    bins = int(np.ceil((values.max() - values.min()) / width))
    return int(np.clip(bins, min_bins, max_bins))

def gaussian_kde_manual(values, grid):
    """Simple Gaussian KDE using Silverman's bandwidth rule."""
    values = np.asarray(values, dtype=float)
    grid = np.asarray(grid, dtype=float)
    n = len(values)
    std = values.std(ddof=1)
    if std <= 0 or n <= 1:
        return np.zeros_like(grid)
    bandwidth = 1.06 * std * n ** (-1 / 5)
    z = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * z**2).sum(axis=1) / (n * bandwidth * np.sqrt(2 * np.pi))
    return density

# --- Common empirical histogram setup ---
combined = np.concatenate([realized_vol, implied_vol])
x_min = max(0, combined.min() * 0.92)
x_max = combined.max() * 1.08

n_bins = freedman_diaconis_bins(combined)
bin_edges = np.linspace(x_min, x_max, n_bins + 1)
bin_width = bin_edges[1] - bin_edges[0]
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

rv_density, _ = np.histogram(realized_vol, bins=bin_edges, density=True)
iv_density, _ = np.histogram(implied_vol, bins=bin_edges, density=True)

x_grid = np.linspace(x_min, x_max, 500)
rv_kde = gaussian_kde_manual(realized_vol, x_grid)
iv_kde = gaussian_kde_manual(implied_vol, x_grid)

y_max = max(rv_density.max(), iv_density.max(), rv_kde.max(), iv_kde.max())
y_axis_top = y_max * 1.22
drop_start = y_axis_top * 1.10

# --- Summary stats ---
rv_mean = realized_vol.mean()
iv_mean = implied_vol.mean()
rv_median = np.median(realized_vol)
iv_median = np.median(implied_vol)
rv_std = realized_vol.std(ddof=1)
iv_std = implied_vol.std(ddof=1)

# --- Figure ---
fig = make_subplots(
    rows=1,
    cols=2,
    shared_yaxes=True,
    horizontal_spacing=0.08,
    subplot_titles=(
        f"21D Realized Vol · Mean {rv_mean:.1%}",
        f"Option-Implied Vol · Mean {iv_mean:.1%}"
    )
)

# Initial traces: histogram bars start above the visible plot area, then "fall" to base=0.
fig.add_trace(
    go.Bar(
        x=bin_centers,
        y=rv_density,
        base=np.full_like(rv_density, drop_start),
        width=bin_width * 0.88,
        marker=dict(color=realized_color, opacity=0.72, line=dict(color="rgba(255,255,255,0.18)", width=0.5)),
        name="Realized Vol Histogram",
        customdata=np.column_stack([bin_edges[:-1], bin_edges[1:]]),
        hovertemplate="Bin: %{customdata[0]:.1%}–%{customdata[1]:.1%}<br>Density: %{y:.2f}<extra></extra>"
    ),
    row=1,
    col=1
)

fig.add_trace(
    go.Scatter(
        x=x_grid,
        y=np.zeros_like(rv_kde),
        mode="lines",
        line=dict(color=realized_color, width=3),
        name="Realized Vol KDE",
        hovertemplate="Vol: %{x:.2%}<br>KDE density: %{y:.2f}<extra></extra>"
    ),
    row=1,
    col=1
)

fig.add_trace(
    go.Bar(
        x=bin_centers,
        y=iv_density,
        base=np.full_like(iv_density, drop_start),
        width=bin_width * 0.88,
        marker=dict(color=implied_color, opacity=0.72, line=dict(color="rgba(255,255,255,0.18)", width=0.5)),
        name="Implied Vol Histogram",
        customdata=np.column_stack([bin_edges[:-1], bin_edges[1:]]),
        hovertemplate="Bin: %{customdata[0]:.1%}–%{customdata[1]:.1%}<br>Density: %{y:.2f}<extra></extra>"
    ),
    row=1,
    col=2
)

fig.add_trace(
    go.Scatter(
        x=x_grid,
        y=np.zeros_like(iv_kde),
        mode="lines",
        line=dict(color=implied_color, width=3),
        name="Implied Vol KDE",
        hovertemplate="Vol: %{x:.2%}<br>KDE density: %{y:.2f}<extra></extra>"
    ),
    row=1,
    col=2
)

# Mean reference lines as static shapes.
for col, mean_val, color in [(1, rv_mean, realized_color), (2, iv_mean, implied_color)]:
    fig.add_vline(
        x=mean_val,
        line=dict(color=color, width=1, dash="dash"),
        opacity=0.75,
        row=1,
        col=col
    )

# --- Animation frames ---
frames = []
slider_steps = []
n_frames = 56

for k in range(n_frames):
    t = k / (n_frames - 1)
    # Smooth falling / easing motion
    p = 1 - (1 - t) ** 3
    
    # Slightly delay the KDE so the bars land first, then the fitted curve settles on top.
    kde_p = np.clip((p - 0.15) / 0.85, 0, 1)
    kde_p = 1 - (1 - kde_p) ** 3
    
    current_base = np.full_like(rv_density, drop_start * (1 - p))
    frame_name = f"f{k}"
    
    frames.append(go.Frame(
        data=[
            go.Bar(x=bin_centers, y=rv_density, base=current_base),
            go.Scatter(x=x_grid, y=rv_kde * kde_p),
            go.Bar(x=bin_centers, y=iv_density, base=current_base),
            go.Scatter(x=x_grid, y=iv_kde * kde_p),
        ],
        name=frame_name
    ))
    
    slider_steps.append({
        "args": [
            [frame_name],
            {
                "frame": {"duration": 0, "redraw": False},
                "mode": "immediate",
                "fromcurrent": True
            }
        ],
        "label": f"{int(round(p * 100)):d}%",
        "method": "animate"
    })

fig.frames = frames

axis_style = dict(
    showgrid=True,
    gridcolor="rgba(255,255,255,0.1)",
    tickfont=dict(color=off_white),
    linecolor=off_white,
    zeroline=False,
    title_font=dict(color=off_white)
)

# --- Layout ---
fig.update_layout(
    title=dict(
        text=(
            "Empirical Distributions of SPX Realized and Implied Volatility"
            f"<br><sup>Density-normalized histograms with Gaussian KDE overlays · "
            f"RV median: {rv_median:.1%} · IV median: {iv_median:.1%} · "
            f"RV σ: {rv_std:.1%} · IV σ: {iv_std:.1%}</sup>"
        ),
        x=0.5,
        font=dict(color=off_white)
    ),
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=650,
    width=1100,
    margin=dict(t=110, b=150, r=50, l=75),
    legend=dict(
        x=0.98,
        y=0.96,
        xanchor="right",
        yanchor="top",
        bgcolor="rgba(30,30,30,0.8)",
        bordercolor="rgba(255,255,255,0.25)",
        borderwidth=1,
        font=dict(color=off_white)
    ),
    bargap=0.05,
    hovermode="closest",
    updatemenus=[{
        "type": "buttons",
        "buttons": [
            {
                "label": "▶ Play",
                "method": "animate",
                "args": [
                    None,
                    {
                        "frame": {"duration": 28, "redraw": False},
                        "transition": {"duration": 18},
                        "fromcurrent": True
                    }
                ]
            },
            {
                "label": "⏸ Pause",
                "method": "animate",
                "args": [
                    [None],
                    {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "fromcurrent": True
                    }
                ]
            }
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "showactive": False,
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top"
    }],
    sliders=[{
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 14, "color": off_white},
            "prefix": "Formation: ",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 0},
        "pad": {"b": 10, "t": 50},
        "len": 0.85,
        "x": 0.15,
        "y": 0,
        "steps": slider_steps
    }]
)

fig.update_xaxes(axis_style, range=[x_min, x_max], title_text="Annualized Volatility", tickformat=".0%", row=1, col=1)
fig.update_xaxes(axis_style, range=[x_min, x_max], title_text="Annualized Volatility", tickformat=".0%", row=1, col=2)
fig.update_yaxes(axis_style, range=[0, y_axis_top], title_text="Density", row=1, col=1)
fig.update_yaxes(axis_style, range=[0, y_axis_top], row=1, col=2)

# Style subplot titles
for annotation in fig.layout.annotations:
    annotation.font = dict(color=off_white, size=15)

fig.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# Stylized insurance / VRP simulation
# =============================================================================

np.random.seed(7)

n_steps = 252
dates = pd.date_range("2025-01-01", periods=n_steps + 1, freq="B")
start_value = 100.0

# Market path with one severe drawdown event
rets = np.random.normal(0.00035, 0.0055, n_steps)

crash_start = 92
crash_end = 126

rets[crash_start:crash_end] += -0.018
rets[104] += -0.055
rets[113] += -0.040

# Partial recovery
rets[crash_end:185] += 0.0045

market = start_value * np.cumprod(1 + rets)
market = np.insert(market, 0, start_value)

# =============================================================================
# Insurance economics
# =============================================================================

annual_premium_rate = 0.055
daily_premium = start_value * annual_premium_rate / 252
cumulative_premium = np.arange(n_steps + 1) * daily_premium

protective_floor = 85.0

# Insurance payout kicks in when the market breaches the floor.
cumulative_payout = np.maximum(0, protective_floor - market)
cumulative_payout = np.maximum.accumulate(cumulative_payout)

# Seller: earns premium, pays claim.
insurance_seller = start_value + cumulative_premium - cumulative_payout

# Buyer: owns market, pays premium, receives claim.
insured_equity = market - cumulative_premium + cumulative_payout

# =============================================================================
# Drawdowns
# =============================================================================

def drawdown(series):
    series = np.asarray(series, dtype=float)
    return series / np.maximum.accumulate(series) - 1

market_dd = drawdown(market)
insured_dd = drawdown(insured_equity)
seller_dd = drawdown(insurance_seller)

max_market_dd = market_dd.min()
max_insured_dd = insured_dd.min()
max_seller_dd = seller_dd.min()

# =============================================================================
# Styling
# =============================================================================

off_white = "#e0e0e0"
seller_color = "#ff4fd8"
market_color = "#777777"
insured_color = "#00ff88"
baseline_color = "#777777"

axis_style = dict(
    showgrid=True,
    gridcolor="rgba(255,255,255,0.1)",
    tickfont=dict(color=off_white),
    linecolor=off_white,
    zeroline=False,
    title_font=dict(color=off_white)
)

# =============================================================================
# Figure
# =============================================================================

fig = make_subplots(
    rows=1,
    cols=2,
    horizontal_spacing=0.08,
    subplot_titles=(
        f"Selling Insurance · MDD {max_seller_dd:.1%}",
        f"Buying Insurance · Market MDD {max_market_dd:.1%} vs Insured MDD {max_insured_dd:.1%}"
    )
)

# --- Left chart: insurance seller ---

fig.add_trace(
    go.Scatter(
        x=[dates[0], dates[-1]],
        y=[start_value, start_value],
        mode="lines",
        line=dict(color=baseline_color, width=1, dash="dash"),
        opacity=0.45,
        hoverinfo="skip",
        showlegend=False
    ),
    row=1,
    col=1
)

fig.add_trace(
    go.Scatter(
        x=[dates[0]],
        y=[insurance_seller[0]],
        mode="lines",
        line=dict(color=seller_color, width=4),
        showlegend=False,
        customdata=np.column_stack([
            seller_dd[:1],
            cumulative_premium[:1],
            cumulative_payout[:1]
        ]),
        hovertemplate=(
            "Date: %{x|%Y-%m-%d}<br>"
            "Seller value: %{y:.2f}<br>"
            "Drawdown: %{customdata[0]:.1%}<br>"
            "Premium collected: %{customdata[1]:.2f}<br>"
            "Payout owed: %{customdata[2]:.2f}"
            "<extra></extra>"
        )
    ),
    row=1,
    col=1
)

# --- Right chart: market vs insured equity ---

fig.add_trace(
    go.Scatter(
        x=[dates[0], dates[-1]],
        y=[start_value, start_value],
        mode="lines",
        line=dict(color=baseline_color, width=1, dash="dash"),
        opacity=0.45,
        hoverinfo="skip",
        showlegend=False
    ),
    row=1,
    col=2
)

fig.add_trace(
    go.Scatter(
        x=[dates[0]],
        y=[market[0]],
        mode="lines",
        line=dict(color=market_color, width=3),
        opacity=0.85,
        showlegend=False,
        customdata=np.column_stack([market_dd[:1]]),
        hovertemplate=(
            "Date: %{x|%Y-%m-%d}<br>"
            "Market value: %{y:.2f}<br>"
            "Drawdown: %{customdata[0]:.1%}"
            "<extra></extra>"
        )
    ),
    row=1,
    col=2
)

fig.add_trace(
    go.Scatter(
        x=[dates[0]],
        y=[insured_equity[0]],
        mode="lines",
        line=dict(color=insured_color, width=4),
        showlegend=False,
        customdata=np.column_stack([
            insured_dd[:1],
            cumulative_premium[:1],
            cumulative_payout[:1]
        ]),
        hovertemplate=(
            "Date: %{x|%Y-%m-%d}<br>"
            "Insured value: %{y:.2f}<br>"
            "Drawdown: %{customdata[0]:.1%}<br>"
            "Premium paid: %{customdata[1]:.2f}<br>"
            "Payout received: %{customdata[2]:.2f}"
            "<extra></extra>"
        )
    ),
    row=1,
    col=2
)

# Crash-window shading only
for col in [1, 2]:
    fig.add_vrect(
        x0=dates[crash_start],
        x1=dates[crash_end],
        fillcolor="rgba(255,62,62,0.10)",
        line_width=0,
        layer="below",
        row=1,
        col=col
    )

# =============================================================================
# Animation
# =============================================================================

frames = []
slider_steps = []

frame_stride = 4
frame_indices = list(range(0, n_steps + 1, frame_stride))

if frame_indices[-1] != n_steps:
    frame_indices.append(n_steps)

for i in frame_indices:
    frame_name = f"f{i}"

    frames.append(go.Frame(
        data=[
            go.Scatter(
                x=dates[:i + 1],
                y=insurance_seller[:i + 1],
                customdata=np.column_stack([
                    seller_dd[:i + 1],
                    cumulative_premium[:i + 1],
                    cumulative_payout[:i + 1]
                ])
            ),
            go.Scatter(
                x=dates[:i + 1],
                y=market[:i + 1],
                customdata=np.column_stack([
                    market_dd[:i + 1]
                ])
            ),
            go.Scatter(
                x=dates[:i + 1],
                y=insured_equity[:i + 1],
                customdata=np.column_stack([
                    insured_dd[:i + 1],
                    cumulative_premium[:i + 1],
                    cumulative_payout[:i + 1]
                ])
            )
        ],
        traces=[1, 3, 4],
        name=frame_name
    ))

    slider_steps.append({
        "args": [
            [frame_name],
            {
                "frame": {"duration": 0, "redraw": False},
                "mode": "immediate",
                "fromcurrent": True
            }
        ],
        "label": dates[i].strftime("%b"),
        "method": "animate"
    })

fig.frames = frames

# =============================================================================
# Layout
# =============================================================================

y_min = min(
    insurance_seller.min(),
    market.min(),
    insured_equity.min()
)

y_max = max(
    insurance_seller.max(),
    market.max(),
    insured_equity.max()
)

y_pad = (y_max - y_min) * 0.12

fig.update_layout(
    title=dict(
        text="Volatility Risk Premium as Insurance Economics",
        x=0.5,
        font=dict(color=off_white)
    ),
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=680,
    width=1250,
    margin=dict(t=100, b=145, r=55, l=75),
    showlegend=False,
    hovermode="closest",
    updatemenus=[{
        "type": "buttons",
        "buttons": [
            {
                "label": "▶ Play",
                "method": "animate",
                "args": [
                    None,
                    {
                        "frame": {"duration": 35, "redraw": False},
                        "transition": {"duration": 0},
                        "fromcurrent": True
                    }
                ]
            },
            {
                "label": "⏸ Pause",
                "method": "animate",
                "args": [
                    [None],
                    {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "fromcurrent": True
                    }
                ]
            }
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "showactive": False,
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top"
    }],
    sliders=[{
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 14, "color": off_white},
            "prefix": "Through: ",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 0},
        "pad": {"b": 10, "t": 50},
        "len": 0.85,
        "x": 0.15,
        "y": 0,
        "steps": slider_steps
    }]
)

fig.update_xaxes(
    axis_style,
    range=[dates[0], dates[-1]],
    title_text="Date",
    row=1,
    col=1
)

fig.update_xaxes(
    axis_style,
    range=[dates[0], dates[-1]],
    title_text="Date",
    row=1,
    col=2
)

fig.update_yaxes(
    axis_style,
    range=[y_min - y_pad, y_max + y_pad],
    title_text="Strategy Value",
    row=1,
    col=1
)

fig.update_yaxes(
    axis_style,
    range=[y_min - y_pad, y_max + y_pad],
    title_text="Portfolio Value",
    row=1,
    col=2
)

# Style subplot titles
for annotation in fig.layout.annotations:
    annotation.font = dict(color=off_white, size=15)

fig.show()
```


---

## 9. How to Protect your Stock Portfolio Against a Market Crash (Lecture 138)

`2026 Video Lectures/138. How to Protect your Stock Portfolio Against a Market Crash/crash.ipynb` — Hedging a stock portfolio against a crash.

```python
import numpy as np
import pandas as pd
import yfinance as yf  # replaces the Interactive Brokers (ibapi) data scripts


def _yf_close(ticker, start, end=None):
    """Daily adjusted close for one ticker from Yahoo Finance, as a Series indexed by date."""
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    s = raw["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = s.dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.name = ticker
    return s


# Writes CAT_2025.csv, NVDA_2025.csv, SPY_2025.csv (columns: date, close)
for _t in ["CAT", "NVDA", "SPY"]:
    _s = _yf_close(_t, "2025-01-01", "2026-01-01")
    _s.rename("close").rename_axis("date").reset_index().to_csv(f"{_t}_2025.csv", index=False)

```

```python
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# Config
# ============================================================

DATA_DIR = Path(".")
TICKERS = ["CAT", "NVDA", "SPY"]
FILE_TEMPLATE = "{ticker}_2025.csv"
DATE_COLUMN = "date"
PRICE_COLUMN = "close"

# None = identify the SPY high-water mark and deepest subsequent drawdown.
# Set all four values to timestamps such as "2025-01-02" to override.
MANUAL_PRE_START = None
MANUAL_PRE_END = None
MANUAL_CRASH_START = None
MANUAL_CRASH_END = None

REBASE_VALUE = 100.0
INITIAL_POINTS = 8
FRAME_DURATION = 75
FRAME_STRIDE = 1

# Requested dimensions: 1200 high x 800 wide.
FIG_HEIGHT = 800
FIG_WIDTH = 1200


# ============================================================
# Data helpers
# ============================================================

def load_close_series(ticker: str) -> pd.Series:
    path = DATA_DIR / FILE_TEMPLATE.format(ticker=ticker)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Expected files such as CAT_2025.csv, "
            "NVDA_2025.csv, and SPY_2025.csv."
        )

    df = pd.read_csv(path)
    required = {DATE_COLUMN, PRICE_COLUMN}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df[PRICE_COLUMN] = pd.to_numeric(df[PRICE_COLUMN], errors="coerce")
    df = (
        df[[DATE_COLUMN, PRICE_COLUMN]]
        .dropna()
        .drop_duplicates(subset=DATE_COLUMN, keep="last")
        .sort_values(DATE_COLUMN)
    )

    if df.empty:
        raise ValueError(f"{path} contains no usable date/close observations.")

    return df.set_index(DATE_COLUMN)[PRICE_COLUMN].rename(ticker)


def load_prices() -> pd.DataFrame:
    prices = pd.concat([load_close_series(t) for t in TICKERS], axis=1, join="inner")
    prices = prices.dropna().sort_index()
    if len(prices) < 20:
        raise ValueError("Too few common trading dates across CAT, NVDA, and SPY.")
    return prices


def identify_windows(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    manual = [
        MANUAL_PRE_START,
        MANUAL_PRE_END,
        MANUAL_CRASH_START,
        MANUAL_CRASH_END,
    ]

    if any(value is not None for value in manual):
        if not all(value is not None for value in manual):
            raise ValueError(
                "Set all four MANUAL_* dates, or leave all four as None."
            )
        pre = prices.loc[pd.Timestamp(MANUAL_PRE_START):pd.Timestamp(MANUAL_PRE_END)]
        crash = prices.loc[
            pd.Timestamp(MANUAL_CRASH_START):pd.Timestamp(MANUAL_CRASH_END)
        ]
    else:
        spy = prices["SPY"]
        running_high = spy.cummax()
        drawdown = spy / running_high - 1.0

        crash_end = drawdown.idxmin()
        crash_start = spy.loc[:crash_end].idxmax()

        crash_start_pos = prices.index.get_loc(crash_start)
        crash_end_pos = prices.index.get_loc(crash_end)
        crash_length = crash_end_pos - crash_start_pos + 1

        pre_end_pos = crash_start_pos - 1
        pre_start_pos = max(0, pre_end_pos - crash_length + 1)

        if pre_end_pos < 2:
            raise ValueError("Not enough observations before the detected crash.")

        pre = prices.iloc[pre_start_pos:pre_end_pos + 1]
        crash = prices.iloc[crash_start_pos:crash_end_pos + 1]

    if len(pre) < 4 or len(crash) < 4:
        raise ValueError("Each analysis window needs at least four price observations.")

    return pre, crash


def rebase(window: pd.DataFrame) -> pd.DataFrame:
    return REBASE_VALUE * window / window.iloc[0]


def daily_returns(window: pd.DataFrame) -> pd.DataFrame:
    return 100.0 * window.pct_change().dropna()


def padded_range(values, pad_fraction=0.10, min_pad=0.25):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [-1.0, 1.0]

    low = float(values.min())
    high = float(values.max())
    if np.isclose(low, high):
        pad = max(abs(high) * pad_fraction, min_pad)
    else:
        pad = max((high - low) * pad_fraction, min_pad)
    return [low - pad, high + pad]


def endpoint_text(label: str, n_points: int) -> list[str]:
    return [""] * max(0, n_points - 1) + [label]


def regression_stats(returns: pd.DataFrame) -> dict:
    data = returns[["CAT", "NVDA"]].dropna()
    if len(data) < 3 or np.isclose(data["CAT"].std(ddof=0), 0.0):
        return {
            "x": np.array([]),
            "y": np.array([]),
            "line_x": np.array([]),
            "line_y": np.array([]),
            "alpha": np.nan,
            "beta": np.nan,
            "rho": np.nan,
            "r2": np.nan,
            "n": len(data),
        }

    x = data["CAT"].to_numpy(dtype=float)
    y = data["NVDA"].to_numpy(dtype=float)
    beta, alpha = np.polyfit(x, y, 1)
    fitted = alpha + beta * x

    rho = float(np.corrcoef(x, y)[0, 1])
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = np.nan if np.isclose(ss_tot, 0.0) else 1.0 - ss_res / ss_tot

    line_x = np.array([x.min(), x.max()])
    line_y = alpha + beta * line_x

    return {
        "x": x,
        "y": y,
        "line_x": line_x,
        "line_y": line_y,
        "alpha": float(alpha),
        "beta": float(beta),
        "rho": rho,
        "r2": float(r2),
        "n": len(data),
    }


def equation_text(label: str, stats: dict) -> str:
    if stats["n"] < 3 or not np.isfinite(stats["beta"]):
        return f"<b>{label}</b><br>Collecting return observations..."

    sign = "+" if stats["alpha"] >= 0 else "−"
    return (
        f"<b>{label}</b><br>"
        f"NVDA = {stats['beta']:.2f} × CAT {sign} {abs(stats['alpha']):.2f}%<br>"
        f"ρ = {stats['rho']:.2f} | R² = {stats['r2']:.2f} | n = {stats['n']}"
    )


# ============================================================
# Load, align, detect windows, and calculate full-period stats
# ============================================================

prices = load_prices()
pre_prices, crash_prices = identify_windows(prices)

pre_rebased = rebase(pre_prices)
crash_rebased = rebase(crash_prices)
pre_returns = daily_returns(pre_prices)
crash_returns = daily_returns(crash_prices)

pre_full_stats = regression_stats(pre_returns)
crash_full_stats = regression_stats(crash_returns)
correlation_change = crash_full_stats["rho"] - pre_full_stats["rho"]

pre_date_label = f"{pre_prices.index[0]:%b %d} – {pre_prices.index[-1]:%b %d, %Y}"
crash_date_label = (
    f"{crash_prices.index[0]:%b %d} – {crash_prices.index[-1]:%b %d, %Y}"
)


# ============================================================
# Styling
# ============================================================

off_white = "#e0e0e0"
muted_white = "#aaaaaa"
axis_grid = "rgba(255,255,255,0.10)"
baseline_color = "#777777"
regression_color = "#ffffff"

colors = {
    "CAT": "#ffd84d",
    "NVDA": "#18d618",
    "SPY": "#4da3ff",
}

axis_style = dict(
    showgrid=True,
    gridcolor=axis_grid,
    tickfont=dict(color=off_white),
    linecolor=off_white,
    zeroline=False,
    title_font=dict(color=off_white),
)


# ============================================================
# Fixed ranges for stable animation
# ============================================================

pre_path_range = padded_range(pre_rebased.to_numpy().ravel(), 0.08, 1.0)
crash_path_range = padded_range(crash_rebased.to_numpy().ravel(), 0.08, 1.0)

pre_x_range = padded_range(pre_returns["CAT"], 0.12, 0.35)
pre_y_range = padded_range(pre_returns["NVDA"], 0.12, 0.35)
crash_x_range = padded_range(crash_returns["CAT"], 0.12, 0.35)
crash_y_range = padded_range(crash_returns["NVDA"], 0.12, 0.35)


# Modified text_anchor to shift y position down by ≈30px worth in axis coordinates
def text_anchor(x_range, y_range):
    x = x_range[0] + 0.04 * (x_range[1] - x_range[0])
    # pixel shift: figure is 800px high, bottom row is 0.42 of height
    # so 336px for bottom charts; 30px is ~9% of axis range vertically
    axis_height = y_range[1] - y_range[0]
    pixel_shift = 30
    subplot_pixel_height = 0.42 * FIG_HEIGHT
    frac_shift = pixel_shift / subplot_pixel_height  # as a fraction of axis range
    y = y_range[1] - 0.06 * axis_height - frac_shift * axis_height
    return x, y

pre_text_x, pre_text_y = text_anchor(pre_x_range, pre_y_range)
crash_text_x, crash_text_y = text_anchor(crash_x_range, crash_y_range)


# ============================================================
# Figure layout: two rows x two columns
# ============================================================

fig = make_subplots(
    rows=2,
    cols=2,
    specs=[
        [{"type": "xy"}, {"type": "xy"}],
        [{"type": "xy"}, {"type": "xy"}],
    ],
    row_heights=[0.58, 0.42],
    column_widths=[0.50, 0.50],
    horizontal_spacing=0.10,
    vertical_spacing=0.16,
    subplot_titles=(
        f"Before the crash<br><sup>{pre_date_label}</sup>",
        f"During the crash<br><sup>{crash_date_label}</sup>",
        "NVDA returns regressed on CAT — before",
        "NVDA returns regressed on CAT — crash",
    ),
)

initial_pre_n = min(INITIAL_POINTS, len(pre_rebased))
initial_crash_n = min(INITIAL_POINTS, len(crash_rebased))


# ============================================================
# Trace builders
# ============================================================

def path_trace(ticker, rebased, n_points, showlegend):
    return go.Scatter(
        x=rebased.index[:n_points],
        y=rebased[ticker].iloc[:n_points],
        mode="lines+text",
        line=dict(color=colors[ticker], width=3),
        text=endpoint_text(f"  {ticker}", n_points),
        textposition="middle right",
        textfont=dict(color=colors[ticker], size=11),
        name=ticker,
        legendgroup=ticker,
        showlegend=showlegend,
        hovertemplate=(
            f"{ticker}<br>Date: %{{x|%Y-%m-%d}}<br>"
            "Rebased value: $%{y:.2f}<extra></extra>"
        ),
    )


def regression_traces(returns, n_prices, label, x_text, y_text):
    visible_returns = returns.iloc[: max(0, n_prices - 1)]
    stats = regression_stats(visible_returns)

    scatter = go.Scatter(
        x=stats["x"],
        y=stats["y"],
        mode="markers",
        marker=dict(
            color=colors["NVDA"],
            size=8,
            opacity=0.72,
            line=dict(color=off_white, width=0.5),
        ),
        name=f"{label} daily returns",
        showlegend=False,
        customdata=visible_returns.index[: len(stats["x"])],
        hovertemplate=(
            "Date: %{customdata|%Y-%m-%d}<br>"
            "CAT return: %{x:.2f}%<br>"
            "NVDA return: %{y:.2f}%<extra></extra>"
        ),
    )

    fit = go.Scatter(
        x=stats["line_x"],
        y=stats["line_y"],
        mode="lines",
        line=dict(color=regression_color, width=3),
        name=f"{label} fitted regression",
        showlegend=False,
        hoverinfo="skip",
    )

    equation = go.Scatter(
        x=[x_text],
        y=[y_text],
        mode="text",
        text=[equation_text(label, stats)],
        textposition="top right",
        textfont=dict(color=off_white, size=12),
        showlegend=False,
        hoverinfo="skip",
    )

    return scatter, fit, equation


# ============================================================
# Initial traces, in the exact order reused by every frame
# ============================================================

for ticker in TICKERS:
    fig.add_trace(
        path_trace(ticker, pre_rebased, initial_pre_n, showlegend=True),
        row=1,
        col=1,
    )

for ticker in TICKERS:
    fig.add_trace(
        path_trace(ticker, crash_rebased, initial_crash_n, showlegend=False),
        row=1,
        col=2,
    )

for trace in regression_traces(
    pre_returns,
    initial_pre_n,
    "Before",
    pre_text_x,
    pre_text_y,
):
    fig.add_trace(trace, row=2, col=1)

for trace in regression_traces(
    crash_returns,
    initial_crash_n,
    "Crash",
    crash_text_x,
    crash_text_y,
):
    fig.add_trace(trace, row=2, col=2)


# Reference lines
for col in (1, 2):
    fig.add_hline(
        y=REBASE_VALUE,
        line=dict(color=baseline_color, width=1, dash="dash"),
        opacity=0.75,
        row=1,
        col=col,
    )
    fig.add_hline(
        y=0,
        line=dict(color=baseline_color, width=1, dash="dot"),
        opacity=0.65,
        row=2,
        col=col,
    )
    fig.add_vline(
        x=0,
        line=dict(color=baseline_color, width=1, dash="dot"),
        opacity=0.65,
        row=2,
        col=col,
    )


# ============================================================
# Animation frames
# ============================================================

frames = []
slider_steps = []
max_window_length = max(len(pre_rebased), len(crash_rebased))
frame_positions = list(range(max(INITIAL_POINTS - 1, 0), max_window_length, FRAME_STRIDE))
if frame_positions[-1] != max_window_length - 1:
    frame_positions.append(max_window_length - 1)

for frame_position in frame_positions:
    progress = frame_position / max(1, max_window_length - 1)
    pre_n = max(2, 1 + round(progress * (len(pre_rebased) - 1)))
    crash_n = max(2, 1 + round(progress * (len(crash_rebased) - 1)))

    frame_data = []

    # Top row paths.
    for ticker in TICKERS:
        frame_data.append(path_trace(ticker, pre_rebased, pre_n, showlegend=True))
    for ticker in TICKERS:
        frame_data.append(path_trace(ticker, crash_rebased, crash_n, showlegend=False))

    # Bottom row regressions. Including these in every frame prevents flicker.
    frame_data.extend(
        regression_traces(pre_returns, pre_n, "Before", pre_text_x, pre_text_y)
    )
    frame_data.extend(
        regression_traces(
            crash_returns,
            crash_n,
            "Crash",
            crash_text_x,
            crash_text_y,
        )
    )

    frame_name = f"frame_{frame_position}"
    frames.append(
        go.Frame(
            name=frame_name,
            data=frame_data,
            traces=list(range(len(frame_data))),
        )
    )

    slider_steps.append(
        {
            "label": f"{progress:.0%}",
            "method": "animate",
            "args": [
                [frame_name],
                {
                    "frame": {"duration": 0, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                },
            ],
        }
    )

fig.frames = frames


# ============================================================
# Axes and layout
# ============================================================

fig.update_xaxes(
    axis_style,
    row=1,
    col=1,
    range=[pre_rebased.index[0], pre_rebased.index[-1]],
    title_text="Date",
)
fig.update_yaxes(
    axis_style,
    row=1,
    col=1,
    range=pre_path_range,
    title_text="Growth of $100",
    tickprefix="$",
)

fig.update_xaxes(
    axis_style,
    row=1,
    col=2,
    range=[crash_rebased.index[0], crash_rebased.index[-1]],
    title_text="Date",
)
fig.update_yaxes(
    axis_style,
    row=1,
    col=2,
    range=crash_path_range,
    title_text="Growth of $100",
    tickprefix="$",
)

fig.update_xaxes(
    axis_style,
    row=2,
    col=1,
    range=pre_x_range,
    title_text="CAT daily return",
    ticksuffix="%",
)
fig.update_yaxes(
    axis_style,
    row=2,
    col=1,
    range=pre_y_range,
    title_text="NVDA daily return",
    ticksuffix="%",
)

fig.update_xaxes(
    axis_style,
    row=2,
    col=2,
    range=crash_x_range,
    title_text="CAT daily return",
    ticksuffix="%",
)
fig.update_yaxes(
    axis_style,
    row=2,
    col=2,
    range=crash_y_range,
    title_text="NVDA daily return",
    ticksuffix="%",
)

change_word = "increase" if correlation_change >= 0 else "decrease"
fig.update_layout(
    title=dict(
        text=(
            "CAT / NVDA / SPY: Before and During the 2025 Crash"
            "<br><sup>All price paths rebased to $100 | "
            f"NVDA–CAT correlation: {pre_full_stats['rho']:.2f} → "
            f"{crash_full_stats['rho']:.2f} "
            f"({change_word} of {abs(correlation_change):.2f})</sup>"
        ),
        x=0.5,
        font=dict(color=off_white),
    ),
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=FIG_HEIGHT,
    width=FIG_WIDTH,
    margin=dict(t=145, b=135, r=45, l=70),
    hovermode="closest",
    legend=dict(
        orientation="h",
        x=0.5,
        xanchor="center",
        y=1.02,
        yanchor="bottom",
        font=dict(color=off_white),
    ),
    updatemenus=[
        {
            "type": "buttons",
            "direction": "left",
            "showactive": False,
            "x": 0.10,
            "xanchor": "right",
            "y": -0.055,
            "yanchor": "top",
            "pad": {"r": 10, "t": 60},
            "buttons": [
                {
                    "label": "▶ Play",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "frame": {
                                "duration": FRAME_DURATION,
                                "redraw": True,
                            },
                            "transition": {"duration": 0},
                            "fromcurrent": True,
                        },
                    ],
                },
                {
                    "label": "⏸ Pause",
                    "method": "animate",
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                },
            ],
        }
    ],
    sliders=[
        {
            "active": 0,
            "x": 0.15,
            "len": 0.82,
            "y": -0.055,
            "yanchor": "top",
            "currentvalue": {"prefix": "Reveal: ", "font": {"color": muted_white}},
            "pad": {"b": 10, "t": 40},
            "steps": slider_steps,
        }
    ],
)

fig.update_annotations(font=dict(color=off_white, size=14))

# Display only. Replace with fig.write_html("crash_comparison.html") if desired.
fig.show()
```

```python
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. Configuration
# ============================================================

TICKERS = ["CAT", "NVDA"]

FILE_MAP = {
    "CAT": "CAT_2025.csv",
    "NVDA": "NVDA_2025.csv",
}

COLOR_MAP = {
    "CAT": "#ffd84d",   # yellow
    "NVDA": "#18d618",  # green
}

START_DATE = "2025-01-01"

off_white = "#e0e0e0"
muted_white = "#b8b8b8"


# ============================================================
# 2. Data loading
# ============================================================

def load_close_series(ticker, filename):
    """
    Load a CSV and return its adjusted-close or close series.

    Expected columns include:
        date, close

    The column names are treated as case-insensitive.
    """
    df = pd.read_csv(filename)
    df.columns = [str(column).strip().lower() for column in df.columns]

    if "date" not in df.columns:
        raise ValueError(
            f"{filename} does not contain a 'date' column. "
            f"Available columns: {list(df.columns)}"
        )

    # Prefer adjusted close when available.
    price_candidates = [
        "adjusted_close",
        "adj_close",
        "adj close",
        "close",
    ]

    price_column = next(
        (column for column in price_candidates if column in df.columns),
        None,
    )

    if price_column is None:
        raise ValueError(
            f"{filename} does not contain a usable close-price column. "
            f"Available columns: {list(df.columns)}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[price_column] = pd.to_numeric(df[price_column], errors="coerce")

    df = (
        df.dropna(subset=["date", price_column])
        .sort_values("date")
        .drop_duplicates(subset="date", keep="last")
        .set_index("date")
    )

    return df[price_column].rename(ticker)


price_series = []

for ticker in TICKERS:
    filename = FILE_MAP[ticker]

    try:
        series = load_close_series(ticker, filename)
        price_series.append(series)

    except Exception as error:
        raise RuntimeError(
            f"Unable to load {ticker} from {filename}: {error}"
        ) from error


# Keep only dates shared by both securities.
df_prices = pd.concat(price_series, axis=1, join="inner")
df_prices = df_prices.loc[df_prices.index >= START_DATE].copy()
df_prices = df_prices.dropna()

if len(df_prices) < 3:
    raise ValueError(
        "Not enough overlapping CAT and NVDA price observations "
        "to calculate returns and PCA."
    )


# ============================================================
# 3. Calculate returns and PCA
# ============================================================

df_returns = df_prices.pct_change(fill_method=None).dropna()

if len(df_returns) < 2:
    raise ValueError("Not enough return observations to estimate PCA.")

# Standardize each stock's returns:
# mean = 0 and standard deviation = 1.
scaler = StandardScaler()
scaled_returns = scaler.fit_transform(df_returns[TICKERS])

# Two stocks allow at most two principal components.
pca = PCA(n_components=2)
pca.fit(scaled_returns)

loadings = pd.DataFrame(
    pca.components_.T,
    index=TICKERS,
    columns=["PC1", "PC2"],
)

explained_variance = pca.explained_variance_ratio_ * 100


# ============================================================
# 4. Orient component signs for intuitive interpretation
# ============================================================
#
# PCA signs are mathematically arbitrary. Multiplying an entire
# component by -1 does not change the model.
#
# Orient PC1 so that its average loading is positive.
# Orient PC2 so that NVDA's loading is positive.
# ============================================================

if loadings["PC1"].sum() < 0:
    loadings["PC1"] *= -1

if loadings.loc["NVDA", "PC2"] < 0:
    loadings["PC2"] *= -1


# ============================================================
# 5. Build Plotly figure
# ============================================================

fig = make_subplots(
    rows=1,
    cols=2,
    horizontal_spacing=0.12,
    subplot_titles=(
        (
            "<b>PC1: Common Return Factor</b>"
            f"<br><sup>Variance explained: "
            f"{explained_variance[0]:.1f}%</sup>"
        ),
        (
            "<b>PC2: Relative-Performance Factor</b>"
            f"<br><sup>Variance explained: "
            f"{explained_variance[1]:.1f}%</sup>"
        ),
    ),
)


def add_component_bars(component, column):
    subset = loadings[component]

    fig.add_trace(
        go.Bar(
            x=subset.index,
            y=subset.values,
            marker=dict(
                color=[COLOR_MAP[ticker] for ticker in subset.index],
                line=dict(
                    color="rgba(255,255,255,0.30)",
                    width=1,
                ),
            ),
            text=subset.values,
            texttemplate="%{text:.3f}",
            textposition="outside",
            cliponaxis=False,
            customdata=[
                explained_variance[column - 1]
            ] * len(subset),
            hovertemplate=(
                "<b>%{x}</b><br>"
                f"Component: {component}<br>"
                "Loading: %{y:.4f}<br>"
                "Variance explained: %{customdata:.2f}%"
                "<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=column,
    )


add_component_bars("PC1", 1)
add_component_bars("PC2", 2)


# Zero-reference lines.
fig.add_hline(
    y=0,
    line=dict(
        color="rgba(255,255,255,0.60)",
        width=1.5,
        dash="dash",
    ),
    row=1,
    col=1,
)

fig.add_hline(
    y=0,
    line=dict(
        color="rgba(255,255,255,0.60)",
        width=1.5,
        dash="dash",
    ),
    row=1,
    col=2,
)


# Dummy traces create a ticker legend.
for ticker in TICKERS:
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                color=COLOR_MAP[ticker],
                size=11,
            ),
            name=ticker,
            hoverinfo="skip",
        )
    )


# ============================================================
# 6. Axis ranges and styling
# ============================================================

max_abs_loading = max(
    0.75,
    float(loadings.abs().to_numpy().max()) * 1.25,
)

axis_style = dict(
    showgrid=True,
    gridcolor="rgba(255,255,255,0.10)",
    zeroline=False,
    linecolor=off_white,
    tickfont=dict(color=off_white, size=12),
    title_font=dict(color=off_white),
)

fig.update_xaxes(
    axis_style,
    title_text="Security",
    row=1,
    col=1,
)

fig.update_xaxes(
    axis_style,
    title_text="Security",
    row=1,
    col=2,
)

fig.update_yaxes(
    axis_style,
    title_text="PCA loading",
    range=[-max_abs_loading, max_abs_loading],
    row=1,
    col=1,
)

fig.update_yaxes(
    axis_style,
    title_text="PCA loading",
    range=[-max_abs_loading, max_abs_loading],
    row=1,
    col=2,
)


# ============================================================
# 7. Layout
# ============================================================

start_label = df_returns.index.min().strftime("%b %d, %Y")
end_label = df_returns.index.max().strftime("%b %d, %Y")

fig.update_layout(
    title=dict(
        text=(
            "<b>CAT and NVDA Principal Component Loadings</b>"
            "<br>"
            "<sup>"
            "Standardized daily returns | "
            f"{start_label} through {end_label}"
            "</sup>"
        ),
        x=0.5,
        xanchor="center",
        font=dict(
            color=off_white,
            size=22,
        ),
    ),
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=500,
    width=1000,
    margin=dict(
        t=125,
        b=90,
        l=75,
        r=45,
    ),
    bargap=0.30,
    showlegend=True,
    legend=dict(
        orientation="h",
        x=0.5,
        xanchor="center",
        y=-0.16,
        yanchor="top",
        font=dict(color=off_white),
    ),
)

for annotation in fig.layout.annotations:
    annotation.font = dict(
        color=off_white,
        size=14,
    )

# ============================================================
# 9. Display
# ============================================================

fig.show()
```

```python
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# Config
# ============================================================

DATA_DIR = Path('.')
TICKERS = ['CAT', 'NVDA', 'SPY']
FILE_TEMPLATE = '{ticker}_2025.csv'
DATE_COLUMN = 'date'
PRICE_COLUMN = 'close'
START_DATE = '2025-01-01'

X_TICKERS = ['CAT', 'NVDA']
Y_TICKER = 'SPY'

INITIAL_POINTS = 8
FRAME_DURATION = 75
FRAME_STRIDE = 1

FIG_HEIGHT = 800
FIG_WIDTH = 1200


# ============================================================
# Data helpers
# ============================================================

def load_close_series(ticker: str) -> pd.Series:
    path = DATA_DIR / FILE_TEMPLATE.format(ticker=ticker)
    if not path.exists():
        raise FileNotFoundError(
            f'Missing {path}. Expected files such as CAT_2025.csv, '
            'NVDA_2025.csv, and SPY_2025.csv.'
        )

    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = {DATE_COLUMN, PRICE_COLUMN}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'{path} is missing columns: {sorted(missing)}')

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors='coerce')
    df[PRICE_COLUMN] = pd.to_numeric(df[PRICE_COLUMN], errors='coerce')

    df = (
        df[[DATE_COLUMN, PRICE_COLUMN]]
        .dropna()
        .drop_duplicates(subset=DATE_COLUMN, keep='last')
        .sort_values(DATE_COLUMN)
    )

    if df.empty:
        raise ValueError(f'{path} contains no usable observations.')

    return df.set_index(DATE_COLUMN)[PRICE_COLUMN].rename(ticker)


def load_prices() -> pd.DataFrame:
    prices = pd.concat([load_close_series(t) for t in TICKERS], axis=1, join='inner')
    prices = prices.loc[prices.index >= START_DATE].dropna().sort_index()

    if len(prices) < 10:
        raise ValueError('Too few common trading dates across CAT, NVDA, and SPY.')

    return prices


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    rets = 100.0 * prices.pct_change().dropna()
    if rets.empty:
        raise ValueError('No return observations available after pct_change().')
    return rets


def padded_range(values, pad_fraction=0.12, min_pad=0.25):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [-1.0, 1.0]

    low = float(values.min())
    high = float(values.max())
    if np.isclose(low, high):
        pad = max(abs(high) * pad_fraction, min_pad)
    else:
        pad = max((high - low) * pad_fraction, min_pad)
    return [low - pad, high + pad]


def regression_stats(x: pd.Series, y: pd.Series) -> dict:
    data = pd.concat([x.rename('x'), y.rename('y')], axis=1).dropna()

    if len(data) < 3 or np.isclose(data['x'].std(ddof=0), 0.0):
        return {
            'x': np.array([]),
            'y': np.array([]),
            'line_x': np.array([]),
            'line_y': np.array([]),
            'alpha': np.nan,
            'beta': np.nan,
            'rho': np.nan,
            'r2': np.nan,
            'n': len(data),
            'dates': pd.to_datetime([]),
        }

    x_values = data['x'].to_numpy(dtype=float)
    y_values = data['y'].to_numpy(dtype=float)
    beta, alpha = np.polyfit(x_values, y_values, 1)
    fitted = alpha + beta * x_values

    rho = float(np.corrcoef(x_values, y_values)[0, 1])
    ss_res = float(np.sum((y_values - fitted) ** 2))
    ss_tot = float(np.sum((y_values - y_values.mean()) ** 2))
    r2 = np.nan if np.isclose(ss_tot, 0.0) else 1.0 - ss_res / ss_tot

    line_x = np.array([x_values.min(), x_values.max()])
    line_y = alpha + beta * line_x

    return {
        'x': x_values,
        'y': y_values,
        'line_x': line_x,
        'line_y': line_y,
        'alpha': float(alpha),
        'beta': float(beta),
        'rho': rho,
        'r2': float(r2),
        'n': len(data),
        'dates': data.index,
    }


def equation_text(x_label: str, stats: dict) -> str:
    if stats['n'] < 3 or not np.isfinite(stats['beta']):
        return f'<b>SPY on {x_label}</b><br>Collecting return observations...'

    sign = '+' if stats['alpha'] >= 0 else '−'
    return (
        f'<b>SPY on {x_label}</b><br>'
        f'SPY = {stats["beta"]:.2f} × {x_label} {sign} {abs(stats["alpha"]):.2f}%<br>'
        f'ρ = {stats["rho"]:.2f} | R² = {stats["r2"]:.2f} | n = {stats["n"]}'
    )


def text_anchor(x_range, y_range):
    x = x_range[0] + 0.04 * (x_range[1] - x_range[0])
    axis_height = y_range[1] - y_range[0]
    pixel_shift = 25
    subplot_pixel_height = 0.78 * FIG_HEIGHT
    frac_shift = pixel_shift / subplot_pixel_height
    y = y_range[1] - 0.05 * axis_height - frac_shift * axis_height
    return x, y


# ============================================================
# Load data and calculate returns
# ============================================================

prices = load_prices()
returns = daily_returns(prices)

full_stats_left = regression_stats(returns['CAT'], returns['SPY'])
full_stats_right = regression_stats(returns['NVDA'], returns['SPY'])

start_label = f'{returns.index[0]:%b %d, %Y}'
end_label = f'{returns.index[-1]:%b %d, %Y}'


# ============================================================
# Styling
# ============================================================

off_white = '#e0e0e0'
muted_white = '#aaaaaa'
axis_grid = 'rgba(255,255,255,0.10)'
baseline_color = '#777777'
regression_color = '#ffffff'

colors = {
    'CAT': '#ffd84d',
    'NVDA': '#18d618',
    'SPY': '#4da3ff',
}

axis_style = dict(
    showgrid=True,
    gridcolor=axis_grid,
    tickfont=dict(color=off_white),
    linecolor=off_white,
    zeroline=False,
    title_font=dict(color=off_white),
)


# ============================================================
# Fixed axis ranges for stable animation
# ============================================================

cat_x_range = padded_range(returns['CAT'], 0.12, 0.35)
nvda_x_range = padded_range(returns['NVDA'], 0.12, 0.35)
spy_y_range = padded_range(returns['SPY'], 0.12, 0.35)

left_text_x, left_text_y = text_anchor(cat_x_range, spy_y_range)
right_text_x, right_text_y = text_anchor(nvda_x_range, spy_y_range)


# ============================================================
# Figure layout
# ============================================================

fig = make_subplots(
    rows=1,
    cols=2,
    specs=[[{'type': 'xy'}, {'type': 'xy'}]],
    horizontal_spacing=0.10,
    subplot_titles=(
        f'SPY returns regressed on CAT<br><sup>{start_label} – {end_label}</sup>',
        f'SPY returns regressed on NVDA<br><sup>{start_label} – {end_label}</sup>',
    ),
)

initial_n = min(INITIAL_POINTS, len(returns))


# ============================================================
# Trace builders
# ============================================================

def regression_traces(x_ticker, n_points, x_text, y_text):
    visible = returns.iloc[:n_points]
    stats = regression_stats(visible[x_ticker], visible[Y_TICKER])

    scatter = go.Scatter(
        x=stats['x'],
        y=stats['y'],
        mode='markers',
        marker=dict(
            color=colors[Y_TICKER],
            size=8,
            opacity=0.74,
            line=dict(color=off_white, width=0.5),
        ),
        name=f'{Y_TICKER} vs {x_ticker}',
        showlegend=False,
        customdata=stats['dates'],
        hovertemplate=(
            'Date: %{customdata|%Y-%m-%d}<br>'
            f'{x_ticker} return: %{{x:.2f}}%<br>'
            f'{Y_TICKER} return: %{{y:.2f}}%<extra></extra>'
        ),
    )

    fit = go.Scatter(
        x=stats['line_x'],
        y=stats['line_y'],
        mode='lines',
        line=dict(color=regression_color, width=3),
        name=f'{x_ticker} fitted line',
        showlegend=False,
        hoverinfo='skip',
    )

    equation = go.Scatter(
        x=[x_text],
        y=[y_text],
        mode='text',
        text=[equation_text(x_ticker, stats)],
        textposition='top right',
        textfont=dict(color=off_white, size=12),
        showlegend=False,
        hoverinfo='skip',
    )

    return scatter, fit, equation


# ============================================================
# Initial traces
# ============================================================

for trace in regression_traces('CAT', initial_n, left_text_x, left_text_y):
    fig.add_trace(trace, row=1, col=1)

for trace in regression_traces('NVDA', initial_n, right_text_x, right_text_y):
    fig.add_trace(trace, row=1, col=2)

for col in (1, 2):
    fig.add_hline(
        y=0,
        line=dict(color=baseline_color, width=1, dash='dot'),
        opacity=0.70,
        row=1,
        col=col,
    )
    fig.add_vline(
        x=0,
        line=dict(color=baseline_color, width=1, dash='dot'),
        opacity=0.70,
        row=1,
        col=col,
    )


# ============================================================
# Animation frames
# ============================================================

frames = []
slider_steps = []
frame_positions = list(range(max(INITIAL_POINTS - 1, 0), len(returns), FRAME_STRIDE))
if frame_positions[-1] != len(returns) - 1:
    frame_positions.append(len(returns) - 1)

for frame_position in frame_positions:
    n_points = frame_position + 1
    frame_data = []

    # Include all panel traces in every frame to preserve full redraw stability.
    frame_data.extend(regression_traces('CAT', n_points, left_text_x, left_text_y))
    frame_data.extend(regression_traces('NVDA', n_points, right_text_x, right_text_y))

    frame_name = f'frame_{frame_position}'
    frames.append(
        go.Frame(
            name=frame_name,
            data=frame_data,
            traces=list(range(len(frame_data))),
        )
    )

    progress = n_points / len(returns)
    slider_steps.append(
        {
            'label': f'{progress:.0%}',
            'method': 'animate',
            'args': [
                [frame_name],
                {
                    'frame': {'duration': 0, 'redraw': True},
                    'mode': 'immediate',
                    'transition': {'duration': 0},
                },
            ],
        }
    )

fig.frames = frames


# ============================================================
# Axes and layout
# ============================================================

fig.update_xaxes(
    axis_style,
    row=1,
    col=1,
    range=cat_x_range,
    title_text='CAT daily return',
    ticksuffix='%',
)
fig.update_yaxes(
    axis_style,
    row=1,
    col=1,
    range=spy_y_range,
    title_text='SPY daily return',
    ticksuffix='%',
)

fig.update_xaxes(
    axis_style,
    row=1,
    col=2,
    range=nvda_x_range,
    title_text='NVDA daily return',
    ticksuffix='%',
)
fig.update_yaxes(
    axis_style,
    row=1,
    col=2,
    range=spy_y_range,
    title_text='SPY daily return',
    ticksuffix='%',
)

fig.update_layout(
    title=dict(
        text=(
            'SPY Return Regressions: CAT vs NVDA'
            '<br><sup>'
            f'Full-sample correlations | '
            f'SPY–CAT: {full_stats_left["rho"]:.2f} | '
            f'SPY–NVDA: {full_stats_right["rho"]:.2f}'
            '</sup>'
        ),
        x=0.5,
        font=dict(color=off_white),
    ),
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=FIG_HEIGHT,
    width=FIG_WIDTH,
    margin=dict(t=145, b=135, r=45, l=70),
    hovermode='closest',
    showlegend=False,
    updatemenus=[
        {
            'type': 'buttons',
            'direction': 'left',
            'showactive': False,
            'x': 0.10,
            'xanchor': 'right',
            'y': -0.08,
            'yanchor': 'top',
            'pad': {'r': 10, 't': 60},
            'buttons': [
                {
                    'label': '▶ Play',
                    'method': 'animate',
                    'args': [
                        None,
                        {
                            'frame': {
                                'duration': FRAME_DURATION,
                                'redraw': True,
                            },
                            'transition': {'duration': 0},
                            'fromcurrent': True,
                        },
                    ],
                },
                {
                    'label': '⏸ Pause',
                    'method': 'animate',
                    'args': [
                        [None],
                        {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 0},
                        },
                    ],
                },
            ],
        }
    ],
    sliders=[
        {
            'active': 0,
            'x': 0.15,
            'len': 0.82,
            'y': -0.08,
            'yanchor': 'top',
            'currentvalue': {'prefix': 'Reveal: ', 'font': {'color': muted_white}},
            'pad': {'b': 10, 't': 40},
            'steps': slider_steps,
        }
    ],
)

fig.update_annotations(font=dict(color=off_white, size=14))
fig.show()
```

```python
from pathlib import Path
from math import erf, exp, log, sqrt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# Config
# ============================================================

DATA_DIR = Path('.')
TICKERS = ['CAT', 'NVDA', 'SPY']
FILE_TEMPLATE = '{ticker}_2025.csv'
DATE_COLUMN = 'date'
PRICE_COLUMN = 'close'

# Leave both as None to detect the SPY peak-to-trough drawdown automatically.
MANUAL_DRAWDOWN_START = None
MANUAL_DRAWDOWN_END = None

REBASE_VALUE = 100.0
INITIAL_POINTS = 8
FRAME_DURATION = 50         # <<-- 100% faster: half of 100
FRAME_STRIDE = 2            # <<-- 100% faster: double of 1

FIG_HEIGHT = 800
FIG_WIDTH = 1200

# Underlying basket weights.
WEIGHTS = {
    'CAT': 0.50,
    'NVDA': 0.50,
}

# Stylized SPY put overlay.
PUT_STRIKE_MULTIPLIER = 1.00
PUT_IMPLIED_VOL = 0.35
RISK_FREE_RATE = 0.04
PUT_EXTRA_TRADING_DAYS = 21
OVERHEDGE_NOTIONAL = 1.50

# ============================================================
# Data helpers
# ============================================================

def load_close_series(ticker: str) -> pd.Series:
    path = DATA_DIR / FILE_TEMPLATE.format(ticker=ticker)
    if not path.exists():
        raise FileNotFoundError(
            f'Missing {path}. Expected CAT_2025.csv, NVDA_2025.csv, '
            'and SPY_2025.csv in the same directory as this script.'
        )

    df = pd.read_csv(path)
    df.columns = [str(column).strip().lower() for column in df.columns]

    required = {DATE_COLUMN, PRICE_COLUMN}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'{path} is missing columns: {sorted(missing)}')

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors='coerce')
    df[PRICE_COLUMN] = pd.to_numeric(df[PRICE_COLUMN], errors='coerce')

    df = (
        df[[DATE_COLUMN, PRICE_COLUMN]]
        .dropna()
        .drop_duplicates(subset=DATE_COLUMN, keep='last')
        .sort_values(DATE_COLUMN)
    )

    if df.empty:
        raise ValueError(f'{path} contains no usable date/close observations.')

    return df.set_index(DATE_COLUMN)[PRICE_COLUMN].rename(ticker)



def load_prices() -> pd.DataFrame:
    prices = pd.concat(
        [load_close_series(ticker) for ticker in TICKERS],
        axis=1,
        join='inner',
    )
    prices = prices.dropna().sort_index()
    prices = prices.loc[prices.index.year == 2025]

    if len(prices) < 20:
        raise ValueError('Too few common 2025 trading dates across CAT, NVDA, and SPY.')

    return prices



def identify_drawdown(prices: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    manual = [MANUAL_DRAWDOWN_START, MANUAL_DRAWDOWN_END]

    if any(value is not None for value in manual):
        if not all(value is not None for value in manual):
            raise ValueError(
                'Set both MANUAL_DRAWDOWN_START and MANUAL_DRAWDOWN_END, '
                'or leave both as None.'
            )

        drawdown_start = pd.Timestamp(MANUAL_DRAWDOWN_START)
        drawdown_end = pd.Timestamp(MANUAL_DRAWDOWN_END)
    else:
        spy = prices['SPY']
        running_high = spy.cummax()
        drawdown = spy / running_high - 1.0

        drawdown_end = drawdown.idxmin()
        drawdown_start = spy.loc[:drawdown_end].idxmax()

    if drawdown_start not in prices.index or drawdown_end not in prices.index:
        raise ValueError('The selected drawdown dates are not common trading dates.')

    if drawdown_end <= drawdown_start:
        raise ValueError('The drawdown end must occur after the drawdown start.')

    return drawdown_start, drawdown_end



def rebased_basket(prices: pd.DataFrame) -> pd.Series:
    basket_growth = sum(
        weight * prices[ticker] / prices[ticker].iloc[0]
        for ticker, weight in WEIGHTS.items()
    )
    return basket_growth.rename('CAT/NVDA basket growth')



def rebase(series: pd.Series, base: float = REBASE_VALUE) -> pd.Series:
    return (base * series / series.iloc[0]).rename(series.name)



def daily_returns(series: pd.Series) -> pd.Series:
    return (100.0 * series.pct_change().dropna()).rename(series.name)



def padded_range(values, pad_fraction=0.10, min_pad=0.25):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return [-1.0, 1.0]

    low = float(values.min())
    high = float(values.max())

    if np.isclose(low, high):
        pad = max(abs(high) * pad_fraction, min_pad)
    else:
        pad = max((high - low) * pad_fraction, min_pad)

    return [low - pad, high + pad]


# ============================================================
# Put valuation and monetization
# ============================================================


def norm_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))



def black_scholes_put_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
) -> float:
    if time_to_expiry <= 0.0:
        return max(strike - spot, 0.0)

    if spot <= 0.0 or strike <= 0.0 or volatility <= 0.0:
        return max(strike * exp(-rate * time_to_expiry) - spot, 0.0)

    sigma_sqrt_t = volatility * sqrt(time_to_expiry)
    d1 = (
        log(spot / strike)
        + (rate + 0.5 * volatility ** 2) * time_to_expiry
    ) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t

    return (
        strike * exp(-rate * time_to_expiry) * norm_cdf(-d2)
        - spot * norm_cdf(-d1)
    )



def build_portfolios(
    analysis_prices: pd.DataFrame,
    monetization_date: pd.Timestamp,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    basket_growth = rebased_basket(analysis_prices)
    unhedged_nav = (REBASE_VALUE * basket_growth).rename('Unhedged 50/50')

    spy = analysis_prices['SPY']
    spot0 = float(spy.iloc[0])
    strike = PUT_STRIKE_MULTIPLIER * spot0

    monetization_position = analysis_prices.index.get_loc(monetization_date)
    expiry_trading_days = max(monetization_position + PUT_EXTRA_TRADING_DAYS, 1)

    option_shares = OVERHEDGE_NOTIONAL * REBASE_VALUE / spot0

    option_values = []
    for position, spot in enumerate(spy):
        remaining_days = max(expiry_trading_days - position, 0)
        time_to_expiry = remaining_days / 252.0
        option_values.append(
            black_scholes_put_price(
                spot=float(spot),
                strike=strike,
                time_to_expiry=time_to_expiry,
                rate=RISK_FREE_RATE,
                volatility=PUT_IMPLIED_VOL,
            )
        )

    put_price = pd.Series(option_values, index=analysis_prices.index, name='SPY put price')
    put_market_value = (option_shares * put_price).rename('SPY put market value')

    initial_put_cost = float(put_market_value.iloc[0])
    initial_equity_capital = REBASE_VALUE - initial_put_cost

    if initial_equity_capital <= 0.0:
        raise ValueError(
            'The put overlay costs at least the entire initial portfolio. '
            'Reduce OVERHEDGE_NOTIONAL, PUT_IMPLIED_VOL, or strike level.'
        )

    pre_sale_nav = initial_equity_capital * basket_growth + put_market_value

    sale_nav = float(pre_sale_nav.loc[monetization_date])
    basket_at_sale = float(basket_growth.loc[monetization_date])
    post_sale_basket_units = sale_nav / basket_at_sale

    hedged_nav = pre_sale_nav.copy()
    hedged_nav.loc[monetization_date:] = (
        post_sale_basket_units * basket_growth.loc[monetization_date:]
    )
    hedged_nav = hedged_nav.rename('Hedged then reinvested')

    return unhedged_nav, hedged_nav, put_market_value


# ============================================================
# Regression helpers
# ============================================================


def regression_fit(x: pd.Series, y: pd.Series) -> dict:
    data = pd.concat([x.rename('x'), y.rename('y')], axis=1).dropna()

    if len(data) < 3 or np.isclose(data['x'].std(ddof=0), 0.0):
        return {
            'line_x': np.array([]),
            'line_y': np.array([]),
            'data': data,
        }

    x_values = data['x'].to_numpy(dtype=float)
    y_values = data['y'].to_numpy(dtype=float)
    beta, alpha = np.polyfit(x_values, y_values, 1)

    line_x = np.array([x_values.min(), x_values.max()])
    line_y = alpha + beta * line_x

    return {
        'line_x': line_x,
        'line_y': line_y,
        'data': data,
    }


# ============================================================
# Load data and create the analysis window
# ============================================================

prices = load_prices()
drawdown_start, drawdown_end = identify_drawdown(prices)

# Observe the crisis and the remainder of 2025.
analysis_prices = prices.loc[drawdown_start:].copy()

unhedged_nav, hedged_nav, put_market_value = build_portfolios(
    analysis_prices=analysis_prices,
    monetization_date=drawdown_end,
)

spy_rebased = rebase(analysis_prices['SPY']).rename('SPY')

unhedged_returns = daily_returns(unhedged_nav)
hedged_returns = daily_returns(hedged_nav)
spy_returns = daily_returns(spy_rebased)

analysis_start_label = f'{analysis_prices.index[0]:%b %d, %Y}'
analysis_end_label = f'{analysis_prices.index[-1]:%b %d, %Y}'
monetization_label = f'{drawdown_end:%b %d, %Y}'


# ============================================================
# Styling
# ============================================================

off_white = '#e0e0e0'
muted_white = '#aaaaaa'
axis_grid = 'rgba(255,255,255,0.10)'
baseline_color = '#777777'
regression_color = '#ffffff'

colors = {
    'UNHEDGED': '#ffd84d',
    'HEDGED': '#ff7f50',
    'SPY': '#4da3ff',
    'DRAWDOWN_POINT': '#ff4d4d',
    'RECOVERY_POINT': '#4da3ff',
    'MONETIZATION': '#c084fc',
}

axis_style = dict(
    showgrid=True,
    gridcolor=axis_grid,
    tickfont=dict(color=off_white),
    linecolor=off_white,
    zeroline=False,
    title_font=dict(color=off_white),
)


# ============================================================
# Fixed ranges for stable animation
# ============================================================

top_y_range = padded_range(
    np.r_[
        unhedged_nav.to_numpy(),
        hedged_nav.to_numpy(),
        spy_rebased.to_numpy(),
    ],
    pad_fraction=0.08,
    min_pad=1.0,
)

unhedged_x_range = padded_range(unhedged_returns, 0.12, 0.35)
hedged_x_range = padded_range(hedged_returns, 0.12, 0.35)
spy_y_range = padded_range(spy_returns, 0.12, 0.35)


# ============================================================
# Figure layout
# ============================================================

fig = make_subplots(
    rows=2,
    cols=2,
    specs=[
        [{'type': 'xy'}, {'type': 'xy'}],
        [{'type': 'xy'}, {'type': 'xy'}],
    ],
    row_heights=[0.58, 0.42],
    column_widths=[0.50, 0.50],
    horizontal_spacing=0.10,
    vertical_spacing=0.16,
    subplot_titles=(
        'Unhedged 50/50 CAT + NVDA',
        f'Put monetized {monetization_label}; proceeds reinvested',
        'SPY on unhedged portfolio returns',
        'SPY on hedged portfolio returns',
    ),
)

initial_n = min(INITIAL_POINTS, len(analysis_prices))


# ============================================================
# Trace builders
# ============================================================


def path_trace(series: pd.Series, n_points: int, color: str, dash: str | None = None):
    return go.Scatter(
        x=series.index[:n_points],
        y=series.iloc[:n_points],
        mode='lines',
        line=dict(color=color, width=3, dash=dash),
        showlegend=False,
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Rebased value: $%{y:.2f}<extra></extra>',
    )



def monetization_marker(n_points: int):
    visible_dates = analysis_prices.index[:n_points]
    if drawdown_end not in visible_dates:
        return go.Scatter(x=[], y=[], mode='markers', showlegend=False, hoverinfo='skip')

    return go.Scatter(
        x=[drawdown_end],
        y=[hedged_nav.loc[drawdown_end]],
        mode='markers',
        marker=dict(
            color=colors['MONETIZATION'],
            size=14,
            symbol='diamond',
            line=dict(color=off_white, width=1),
        ),
        showlegend=False,
        hovertemplate=(
            'Put monetized and proceeds reinvested<br>'
            f'Date: {monetization_label}<br>'
            'Portfolio value: $%{y:.2f}<extra></extra>'
        ),
    )



def regression_traces(portfolio_returns: pd.Series, n_prices: int):
    n_returns = max(0, n_prices - 1)
    visible_portfolio = portfolio_returns.iloc[:n_returns]
    visible_spy = spy_returns.iloc[:n_returns]

    fit = regression_fit(visible_portfolio, visible_spy)
    data = fit['data']

    drawdown_data = data.loc[data.index <= drawdown_end]
    recovery_data = data.loc[data.index > drawdown_end]

    fit_line = go.Scatter(
        x=fit['line_x'],
        y=fit['line_y'],
        mode='lines',
        line=dict(color=regression_color, width=3),
        showlegend=False,
        hoverinfo='skip',
    )

    recovery_points = go.Scatter(
        x=recovery_data['x'].to_numpy(dtype=float),
        y=recovery_data['y'].to_numpy(dtype=float),
        mode='markers',
        marker=dict(
            color=colors['RECOVERY_POINT'],
            size=7,
            opacity=0.55,
            line=dict(color=off_white, width=0.4),
        ),
        showlegend=False,
        customdata=recovery_data.index,
        hovertemplate=(
            'Phase: Recovery<br>'
            'Date: %{customdata|%Y-%m-%d}<br>'
            'Portfolio return: %{x:.2f}%<br>'
            'SPY return: %{y:.2f}%<extra></extra>'
        ),
    )

    # Draw drawdown points last so they always sit on top.
    drawdown_points = go.Scatter(
        x=drawdown_data['x'].to_numpy(dtype=float),
        y=drawdown_data['y'].to_numpy(dtype=float),
        mode='markers',
        marker=dict(
            color=colors['DRAWDOWN_POINT'],
            size=9,
            opacity=0.95,
            line=dict(color=off_white, width=0.6),
        ),
        showlegend=False,
        customdata=drawdown_data.index,
        hovertemplate=(
            'Phase: Drawdown<br>'
            'Date: %{customdata|%Y-%m-%d}<br>'
            'Portfolio return: %{x:.2f}%<br>'
            'SPY return: %{y:.2f}%<extra></extra>'
        ),
    )

    return fit_line, recovery_points, drawdown_points


# ============================================================
# Initial traces — exact order is reused by every frame
# ============================================================

# Top-left: unhedged portfolio and SPY.
fig.add_trace(path_trace(unhedged_nav, initial_n, colors['UNHEDGED']), row=1, col=1)
fig.add_trace(path_trace(spy_rebased, initial_n, colors['SPY'], dash='dash'), row=1, col=1)

# Top-right: monetized/reinvested hedge portfolio and SPY.
fig.add_trace(path_trace(hedged_nav, initial_n, colors['HEDGED']), row=1, col=2)
fig.add_trace(path_trace(spy_rebased, initial_n, colors['SPY'], dash='dash'), row=1, col=2)
fig.add_trace(monetization_marker(initial_n), row=1, col=2)

# Bottom regressions: fit line first, then recovery, then drawdown on top.
for trace in regression_traces(unhedged_returns, initial_n):
    fig.add_trace(trace, row=2, col=1)

for trace in regression_traces(hedged_returns, initial_n):
    fig.add_trace(trace, row=2, col=2)


# Fixed labels inside top-row panels to avoid animated endpoint-text jitter.
fig.add_annotation(
    x=0.02,
    y=0.96,
    xref='x domain',
    yref='y domain',
    text=f"<span style='color:{colors['UNHEDGED']}'>● Unhedged 50/50</span><br><span style='color:{colors['SPY']}'>● SPY</span>",
    showarrow=False,
    align='left',
    font=dict(size=12, color=off_white),
    bgcolor='rgba(0,0,0,0.20)',
    bordercolor='rgba(255,255,255,0.15)',
    borderwidth=1,
    row=1,
    col=1,
)
fig.add_annotation(
    x=0.02,
    y=0.96,
    xref='x2 domain',
    yref='y2 domain',
    text=f"<span style='color:{colors['HEDGED']}'>● Hedged then reinvested</span><br><span style='color:{colors['SPY']}'>● SPY</span>",
    showarrow=False,
    align='left',
    font=dict(size=12, color=off_white),
    bgcolor='rgba(0,0,0,0.20)',
    bordercolor='rgba(255,255,255,0.15)',
    borderwidth=1,
    row=1,
    col=2,
)


# Reference lines and monetization date.
for col in (1, 2):
    fig.add_hline(
        y=REBASE_VALUE,
        line=dict(color=baseline_color, width=1, dash='dash'),
        opacity=0.75,
        row=1,
        col=col,
    )
    fig.add_vline(
        x=drawdown_end,
        line=dict(color=colors['MONETIZATION'], width=1.5, dash='dot'),
        opacity=0.75,
        row=1,
        col=col,
    )
    fig.add_hline(
        y=0,
        line=dict(color=baseline_color, width=1, dash='dot'),
        opacity=0.65,
        row=2,
        col=col,
    )
    fig.add_vline(
        x=0,
        line=dict(color=baseline_color, width=1, dash='dot'),
        opacity=0.65,
        row=2,
        col=col,
    )


# ============================================================
# Animation frames
# ============================================================

frames = []
slider_steps = []
frame_positions = list(range(max(INITIAL_POINTS - 1, 0), len(analysis_prices), FRAME_STRIDE))
if frame_positions[-1] != len(analysis_prices) - 1:
    frame_positions.append(len(analysis_prices) - 1)

TRACE_COUNT = 11

for frame_position in frame_positions:
    n_points = frame_position + 1
    frame_data = []

    # Top row.
    frame_data.append(path_trace(unhedged_nav, n_points, colors['UNHEDGED']))
    frame_data.append(path_trace(spy_rebased, n_points, colors['SPY'], dash='dash'))
    frame_data.append(path_trace(hedged_nav, n_points, colors['HEDGED']))
    frame_data.append(path_trace(spy_rebased, n_points, colors['SPY'], dash='dash'))
    frame_data.append(monetization_marker(n_points))

    # Bottom row. Keep order fixed: fit, recovery, drawdown.
    frame_data.extend(regression_traces(unhedged_returns, n_points))
    frame_data.extend(regression_traces(hedged_returns, n_points))

    frame_name = f'frame_{frame_position}'
    frames.append(
        go.Frame(
            name=frame_name,
            data=frame_data,
            traces=list(range(TRACE_COUNT)),
        )
    )

    current_date = analysis_prices.index[frame_position]
    slider_steps.append(
        {
            'label': f'{current_date:%b %d}',
            'method': 'animate',
            'args': [
                [frame_name],
                {
                    'frame': {'duration': 0, 'redraw': False},
                    'mode': 'immediate',
                    'transition': {'duration': 0},
                },
            ],
        }
    )

fig.frames = frames


# ============================================================
# Axes and layout
# ============================================================

for col in (1, 2):
    fig.update_xaxes(
        axis_style,
        row=1,
        col=col,
        range=[analysis_prices.index[0], analysis_prices.index[-1]],
        title_text='Date',
    )
    fig.update_yaxes(
        axis_style,
        row=1,
        col=col,
        range=top_y_range,
        title_text='Growth of $100',
        tickprefix='$',
    )

fig.update_xaxes(
    axis_style,
    row=2,
    col=1,
    range=unhedged_x_range,
    title_text='Unhedged portfolio daily return',
    ticksuffix='%',
)
fig.update_yaxes(
    axis_style,
    row=2,
    col=1,
    range=spy_y_range,
    title_text='SPY daily return',
    ticksuffix='%',
)

fig.update_xaxes(
    axis_style,
    row=2,
    col=2,
    range=hedged_x_range,
    title_text='Hedged portfolio daily return',
    ticksuffix='%',
)
fig.update_yaxes(
    axis_style,
    row=2,
    col=2,
    range=spy_y_range,
    title_text='SPY daily return',
    ticksuffix='%',
)

fig.update_layout(
    title=dict(
        text=(
            'Monetizing Convexity: Hedge the Drawdown, Then Reinvest'
            '<br><sup>'
            f'{analysis_start_label} – {analysis_end_label} | '
            'red regression points = drawdown; blue = post-monetization recovery'
            '</sup>'
        ),
        x=0.5,
        font=dict(color=off_white),
    ),
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=FIG_HEIGHT,
    width=FIG_WIDTH,
    margin=dict(t=145, b=135, r=45, l=70),
    hovermode='closest',
    showlegend=False,
    updatemenus=[
        {
            'type': 'buttons',
            'direction': 'left',
            'showactive': False,
            'x': 0.10,
            'xanchor': 'right',
            'y': -0.055,
            'yanchor': 'top',
            'pad': {'r': 10, 't': 60},
            'buttons': [
                {
                    'label': '▶ Play',
                    'method': 'animate',
                    'args': [
                        None,
                        {
                            'frame': {'duration': FRAME_DURATION, 'redraw': False},
                            'transition': {'duration': 0},
                            'fromcurrent': True,
                        },
                    ],
                },
                {
                    'label': '⏸ Pause',
                    'method': 'animate',
                    'args': [
                        [None],
                        {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0},
                        },
                    ],
                },
            ],
        }
    ],
    sliders=[
        {
            'active': 0,
            'x': 0.15,
            'len': 0.82,
            'y': -0.055,
            'yanchor': 'top',
            'currentvalue': {
                'prefix': 'Date: ',
                'font': {'color': muted_white},
            },
            'pad': {'b': 10, 't': 40},
            'steps': slider_steps,
        }
    ],
)

fig.update_annotations(font=dict(color=off_white, size=14))
fig.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import erf, exp, log, sqrt

# ============================================================
# 1. Configuration
# ============================================================

SPOT0 = 100.0
RISK_FREE_RATE = 0.03

# Chart horizon and option life.
N_DAYS = 126                        # ~6 months of trading days
DATES = pd.bdate_range('2025-01-02', periods=N_DAYS)
DT_YEARS = 1.0 / 252.0

# Crash timing for the synthetic spot path.
CRASH_START = 48
CRASH_END = 62

# Strategy A: rich-skew tail put.
PUT_A_STRIKE = 90.0

# Strategy B: put spread designed to have similar crash protection
# but lower carry bleed because the expensive skew is partly sold.
PUT_B_LONG_STRIKE = 95.0
PUT_B_SHORT_STRIKE = 85.0

# Use a calibration point during the crash to scale the spread so both
# strategies have roughly similar protection/value when the market breaks.
CALIBRATION_DAY = CRASH_END

# ============================================================
# 2. Option math helpers
# ============================================================

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def black_scholes_put_price(spot: float, strike: float, tau: float, rate: float, vol: float) -> float:
    if tau <= 0.0:
        return max(strike - spot, 0.0)
    if spot <= 0.0 or strike <= 0.0 or vol <= 0.0:
        return max(strike * exp(-rate * tau) - spot, 0.0)
    sigma_sqrt_t = vol * sqrt(tau)
    d1 = (log(spot / strike) + (rate + 0.5 * vol ** 2) * tau) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    return strike * exp(-rate * tau) * norm_cdf(-d2) - spot * norm_cdf(-d1)

def synthetic_iv_surface(strike, tau, spot_ref=SPOT0):
    """
    Synthetic but equity-like implied vol surface.
    - Lower strikes carry higher IV (put skew / smirk).
    - Longer expiries carry slightly more IV.
    - Mild convexity keeps the shape surface-like.
    """
    strike = np.asarray(strike, dtype=float)
    tau = np.asarray(tau, dtype=float)
    log_moneyness = np.log(strike / spot_ref)
    base = 0.16
    term = 0.06 * np.sqrt(np.maximum(tau, 1e-6))
    skew = -0.24 * log_moneyness
    smile = 0.30 * log_moneyness ** 2
    iv = base + term + skew + smile
    return np.clip(iv, 0.10, 0.85)

# ============================================================
# 3. Build a synthetic market path
# ============================================================

def synthetic_spot_path() -> pd.Series:
    """
    Piecewise-smooth path:
    - slow grind lower before the crash,
    - sharp crash,
    - partial recovery into expiry.
    """
    t = np.arange(N_DAYS)
    path = np.empty(N_DAYS, dtype=float)

    for i in t:
        if i <= CRASH_START:
            frac = i / CRASH_START if CRASH_START > 0 else 0.0
            path[i] = SPOT0 - 4.0 * frac + 0.6 * np.sin(i / 8.0)
        elif i <= CRASH_END:
            frac = (i - CRASH_START) / (CRASH_END - CRASH_START)
            start_level = SPOT0 - 4.0 + 0.6 * np.sin(CRASH_START / 8.0)
            path[i] = start_level + (78.0 - start_level) * frac + 0.5 * np.sin(i / 2.0)
        else:
            frac = (i - CRASH_END) / (N_DAYS - 1 - CRASH_END)
            path[i] = 78.0 + (94.0 - 78.0) * frac + 0.9 * np.sin(i / 10.0)

    return pd.Series(path, index=DATES, name='Underlying')

spot_path = synthetic_spot_path()

# Remaining option maturity decreases through time.
remaining_tau = pd.Series(
    np.maximum((N_DAYS - 1 - np.arange(N_DAYS)) * DT_YEARS, 1e-6),
    index=DATES,
    name='tau',
)

# ============================================================
# 4. Price the two put strategies
# ============================================================

# Strategy A: outright 90 put.
put_a_iv = synthetic_iv_surface(PUT_A_STRIKE, remaining_tau.to_numpy(), spot_ref=SPOT0)
put_a_value = pd.Series(
    [
        black_scholes_put_price(float(s), PUT_A_STRIKE, float(tau), RISK_FREE_RATE, float(iv))
        for s, tau, iv in zip(spot_path, remaining_tau, put_a_iv)
    ],
    index=DATES,
    name='Long 90 Put',
)

# Strategy B: scaled 95/85 put spread.
put_b_long_iv = synthetic_iv_surface(PUT_B_LONG_STRIKE, remaining_tau.to_numpy(), spot_ref=SPOT0)
put_b_short_iv = synthetic_iv_surface(PUT_B_SHORT_STRIKE, remaining_tau.to_numpy(), spot_ref=SPOT0)

put_b_long = pd.Series(
    [
        black_scholes_put_price(float(s), PUT_B_LONG_STRIKE, float(tau), RISK_FREE_RATE, float(iv))
        for s, tau, iv in zip(spot_path, remaining_tau, put_b_long_iv)
    ],
    index=DATES,
    name='Long 95 Put',
)

put_b_short = pd.Series(
    [
        black_scholes_put_price(float(s), PUT_B_SHORT_STRIKE, float(tau), RISK_FREE_RATE, float(iv))
        for s, tau, iv in zip(spot_path, remaining_tau, put_b_short_iv)
    ],
    index=DATES,
    name='Short 85 Put',
)

raw_spread = put_b_long - put_b_short

# Scale the spread to roughly match the outright put during the crash.
calibration_date = DATES[CALIBRATION_DAY]
spread_scale = put_a_value.loc[calibration_date] / raw_spread.loc[calibration_date]
strategy_b_value = (spread_scale * raw_spread).rename('Scaled 95/85 Put Spread')

# ============================================================
# 5. Build the implied vol surface for the left panel
# ============================================================

strike_grid = np.linspace(70, 115, 61)
days_grid = np.arange(14, 181, 7)
tau_grid = days_grid / 252.0

X_strikes, Y_days = np.meshgrid(strike_grid, days_grid)
Z_iv = 100.0 * synthetic_iv_surface(X_strikes, Y_days / 252.0, spot_ref=SPOT0)

# ============================================================
# 6. Plotly figure
# ============================================================

off_white = '#e0e0e0'
muted_white = '#b8b8b8'
color_a = '#ff4d4d'     # richer / bleedier strategy
color_b = '#18d618'     # lower-carry strategy
underlying_color = '#4da3ff'

fig = make_subplots(
    rows=1,
    cols=2,
    specs=[[{'type': 'surface'}, {'type': 'xy'}]],
    column_widths=[0.48, 0.52],
    horizontal_spacing=0.08,
    subplot_titles=(
        'Synthetic Put Implied Volatility Surface',
        'Two Put Strategies: Similar Crash Protection, Different Carry Bleed',
    ),
)

# ----- Left: IV surface -----
fig.add_trace(
    go.Surface(
        x=X_strikes,
        y=Y_days,
        z=Z_iv,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title='IV %'),
        contours={
            'z': dict(show=True, usecolormap=True, highlightcolor='white', project_z=True)
        },
        hovertemplate=(
            'Strike: %{x:.0f}<br>'
            'Days to expiry: %{y:.0f}<br>'
            'Implied vol: %{z:.1f}%<extra></extra>'
        ),
        name='IV Surface',
    ),
    row=1,
    col=1,
)

# Removed the points/text overlay on the surface panel per instructions.

# ----- Right: strategy value paths -----
fig.add_trace(
    go.Scatter(
        x=DATES,
        y=put_a_value,
        mode='lines',
        line=dict(color=color_a, width=3),
        name='Long 90 Put',
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Value: %{y:.2f}<extra>Long 90 Put</extra>',
    ),
    row=1,
    col=2,
)

fig.add_trace(
    go.Scatter(
        x=DATES,
        y=strategy_b_value,
        mode='lines',
        line=dict(color=color_b, width=3),
        name=f'{spread_scale:.2f}x 95/85 Put Spread',
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Value: %{y:.2f}<extra>Scaled 95/85 Put Spread</extra>',
    ),
    row=1,
    col=2,
)

# Optional dashed underlying path, lightly scaled, to visualize the crash timing.
fig.add_trace(
    go.Scatter(
        x=DATES,
        y=0.12 * (SPOT0 - spot_path),
        mode='lines',
        line=dict(color=underlying_color, width=1.5, dash='dot'),
        name='Crash proxy (scaled)',
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Scaled crash proxy: %{y:.2f}<extra></extra>',
        opacity=0.55,
    ),
    row=1,
    col=2,
)

# Shade the crash window -- PATCH: use add_shape (not add_vrect) due to subplot axis referencing.
# We need to use the correct xref and yref for subplot "x2"/"y2":
fig.add_shape(
    type="rect",
    x0=DATES[CRASH_START],
    x1=DATES[CRASH_END],
    y0=0,
    y1=1,
    xref="x2",
    yref="paper",
    fillcolor='rgba(255, 77, 77, 0.14)',
    line_width=0,
    layer='below',
)

# Horizontal guide at the matched crash protection level.
match_level = float(put_a_value.loc[calibration_date])
# --- PATCH: Use add_shape to draw a horizontal line in the correct subplot
# Use xref="x2" and yref="y2" for the right panel
fig.add_shape(
    type="line",
    x0=DATES[0],
    x1=DATES[-1],
    y0=match_level,
    y1=match_level,
    line=dict(color='rgba(255,255,255,0.35)', width=1, dash='dot'),
    xref="x2",
    yref="y2",
    layer='above',
)

fig.add_annotation(
    x=calibration_date,
    y=match_level,
    xref='x2',
    yref='y2',
    text='Crash-value match point',
    showarrow=True,
    arrowhead=2,
    ax=40,
    ay=-35,
    font=dict(color=off_white, size=11),
    arrowcolor=off_white,
    bgcolor='rgba(30,30,30,0.75)',
    bordercolor='rgba(255,255,255,0.20)',
    borderwidth=1,
)

# ============================================================
# 7. Layout and styling
# ============================================================

fig.update_layout(
    title=dict(
        text=(
            '<b>Put Skew and Put Strategy Carry</b>'
            '<br><sup>'
            'Left: synthetic skewed implied-vol surface | '
            'Right: a rich-skew outright put vs a lower-carry put spread sized '
            'to provide similar crash protection'
            '</sup>'
        ),
        x=0.5,
        font=dict(color=off_white, size=22),
    ),
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=650,
    width=1300,
    margin=dict(t=110, b=60, l=50, r=50),
    legend=dict(
        orientation='h',
        x=0.5,
        xanchor='center',
        y=-0.12,
        yanchor='top',
        font=dict(color=off_white),
    ),
)

# Surface axes.
fig.update_scenes(
    dict(
        xaxis_title='Strike',
        yaxis_title='Days to Expiry',
        zaxis_title='Implied Vol (%)',
        xaxis=dict(
            backgroundcolor='rgba(0,0,0,0)',
            gridcolor='rgba(255,255,255,0.10)',
            tickfont=dict(color=off_white),
            # titlefont=dict(color=off_white),  # <-- REMOVE INVALID PROPERTY
        ),
        yaxis=dict(
            backgroundcolor='rgba(0,0,0,0)',
            gridcolor='rgba(255,255,255,0.10)',
            tickfont=dict(color=off_white),
            # titlefont=dict(color=off_white),  # <-- REMOVE INVALID PROPERTY
        ),
        zaxis=dict(
            backgroundcolor='rgba(0,0,0,0)',
            gridcolor='rgba(255,255,255,0.10)',
            tickfont=dict(color=off_white),
            # titlefont=dict(color=off_white),  # <-- REMOVE INVALID PROPERTY
        ),
        camera=dict(eye=dict(x=1.55, y=1.45, z=0.85)),
    ),
    row=1,
    col=1,
)

# Right-panel axes.
axis_style = dict(
    showgrid=True,
    gridcolor='rgba(255,255,255,0.10)',
    tickfont=dict(color=off_white),
    linecolor=off_white,
    zeroline=False,
    title_font=dict(color=off_white),
)

fig.update_xaxes(
    axis_style,
    row=1,
    col=2,
    title_text='Date',
)
fig.update_yaxes(
    axis_style,
    row=1,
    col=2,
    title_text='Option Strategy Value',
)

for annotation in fig.layout.annotations:
    annotation.font = dict(color=off_white, size=14)

fig.show()
```


---

## 10. Master Volatility with ARCH & GARCH Models (Lecture 47)

`2025 Video Lectures/47. Master Volatility with ARCH & GARCH Models/arch_garch.ipynb` — ARCH/GARCH volatility fitting and forecasting.

```python
import numpy as np
import pandas as pd
import yfinance as yf  # replaces the Interactive Brokers (ibapi) data scripts


def _yf_close(ticker, start, end=None):
    """Daily adjusted close for one ticker from Yahoo Finance, as a Series indexed by date."""
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    s = raw["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = s.dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.name = ticker
    return s


# Writes aapl_volatility_data_2023.csv and aapl_volatility_data_2023_2025.csv
# columns: date, close, returns, ib_realized_vol_30d (name kept for the notebook), implied_vol
_px = _yf_close("AAPL", "2022-10-01", "2026-01-01")
_iv = _yf_close("^VXAPL", "2022-10-01", "2026-01-01") / 100.0   # CBOE Apple VIX = implied-vol proxy
_d = pd.DataFrame({"close": _px})
_d["returns"] = np.log(_d["close"] / _d["close"].shift(1))
_d["ib_realized_vol_30d"] = _d["returns"].rolling(30).std() * np.sqrt(252)
_d = _d.join(_iv.rename("implied_vol"), how="left")
for _name, _a, _b in [("aapl_volatility_data_2023.csv", "2023-01-01", "2023-12-31"),
                      ("aapl_volatility_data_2023_2025.csv", "2023-01-01", "2025-12-31")]:
    _d.loc[_a:_b].rename_axis("date").reset_index().to_csv(_name, index=False)

```

```python
%%html
<style>
/* Overwrite the hard-coded white background for ipywidgets */
.cell-output-ipywidget-background {
    background-color: transparent !important;
}
/* Set widget foreground text and color to match the VS Code dark theme */
:root {
    --jp-widgets-color: var(--vscode-editor-foreground);
    --jp-widgets-font-size: var(--vscode-editor-font-size);
}
</style>
```

```python
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Read the volatility data
df = pd.read_csv('aapl_volatility_data_2023.csv')
df['date'] = pd.to_datetime(df['date'])

# Calculate annualized squared deviations from rolling means
df['sq_dev_7d'] = (df['returns'] - df['returns'].rolling(7).mean())**2 * 252
df['sq_dev_30d'] = (df['returns'] - df['returns'].rolling(30).mean())**2 * 252

# Cut off first 30 days
df = df[30:]

# Create subplots - 1 row, 2 columns
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        'AAPL Returns', 
        'Volatility Measures'
    ),
    horizontal_spacing=0.1
)

# Plot returns
fig.add_trace(
    go.Scatter(
        x=df['date'],
        y=df['returns'],
        mode='lines',
        name='Returns',
        line=dict(color='#FF00FF', width=1.5),  # Neon pink
        showlegend=True
    ),
    row=1, col=1
)

# Plot volatility measures
fig.add_trace(
    go.Scatter(
        x=df['date'],
        y=df['ib_realized_vol_30d'],
        mode='lines',
        name='30-day Realized Vol',
        line=dict(color='#39FF14', width=1.5),  # Neon green
        showlegend=True
    ),
    row=1, col=2
)

fig.add_trace(
    go.Scatter(
        x=df['date'],
        y=np.sqrt(df['sq_dev_7d'].rolling(7).mean()),
        mode='lines',
        name='7-day Rolling Vol',
        line=dict(color='#FF3131', width=1.5),  # Neon red
        showlegend=True
    ),
    row=1, col=2
)

fig.add_trace(
    go.Scatter(
        x=df['date'],
        y=np.sqrt(df['sq_dev_30d'].rolling(30).mean()),
        mode='lines',
        name='30-day Rolling Vol',
        line=dict(color='#00FFFF', width=1.5),  # Neon cyan
        showlegend=True
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    width=1200,
    height=500,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes
for col in [1,2]:
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        title_text="Date",
        row=1, col=col
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        title_text="Returns" if col==1 else "Volatility",
        row=1, col=col
    )

fig.show()
```

```python
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Read the volatility data
df = pd.read_csv('aapl_volatility_data_2023.csv')
df['date'] = pd.to_datetime(df['date'])

# Calculate annualized squared deviations from rolling means
df['sq_dev_7d'] = (df['returns'] - df['returns'].rolling(7).mean())**2 * 252
df['sq_dev_30d'] = (df['returns'] - df['returns'].rolling(30).mean())**2 * 252

# Calculate rolling 30-day correlation
rolling_corr = df['ib_realized_vol_30d'].rolling(30).corr(df['implied_vol'])

# Remove first 30 days of data
df = df[30:]

# Create subplots - 2 rows
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'AAPL Returns', 
        'Volatility Measures',
        'Rolling 30-day Correlation between Realized and Implied Volatility',
        ''
    ),
    horizontal_spacing=0.1,
    vertical_spacing=0.15,
    row_heights=[0.7, 0.3],
    specs=[[{}, {}], [{"colspan": 2}, None]]
)

# Plot returns
fig.add_trace(
    go.Scatter(
        x=df['date'],
        y=df['returns'],
        mode='lines',
        name='Returns',
        line=dict(color='#B026FF', width=1.5), # Neon purple
        showlegend=True
    ),
    row=1, col=1
)

# Plot volatility measures
fig.add_trace(
    go.Scatter(
        x=df['date'],
        y=df['ib_realized_vol_30d'],
        mode='lines',
        name='30-day Realized Vol',
        line=dict(color='#39FF14', width=1.5), # Neon green
        showlegend=True
    ),
    row=1, col=2
)

fig.add_trace(
    go.Scatter(
        x=df['date'],
        y=df['implied_vol'],
        mode='lines',
        name='Implied Vol',
        line=dict(color='#00FFFF', width=1.5), # Neon cyan
        showlegend=True
    ),
    row=1, col=2
)

# Plot rolling correlation
fig.add_trace(
    go.Scatter(
        x=df['date'],
        y=rolling_corr[60:],
        mode='lines',
        name='Rolling Correlation',
        line=dict(color='#FF4500', width=1.5), # Neon orange
        showlegend=True
    ),
    row=2, col=1
)

# Update layout
fig.update_layout(
    width=1200,
    height=800,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes for top row
for col in [1,2]:
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        title_text="Date",
        row=1, col=col
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        title_text="Returns" if col==1 else "Volatility",
        row=1, col=col
    )

# Update axes for bottom row
fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    title_text="Date",
    row=2, col=1
)
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    title_text="Correlation",
    range=[-1, 1],
    row=2, col=1
)

fig.show()
```

```python
# Create lagged implied vol and forward implied vol
df['ivol_lag'] = df['implied_vol']
# Calculate 5-day forward average implied vol
n = 20
df['ivol_fwd'] = df['implied_vol'].rolling(window=n, min_periods=1).mean().shift(-n)

# Remove NaN values
df_reg = df.dropna(subset=['ivol_lag', 'ivol_fwd'])

# Create subplots
fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1)

# First subplot - Regression with y=x line
fig.add_trace(
    go.Scatter(
        x=df_reg['ivol_lag'],
        y=df_reg['ivol_fwd'],
        mode='markers',
        name='Data Points',
        marker=dict(
            color='#39FF14',
            size=8,
            opacity=0.6
        )
    ),
    row=1, col=1
)

# Calculate and add regression line
z = np.polyfit(df_reg['ivol_lag'], df_reg['ivol_fwd'], 1)
p = np.poly1d(z)
x_reg = np.linspace(df_reg['ivol_lag'].min(), df_reg['ivol_lag'].max(), 100)
fig.add_trace(
    go.Scatter(
        x=x_reg,
        y=p(x_reg),
        mode='lines',
        name=f'Regression Line (slope={z[0]:.3f})',
        line=dict(color='#00FFFF', width=2)
    ),
    row=1, col=1
)

# Add y=x baseline to first subplot
fig.add_trace(
    go.Scatter(
        x=x_reg,
        y=x_reg,
        mode='lines',
        name='y=x (Perfect Realization)',
        line=dict(
            color='red',
            width=2,
            dash='dot'
        )
    ),
    row=1, col=1
)

# Find intersection point of regression line and y=x
intersection_x = -z[1]/(z[0]-1)
intersection_y = p(intersection_x)

# Split data into low and high vol regimes
df_low = df_reg[df_reg['ivol_lag'] <= intersection_x]
df_high = df_reg[df_reg['ivol_lag'] > intersection_x]

# Second subplot - regime specific regressions
fig.add_trace(
    go.Scatter(
        x=df_low['ivol_lag'],
        y=df_low['ivol_fwd'],
        mode='markers',
        name='Low Vol Regime',
        marker=dict(
            color='#39FF14',
            size=8,
            opacity=0.6
        )
    ),
    row=1, col=2
)

fig.add_trace(
    go.Scatter(
        x=df_high['ivol_lag'],
        y=df_high['ivol_fwd'],
        mode='markers',
        name='High Vol Regime',
        marker=dict(
            color='#FF10F0',
            size=8,
            opacity=0.6
        )
    ),
    row=1, col=2
)

# Add regime specific regression lines
z_low = np.polyfit(df_low['ivol_lag'], df_low['ivol_fwd'], 1)
p_low = np.poly1d(z_low)
x_low = np.linspace(df_low['ivol_lag'].min(), df_low['ivol_lag'].max(), 100)

z_high = np.polyfit(df_high['ivol_lag'], df_high['ivol_fwd'], 1)
p_high = np.poly1d(z_high)
x_high = np.linspace(df_high['ivol_lag'].min(), df_high['ivol_lag'].max(), 100)

fig.add_trace(
    go.Scatter(
        x=x_low,
        y=p_low(x_low),
        mode='lines',
        name=f'Low Vol Regression (slope={z_low[0]:.3f})',
        line=dict(color='#00FFFF', width=2)
    ),
    row=1, col=2
)

fig.add_trace(
    go.Scatter(
        x=x_high,
        y=p_high(x_high),
        mode='lines',
        name=f'High Vol Regression (slope={z_high[0]:.3f})',
        line=dict(color='#FF00FF', width=2)
    ),
    row=1, col=2
)

# Add vertical line at regime split point
fig.add_vline(
    x=intersection_x,
    line_dash="dash",
    line_color="red",
    row=1,
    col=2
)

# Update layout
fig.update_layout(
    width=1600,
    height=600,
    title='Implied Vol Analysis with Regime Split',
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes for both subplots
for i in [1, 2]:
    fig.update_xaxes(
        title_text='Current Implied Volatility',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        row=1, col=i
    )
    fig.update_yaxes(
        title_text='5-Day Forward Average Implied Volatility',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        row=1, col=i
    )

fig.show()
```

```python
# Import necessary packages
import numpy as np
from scipy.stats import gaussian_kde
import plotly.graph_objects as go

# Calculate returns distribution statistics
returns_mean = df['returns'].mean()
returns_std = df['returns'].std()
returns_kurt = df['returns'].kurtosis() + 3  # Convert from excess kurtosis to regular kurtosis

# Generate normal distribution with same mean and std
normal_dist = np.random.normal(returns_mean, returns_std, 100000)
normal_kurt = 3  # Normal distribution has kurtosis of 3

# Create KDE plots
kde_returns = gaussian_kde(df['returns'])
kde_normal = gaussian_kde(normal_dist)

# Create evaluation points
x_range = np.linspace(min(df['returns'].min(), normal_dist.min()), 
                      max(df['returns'].max(), normal_dist.max()), 
                      1000)

# Create figure
fig = go.Figure()

# Add returns KDE
fig.add_trace(
    go.Scatter(
        x=x_range,
        y=kde_returns(x_range),
        name='Returns Distribution',
        line=dict(color='#39FF14', width=2),
        mode='lines'
    )
)

# Add normal KDE
fig.add_trace(
    go.Scatter(
        x=x_range,
        y=kde_normal(x_range),
        name='Normal Distribution',
        line=dict(color='#FF10F0', width=2),
        mode='lines'
    )
)

# Update layout
fig.update_layout(
    width=1200,
    height=500,
    title=f'Returns Distribution vs Normal<br>Returns Kurtosis: {returns_kurt:.2f}, Normal Kurtosis: {normal_kurt:.2f}',
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes
fig.update_xaxes(
    title_text='Returns',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.update_yaxes(
    title_text='Density',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.show()
```

```python
# Calculate empirical and normal 0.05% VaR
empirical_var = np.percentile(df['returns'], 0.05)
normal_var = np.percentile(normal_dist, 0.05)

# Calculate average loss beyond VaR (Expected Shortfall)
empirical_es = df['returns'][df['returns'] <= empirical_var].mean()
normal_es = normal_dist[normal_dist <= normal_var].mean()

# Calculate percent differences
var_diff_pct = (normal_var - empirical_var) / abs(empirical_var) * 100
es_diff_pct = (normal_es - empirical_es) / abs(empirical_es) * 100

# Create figure
fig = go.Figure()

# Add returns distribution with gradient fill
fig.add_trace(
    go.Scatter(
        x=x_range,
        y=kde_returns(x_range),
        line=dict(color='#39FF14', width=2.5),
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(57, 255, 20, 0.2)'
    )
)

# Add normal distribution with gradient fill
fig.add_trace(
    go.Scatter(
        x=x_range,
        y=kde_normal(x_range),
        line=dict(color='#FF10F0', width=2.5),
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(255, 16, 240, 0.1)'
    )
)

# Add VaR lines with values
fig.add_vline(x=empirical_var, line_dash="dash", line_color="#39FF14",
              annotation=dict(
                  text=f"{empirical_var:.3f}<br>{empirical_es:.3f}",
                  x=empirical_var,
                  y=1.15,
                  yref="paper",
                  showarrow=False,
                  font=dict(size=12)
              ))
fig.add_vline(x=normal_var, line_dash="dash", line_color="#FF10F0",
              annotation=dict(
                  text=f"{normal_var:.3f}<br>{normal_es:.3f}",
                  x=normal_var,
                  y=0.85,
                  yref="paper",
                  showarrow=False,
                  font=dict(size=12)
              ))

# Update layout with enhanced styling
fig.update_layout(
    width=900,
    height=600,
    title=dict(
        text=f'Returns vs Normal Distribution<br>VaR Difference: {var_diff_pct:.1f}% | ES Difference: {es_diff_pct:.1f}%',
        x=0.5,
        y=0.95
    ),
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white', size=12)
)

# Update axes with enhanced styling
fig.update_xaxes(
    title_text='Returns',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1.5,
    zerolinecolor='rgba(128,128,128,0.5)',
    title_font=dict(size=14)
)

fig.update_yaxes(
    title_text='Density',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1.5,
    zerolinecolor='rgba(128,128,128,0.5)',
    title_font=dict(size=14)
)

fig.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# Create subplots to show stylized facts
fig = make_subplots(rows=1, cols=2, 
                    subplot_titles=('Volatility Mean Reversion',
                                  'Leverage Effect'),
                    horizontal_spacing=0.1)

# Plot 1: Volatility Mean Reversion
rolling_vol = df['returns'].rolling(window=20).std() * np.sqrt(252)
long_term_vol = rolling_vol.mean()

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=rolling_vol,
        mode='lines',
        name='20-day Rolling Volatility',
        line=dict(color='#00FFFF', width=1.5)
    ),
    row=1, col=1
)

fig.add_hline(
    y=long_term_vol,
    line_dash="dash",
    line_color="red",
    annotation_text="Long-term Average",
    row=1, col=1
)

# Plot 2: Leverage Effect
returns_lag = df['returns'].shift(1)
vol_change = rolling_vol.diff()

neg_returns = returns_lag[returns_lag < 0]
pos_returns = returns_lag[returns_lag > 0]
neg_vol_change = vol_change[returns_lag < 0]
pos_vol_change = vol_change[returns_lag > 0]

# Calculate average vol changes
avg_neg_change = neg_vol_change.mean()
avg_pos_change = pos_vol_change.mean()

fig.add_trace(
    go.Scatter(
        x=neg_returns,
        y=neg_vol_change,
        mode='markers',
        name=f'After Negative Returns (avg: {avg_neg_change:.4f})',
        marker=dict(color='#FF10F0', size=6, opacity=0.6)
    ),
    row=1, col=2
)

fig.add_trace(
    go.Scatter(
        x=pos_returns,
        y=pos_vol_change,
        mode='markers',
        name=f'After Positive Returns (avg: {avg_pos_change:.4f})',
        marker=dict(color='#39FF14', size=6, opacity=0.6)
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    width=1600,
    height=600,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white', size=12)
)

# Update axes styling for all subplots
for i in range(1, 2):
    for j in range(1, 3):
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=True,
            zerolinewidth=1.5,
            zerolinecolor='rgba(128,128,128,0.5)',
            row=i, col=j
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=True,
            zerolinewidth=1.5,
            zerolinecolor='rgba(128,128,128,0.5)',
            row=i, col=j
        )

# Add specific axis labels
fig.update_xaxes(title_text="Date", row=1, col=1)
fig.update_xaxes(title_text="Previous Return", row=1, col=2)

fig.update_yaxes(title_text="Volatility", row=1, col=1)
fig.update_yaxes(title_text="Volatility Change", row=1, col=2)

fig.show()
```

```python
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm

# -----------------------
# Load data and prepare returns
# -----------------------
df = pd.read_csv('aapl_volatility_data_2023_2025.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
df = df[30:]

# Realized Variance (Daily, annualized)
realized_var = (df['returns']**2) * 252  

# -----------------------
# Train/Test Split
# -----------------------
train_end_idx = int(len(df) * 0.7)  # 70% training
train_data = df.iloc[:train_end_idx]
test_data = df.iloc[train_end_idx:]
train_end = train_data.index[-1]

# -----------------------
# EWMA Forecast Function
# -----------------------
def ewma_forecast(returns, lam):
    ewma_var = np.zeros(len(returns))
    ewma_var[0] = returns.var()  # initialize with unconditional variance
    for t in range(1, len(returns)):
        ewma_var[t] = (
            lam * ewma_var[t-1] + (1-lam) * returns.iloc[t-1]**2
        )
    return ewma_var * 252  # annualize

# -----------------------
# Grid Search for Optimal λ (train set only)
# -----------------------
lambda_grid = np.linspace(0.01, 0.90, 100)
best_rmse = float('inf')
best_lambda = None

for lam in lambda_grid:
    ewma_train = ewma_forecast(train_data['returns'], lam)
    rmse = np.sqrt(((ewma_train - realized_var[train_data.index])**2).mean())
    if rmse < best_rmse:
        best_rmse = rmse
        best_lambda = lam

print(f"Best λ found: {best_lambda:.3f} (Train RMSE={best_rmse:.6f})")

# -----------------------
# Out-of-Sample Forecast using Best λ
# -----------------------
ewma_all = ewma_forecast(df['returns'], best_lambda)
ewma_forecast_var = pd.Series(ewma_all, index=df.index)

aligned = pd.DataFrame({
    'realized_var': realized_var,
    'ewma_var': ewma_forecast_var
})

# OOS performance
oos_data = aligned[train_end:]
mse = ((oos_data['ewma_var'] - oos_data['realized_var'])**2).mean()
rmse = np.sqrt(mse)
mae = (oos_data['ewma_var'] - oos_data['realized_var']).abs().mean()

X = sm.add_constant(oos_data['ewma_var'])
y = oos_data['realized_var']
model = sm.OLS(y, X).fit()
r2 = model.rsquared

# -----------------------
# Visualization
# -----------------------
fig = go.Figure()

# Realized variance
fig.add_trace(
    go.Scatter(
        x=aligned.index,
        y=aligned['realized_var'],
        mode='lines',
        name='Realized Variance (Daily)',
        line=dict(color='#BC13FE', width=1.5)
    )
)

# EWMA variance forecast
fig.add_trace(
    go.Scatter(
        x=aligned.index,
        y=aligned['ewma_var'],
        mode='lines',
        name=f"EWMA Forecast (λ={best_lambda:.3f})",
        line=dict(color='#00FFFF', width=1.5)
    )
)

# Train/Test split marker
fig.add_shape(
    type='line',
    x0=train_end,
    x1=train_end,
    y0=0,
    y1=1,
    yref='paper',
    line=dict(color='red', dash='dash'),
)

fig.add_annotation(
    x=train_end,
    y=1,
    yref='paper',
    text='Train/Test Split',
    showarrow=False,
    textangle=-90,
)

fig.update_layout(
    title="EWMA Forecast vs Realized Variance (Daily, Annualized)",
    width=1200,
    height=500,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    title_text="Date"
)
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    title_text="Variance"
)
fig.show()

print("\nOut-of-Sample Performance Metrics:")
print(f"RMSE (Variance units): {rmse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"R-squared: {r2:.2%}")
```

```python
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
from itertools import product

# -----------------------
# Load data
# -----------------------
df = pd.read_csv('aapl_volatility_data_2023_2025.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# Realized variance (annualized)
realized_var = (df['returns']**2) * 252

# -----------------------
# ARCH(1) Forecast Function
# -----------------------
def arch1_forecast(returns, omega, alpha):
    """Generate ARCH(1) conditional variance forecasts (annualized)."""
    var = np.zeros(len(returns))
    var[0] = np.var(returns)  # unconditional variance init
    for t in range(1, len(returns)):
        var[t] = omega + alpha * returns.iloc[t-1]**2
    return var * 252

# -----------------------
# Candidate grids
# -----------------------
window_candidates = [50, 100, 250, 500, 750]  # you can add more
omega_grid = np.linspace(0.00001, 0.001, 10)
alpha_grid = np.linspace(0.05, 0.95, 10)

best_rmse = float("inf")
best_params = None
best_forecast = None
best_train_end = None

# -----------------------
# Grid Search over (window, omega, alpha)
# -----------------------
for window in window_candidates:
    if window >= len(df):
        continue  # skip if window > data
    
    # define train/test split based on window size
    train_data = df.iloc[:window]
    test_data = df.iloc[window:]
    train_end = train_data.index[-1]

    for omega, alpha in product(omega_grid, alpha_grid):
        # Forecast on training data
        var_train = arch1_forecast(train_data['returns'], omega, alpha)

        # Out-of-sample forecast
        var_all = pd.Series(index=df.index, dtype=float)
        var_all.iloc[:window] = var_train
        for t in range(window, len(df)):
            var_all.iloc[t] = (omega + alpha * df['returns'].iloc[t-1]**2) * 252

        # Compute OOS RMSE
        oos_data = pd.DataFrame({
            'realized': realized_var[train_end:],
            'arch_var': var_all[train_end:]
        }).dropna()
        rmse = np.sqrt(((oos_data['arch_var'] - oos_data['realized'])**2).mean())

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = (window, omega, alpha)
            best_forecast = var_all
            best_train_end = train_end

# -----------------------
# Report Best Params
# -----------------------
print(f"Best parameters: window={best_params[0]}, omega={best_params[1]:.6f}, alpha={best_params[2]:.6f}")
print(f"Out-of-sample RMSE: {best_rmse:.6f}")

# -----------------------
# Regression (OOS only)
# -----------------------
aligned = pd.DataFrame({
    'realized_var': realized_var,
    'arch_var': best_forecast
})

oos_data = aligned[best_train_end:]
X = sm.add_constant(oos_data['arch_var'])
y = oos_data['realized_var']
model = sm.OLS(y, X).fit()
r2 = model.rsquared

print(f"Out-of-sample R²: {r2:.2%}")

# -----------------------
# Visualization
# -----------------------
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=aligned.index,
        y=aligned['realized_var'],
        mode='lines',
        name='Realized Variance',
        line=dict(color='#BC13FE', width=1.5)
    )
)

fig.add_trace(
    go.Scatter(
        x=aligned.index,
        y=aligned['arch_var'],
        mode='lines',
        name=f'ARCH(1) Forecast (window={best_params[0]})',
        line=dict(color='#00FF7F', width=1.5)
    )
)

# Add vertical line for train/test split
fig.add_shape(
    type='line',
    x0=best_train_end,
    x1=best_train_end,
    y0=0,
    y1=1,
    yref='paper',
    line=dict(color='red', dash='dash'),
)

fig.add_annotation(
    x=best_train_end,
    y=1,
    yref='paper',
    text='Train/Test Split',
    showarrow=False,
    textangle=-90,
)

fig.update_layout(
    title="ARCH(1) Forecast vs Realized Variance (Annualized)",
    width=1200,
    height=500,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    title_text="Date"
)
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    title_text="Variance"
)

fig.show()
```

```python
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
from itertools import product

# -----------------------
# Load data
# -----------------------
df = pd.read_csv('aapl_volatility_data_2023_2025.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
df = df[30:]

# Realized variance (annualized)
realized_var = (df['returns']**2) * 252

# -----------------------
# GARCH(1,1) Forecast Function
# -----------------------
def garch11_forecast(returns, omega, alpha, beta):
    """Generate GARCH(1,1) conditional variance forecasts (annualized)."""
    var = np.zeros(len(returns))
    var[0] = np.var(returns)  # unconditional variance init
    for t in range(1, len(returns)):
        var[t] = omega + alpha * returns.iloc[t-1]**2 + beta * var[t-1]
    return var * 252

# -----------------------
# Candidate grids
# -----------------------
window_candidates = [50, 100, 250, 500, 750]
omega_grid = np.linspace(0.00001, 0.001, 5)
alpha_grid = np.linspace(0.05, 0.95, 5)
beta_grid = np.linspace(0.05, 0.95, 5)

best_rmse = float("inf")
best_params = None
best_forecast = None
best_train_end = None

# -----------------------
# Grid Search over (window, omega, alpha, beta)
# -----------------------
for window in window_candidates:
    if window >= len(df):
        continue
    
    train_data = df.iloc[:window]
    test_data = df.iloc[window:]
    train_end = train_data.index[-1]

    for omega, alpha, beta in product(omega_grid, alpha_grid, beta_grid):
        if alpha + beta >= 1:  
            continue  # ensure stationarity

        # Forecast on training data
        var_train = garch11_forecast(train_data['returns'], omega, alpha, beta)

        # Out-of-sample forecast
        var_all = pd.Series(index=df.index, dtype=float)
        var_all.iloc[:window] = var_train
        for t in range(window, len(df)):
            var_all.iloc[t] = (omega +
                               alpha * df['returns'].iloc[t-1]**2 +
                               beta * var_all.iloc[t-1]/252) * 252

        # Compute OOS RMSE
        oos_data = pd.DataFrame({
            'realized': realized_var[train_end:],
            'garch_var': var_all[train_end:]
        }).dropna()
        rmse = np.sqrt(((oos_data['garch_var'] - oos_data['realized'])**2).mean())

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = (window, omega, alpha, beta)
            best_forecast = var_all
            best_train_end = train_end

# -----------------------
# Report Best Params
# -----------------------
print(f"Best parameters: window={best_params[0]}, "
      f"omega={best_params[1]:.6f}, alpha={best_params[2]:.6f}, beta={best_params[3]:.6f}")
print(f"Out-of-sample RMSE: {best_rmse:.6f}")

# -----------------------
# Regression (OOS only)
# -----------------------
aligned = pd.DataFrame({
    'realized_var': realized_var,
    'garch_var': best_forecast
})

oos_data = aligned[best_train_end:]
X = sm.add_constant(oos_data['garch_var'])
y = oos_data['realized_var']
model = sm.OLS(y, X).fit()
r2 = model.rsquared

print(f"Out-of-sample R²: {r2:.2%}")

# -----------------------
# Visualization
# -----------------------
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=aligned.index,
        y=aligned['realized_var'],
        mode='lines',
        name='Realized Variance',
        line=dict(color='#BC13FE', width=1.5)
    )
)

fig.add_trace(
    go.Scatter(
        x=aligned.index,
        y=aligned['garch_var'],
        mode='lines',
        name=f'GARCH(1,1) Forecast (window={best_params[0]})',
        line=dict(color='#00FF7F', width=1.5)
    )
)

fig.add_shape(
    type='line',
    x0=best_train_end,
    x1=best_train_end,
    y0=0,
    y1=1,
    yref='paper',
    line=dict(color='red', dash='dash'),
)

fig.add_annotation(
    x=best_train_end,
    y=1,
    yref='paper',
    text='Train/Test Split',
    showarrow=False,
    textangle=-90,
)

fig.update_layout(
    title="GARCH(1,1) Forecast vs Realized Variance (Annualized)",
    width=1200,
    height=500,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    title_text="Date"
)
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    title_text="Variance"
)

fig.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

# -----------------------
# Parameters
# -----------------------
alpha = 0.05   # 5% VaR
z_alpha = norm.ppf(alpha)  # ≈ -1.645

# -----------------------
# Naïve (parametric) VaR
# -----------------------
# Use unconditional volatility (std of returns)
sigma_naive = df['returns'].std()
sigma_naive_daily = sigma_naive / np.sqrt(252)   # convert annual to daily if needed

VaR_naive = z_alpha * sigma_naive_daily

# -----------------------
# GARCH(1,1) VaR
# -----------------------
sigma_daily = np.sqrt(best_forecast / 252)
VaR_garch = z_alpha * sigma_daily

# -----------------------
# Create DataFrame
# -----------------------
var_df = pd.DataFrame({
    'returns': df['returns'],
    'VaR_Naive': VaR_naive,
    'VaR_GARCH': VaR_garch
}).dropna()

# -----------------------
# Exceedances
# -----------------------
exceed_naive = var_df[var_df['returns'] < var_df['VaR_Naive']]
exceed_garch = var_df[var_df['returns'] < var_df['VaR_GARCH']]

prop_exceed_naive = len(exceed_naive) / len(var_df)
prop_exceed_garch = len(exceed_garch) / len(var_df)

print(f"Proportion of exceedances (Naïve): {prop_exceed_naive:.2%}")
print(f"Proportion of exceedances (GARCH): {prop_exceed_garch:.2%}")

# -----------------------
# Visualization
# -----------------------
fig = go.Figure()

# Portfolio returns
fig.add_trace(
    go.Scatter(
        x=var_df.index,
        y=var_df['returns'],
        mode='lines',
        name='Portfolio Returns',
        line=dict(color='white', width=1)
    )
)

# Naïve VaR
fig.add_trace(
    go.Scatter(
        x=var_df.index,
        y=var_df['VaR_Naive'],
        mode='lines',
        name=f'5% Naïve VaR (Exceed: {prop_exceed_naive:.2%})',
        line=dict(color='orange', width=2, dash='dot')
    )
)

# GARCH VaR
fig.add_trace(
    go.Scatter(
        x=var_df.index,
        y=var_df['VaR_GARCH'],
        mode='lines',
        name=f'5% GARCH(1,1) VaR (Exceed: {prop_exceed_garch:.2%})',
        line=dict(color='#FF3131', width=2, dash='dash')
    )
)

# Highlight exceedances (Naïve)
fig.add_trace(
    go.Scatter(
        x=exceed_naive.index,
        y=exceed_naive['returns'],
        mode='markers',
        name='Naïve Exceedances',
        marker=dict(color='orange', size=7, symbol='x')
    )
)

# Highlight exceedances (GARCH)
fig.add_trace(
    go.Scatter(
        x=exceed_garch.index,
        y=exceed_garch['returns'],
        mode='markers',
        name='GARCH Exceedances',
        marker=dict(color='red', size=7, symbol='cross')
    )
)

# Layout
fig.update_layout(
    title="1-Day 5% VaR: Naïve vs GARCH(1,1)",
    width=1200,
    height=500,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    title_text="Date"
)
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    title_text="Return / VaR"
)

fig.show()
```


---

## 11. Kalman Filter for Quant Finance (Lecture 92)

`2026 Video Lectures/92. Kalman Filter for Quant Finance/kffqf.ipynb` — Kalman filter for state estimation.

```python
%%html
<style>
/* Overwrite the hard-coded white background for ipywidgets */
.cell-output-ipywidget-background {
    background-color: transparent !important;
}
/* Set widget foreground text and color to match the VS Code dark theme */
:root {
    --jp-widgets-color: var(--vscode-editor-foreground);
    --jp-widgets-font-size: var(--vscode-editor-font-size);
}
</style>
```

```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Data Generation ---
np.random.seed(42)
n_points = 60
# Generate 2D X data (not a time series)
X = np.linspace(-5, 5, n_points)

# Add some noise to make it realistic
noise_lin = np.random.normal(0, 3, n_points)
noise_quad = np.random.normal(0, 4, n_points)

# True Data Generating Processes
# Left: Linear
Y_lin = 4 * X + 15 + noise_lin
# Right: Quadratic (Non-linear)
Y_quad = 1.5 * X**2 - 2 * X + 10 + noise_quad

# --- Precompute Models ---
# We will animate the addition of points. We need at least 2 points to fit a line.
min_points = 2
lin_fits = []
quad_fits = []

# X coordinates for drawing the regression line across the whole plot
x_grid = np.array([-6, 6])

for t in range(min_points, n_points + 1):
    x_curr = X[:t]
    y_lin_curr = Y_lin[:t]
    y_quad_curr = Y_quad[:t]
    
    # Model 1: Linear fit on Linear Data
    slope_lin, int_lin = np.polyfit(x_curr, y_lin_curr, 1)
    line_lin = int_lin + slope_lin * x_grid
    lin_fits.append(line_lin)
    
    # Model 2: Linear fit on Quadratic Data (Modeling Issue)
    slope_quad, int_quad = np.polyfit(x_curr, y_quad_curr, 1)
    line_quad = int_quad + slope_quad * x_grid
    quad_fits.append(line_quad)

# --- Graphics Setup ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "<b>Appropriate Model Selection</b><br>Linear fit captures the dynamics", 
        "<b>Modeling Issue (Underfitting)</b><br>Linear model cannot capture quadratic dynamics"
    ),
    vertical_spacing=0.15, horizontal_spacing=0.08
)

c_lin = "#00FFCC"      # Teal
c_bad = "#FF3366"      # Pinkish red
c_pts = "rgba(255, 255, 255, 0.8)"

# Dynamic Traces: Scatter Points up to t=2 (Initial State)
fig.add_trace(go.Scatter(x=X[:min_points], y=Y_lin[:min_points], mode='markers', 
                         marker=dict(color='white', size=8, line=dict(color='black', width=1)), 
                         name='Linear Data'), row=1, col=1) # Trace 0

fig.add_trace(go.Scatter(x=X[:min_points], y=Y_quad[:min_points], mode='markers', 
                         marker=dict(color='white', size=8, line=dict(color='black', width=1)), 
                         name='Quadratic Data'), row=1, col=2) # Trace 1

# Dynamic Traces: Regression Lines (Initial State)
fig.add_trace(go.Scatter(x=x_grid, y=lin_fits[0], mode='lines', 
                         line=dict(color=c_lin, width=4), name='Valid Fit'), row=1, col=1) # Trace 2

fig.add_trace(go.Scatter(x=x_grid, y=quad_fits[0], mode='lines', 
                         line=dict(color=c_bad, width=4), name='Misspecified Fit'), row=1, col=2) # Trace 3


# --- Animation Logic ---
frames = []
slider_steps = []

for i, t in enumerate(range(min_points, n_points + 1)):
    frame_name = f"f{t}"
    
    frames.append(go.Frame(
        data=[
            go.Scatter(x=X[:t], y=Y_lin[:t]),      # 0: Update scatter left
            go.Scatter(x=X[:t], y=Y_quad[:t]),     # 1: Update scatter right
            go.Scatter(x=x_grid, y=lin_fits[i]),   # 2: Update line left
            go.Scatter(x=x_grid, y=quad_fits[i])   # 3: Update line right
        ],
        traces=[0, 1, 2, 3],
        name=frame_name
    ))
    
    step = {
        "args": [[frame_name], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
        "label": f"n={t}",
        "method": "animate"
    }
    slider_steps.append(step)

fig.frames = frames

# --- Layout & Styling ---
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=600, width=1100,
    margin=dict(t=80, b=100, l=60, r=60),
    showlegend=False,
    
    updatemenus=[{
        "type": "buttons",
        "showactive": False,
        "x": 0.05, "y": -0.20,
        "pad": {"r": 10, "t": 30},
        "bgcolor": "#333",
        "font": {"color": "white"},
        "bordercolor": "#555",
        "borderwidth": 1,
        "buttons": [
            {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 100, "redraw": False}, "fromcurrent": True}]},
            {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "fromcurrent": True}]}
        ]
    }],
    
    sliders=[{
        "active": 0,
        "yanchor": "top", "xanchor": "left",
        "currentvalue": {"font": {"size": 16, "color": "white"}, "prefix": "Data Points: ", "visible": True},
        "transition": {"duration": 0, "easing": "linear"},
        "pad": {"b": 10, "t": 50},
        "len": 0.8, "x": 0.15, "y": -0.20,
        "steps": slider_steps,
        "bgcolor": "#333",
        "font": {"color": "white"}
    }]
)

# Fix axis ranges so the plot is stable during the animation
ax_opts = dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', zeroline=False)

fig.update_xaxes(range=[-6, 6], title="Feature (X)", **ax_opts)

# Set Y-axis ranges based on the generated data
fig.update_yaxes(range=[min(Y_lin)-10, max(Y_lin)+10], title="Target (Y)", row=1, col=1, **ax_opts)
fig.update_yaxes(range=[min(Y_quad)-10, max(Y_quad)+10], title="Target (Y)", row=1, col=2, **ax_opts)

fig.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# --- 1. Load or Generate Data ---
# (Comment this block out and use your real vix.csv)
np.random.seed(42)
n_days = 150
dates = pd.date_range('2024-01-01', periods=n_days)
mock_vix = np.zeros(n_days)
mock_vix[0] = 13.0
for i in range(1, n_days):
    mock_vix[i] = mock_vix[i-1] + 0.08 * (15.0 - mock_vix[i-1]) + np.random.normal(0, 1.5)
    if i == 60: mock_vix[i] += 25.0 # Simulate massive Volatility Spike
pd.DataFrame({'Date': dates, 'SPY': 500, 'VIX': mock_vix}).to_csv('vix.csv', index=False)
# ---------------------------------

df = pd.read_csv('vix.csv')
dates = pd.to_datetime(df['Date'])
vix = df['VIX'].values
n_steps = len(vix)

# --- 2. Rolling Model Setup ---
W = 20  # Lookback window to fit the model parameters
H = 60  # EXTREME Forecast horizon (days ahead) to show terminal behavior

lin_forecasts, lin_stdevs = [], []
mr_forecasts, mr_stdevs = [], []

for t in range(n_steps):
    if t < W:
        # Not enough data to fit, flatline
        lin_forecasts.append(np.ones(H) * vix[t])
        lin_stdevs.append(np.ones(H))
        mr_forecasts.append(np.ones(H) * vix[t])
        mr_stdevs.append(np.ones(H))
        continue
        
    window_vix = vix[t-W:t]
    
    # Model A: Arbitrary Linear Trend (Naive)
    x = np.arange(W)
    slope, intercept = np.polyfit(x, window_vix, 1)
    std_lin = np.std(window_vix - (intercept + slope * x))
    
    # Linear projection straight into the abyss
    f_lin = intercept + slope * (W + np.arange(H))
    s_lin = std_lin * np.sqrt(np.arange(1, H+1)) 
    
    lin_forecasts.append(f_lin)
    lin_stdevs.append(s_lin)
    
    # Model B: Economic Mean Reversion (AR1 / Discrete Ornstein-Uhlenbeck)
    dV = np.diff(window_vix)
    V_lag = window_vix[:-1]
    
    if len(np.unique(V_lag)) > 1:
        b, a = np.polyfit(V_lag, dV, 1)
    else:
        b, a = 0, 0
        
    std_mr = np.std(dV - (a + b * V_lag)) if len(dV) > 0 else 1.0
    
    # Bound parameters to ensure mean reversion mathematically
    if b >= 0:
        b = -0.05
        a = 0.05 * np.mean(window_vix)
        
    f_mr = np.zeros(H)
    s_mr = np.zeros(H)
    curr_v = vix[t-1]
    curr_var = 0
    
    # Project forward curving to terminal theory
    for h in range(H):
        curr_v = a + (1+b) * curr_v
        f_mr[h] = curr_v
        curr_var += std_mr**2 * (1+b)**(2*h)
        s_mr[h] = np.sqrt(curr_var)
        
    mr_forecasts.append(f_mr)
    mr_stdevs.append(s_mr)

# --- 3. Graphics Setup ---
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "<b>Linear Model Forecast</b><br>Shoots to Infinity/Negative", 
        "<b>Mean Reversion Forecast</b><br>Roots back to Terminal Theory",
        "Terminal Forecast Distribution (Linear, t+H)", 
        "Terminal Forecast Distribution (Mean Reversion, t+H)"
    ),
    vertical_spacing=0.15, horizontal_spacing=0.08
)

c_lin = "#FF3366"      # Pink/Red
c_mr = "#00FFCC"       # Teal
c_transparent = "rgba(0,0,0,0)"

# Static Traces: True VIX
fig.add_trace(go.Scatter(x=dates, y=vix, mode='lines', line=dict(color='gray', width=1, dash='dash')), row=1, col=1)
fig.add_trace(go.Scatter(x=dates, y=vix, mode='lines', line=dict(color='gray', width=1, dash='dash')), row=1, col=2)

# Dynamic Traces: Historical Line up to t
fig.add_trace(go.Scatter(x=[dates[0]], y=[vix[0]], mode='lines', line=dict(color='white', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=[dates[0]], y=[vix[0]], mode='lines', line=dict(color='white', width=2)), row=1, col=2)

# Dynamic Traces: Forecast Paths
fig.add_trace(go.Scatter(x=[dates[0]], y=[vix[0]], mode='lines', line=dict(color=c_lin, width=3)), row=1, col=1)
fig.add_trace(go.Scatter(x=[dates[0]], y=[vix[0]], mode='lines', line=dict(color=c_mr, width=3)), row=1, col=2)

# Dynamic Traces: Densities at t+H
x_grid = np.linspace(-40, 120, 500) 
y_zero = np.zeros_like(x_grid)

fig.add_trace(go.Scatter(x=x_grid, y=y_zero, fill='tozeroy', line=dict(color=c_lin)), row=2, col=1)
fig.add_trace(go.Scatter(x=x_grid, y=y_zero, fill='tozeroy', line=dict(color=c_mr)), row=2, col=2)

# --- 4. Animation Logic ---
frames = []
slider_steps = []
steps_to_anim = range(W, n_steps - 10, 2) 

# Create the future coordinates for the forecast arrays to map to
extended_dates = pd.date_range(dates[0], periods=n_days + H)

for t in steps_to_anim:
    frame_name = f"f{t}"
    
    curr_dates = dates[:t]
    curr_vix = vix[:t]
    forecast_dates = extended_dates[t:t+H]
    
    # Generate Probability Density Functions (PDFs) based on the absolute terminal point
    pdf_lin = norm.pdf(x_grid, lin_forecasts[t][-1], lin_stdevs[t][-1] + 1e-6)
    pdf_mr = norm.pdf(x_grid, mr_forecasts[t][-1], mr_stdevs[t][-1] + 1e-6)

    frames.append(go.Frame(
        data=[
            go.Scatter(x=curr_dates, y=curr_vix),                 # Historic Line L
            go.Scatter(x=curr_dates, y=curr_vix),                 # Historic Line R
            go.Scatter(x=forecast_dates, y=lin_forecasts[t]),     # Forecast Path L
            go.Scatter(x=forecast_dates, y=mr_forecasts[t]),      # Forecast Path R
            go.Scatter(y=pdf_lin),                                # PDF L
            go.Scatter(y=pdf_mr),                                 # PDF R
        ],
        traces=[2, 3, 4, 5, 6, 7],
        name=frame_name
    ))
    
    step = {
        "args": [[frame_name], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
        "label": dates[t].strftime('%Y-%m-%d'),
        "method": "animate"
    }
    slider_steps.append(step)

fig.frames = frames

# --- 5. Layout & Styling ---
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=800, width=1000,
    margin=dict(t=80, b=100, l=60, r=60),
    showlegend=False,
    
    updatemenus=[{
        "type": "buttons", "showactive": False,
        "x": 0.05, "y": -0.15, "pad": {"r": 10, "t": 30},
        "bgcolor": "#333", "font": {"color": "white"},
        "bordercolor": "#555", "borderwidth": 1,
        "buttons": [
            {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 50, "redraw": False}, "fromcurrent": True}]},
            {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "fromcurrent": True}]}
        ]
    }],
    
    sliders=[{
        "active": 0, "yanchor": "top", "xanchor": "left",
        "currentvalue": {"font": {"size": 16, "color": "white"}, "prefix": "Date: ", "visible": True},
        "transition": {"duration": 0, "easing": "linear"},
        "pad": {"b": 10, "t": 50},
        "len": 0.8, "x": 0.15, "y": -0.15,
        "steps": slider_steps, "bgcolor": "#333", "font": {"color": "white"}
    }]
)

ax_opts = dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', zeroline=False)

vix_range = [-10, 100] 

fig.update_yaxes(range=vix_range, title="VIX Index", row=1, col=1, **ax_opts)
fig.update_yaxes(range=vix_range, title="VIX Index", row=1, col=2, **ax_opts)

# STRICTLY LOCK X-AXIS to the end of the time series dates
fig.update_xaxes(range=[dates.iloc[0], dates.iloc[-1]], row=1, col=1, **ax_opts)
fig.update_xaxes(range=[dates.iloc[0], dates.iloc[-1]], row=1, col=2, **ax_opts)

# Density Y axes (shrunk scale slightly so flattened curves are visible)
fig.update_yaxes(range=[0, 0.15], title="Probability Density", row=2, col=1, **ax_opts)
fig.update_yaxes(range=[0, 0.15], title="Probability Density", row=2, col=2, **ax_opts)

# Broaden density X axes to fit linear extremes
fig.update_xaxes(range=[-40, 120], title="Terminal Expected VIX at t+H", row=2, col=1, **ax_opts)
fig.update_xaxes(range=[-40, 120], title="Terminal Expected VIX at t+H", row=2, col=2, **ax_opts)

fig.show()
```

```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# --- 1. Parameters & Simulation ---
np.random.seed(42)
N_paths = 100
n_steps = 100
T = 5.0  # 5 years
dt = T / n_steps
t_array = np.linspace(0, T, n_steps)

# Common Mean
theta = 20.0 
threshold = 60.0

# Model A (Left): Tight "Normal Market" Parameters
kappa_A = 1.0
sigma_A = 8.0 

# Model B (Right): Fat-Tailed "Crisis-Aware" Parameters
kappa_B = 1.0
sigma_B = 25.0

# Simulate Paths (Euler-Maruyama method for OU process)
paths_A = np.zeros((N_paths, n_steps))
paths_B = np.zeros((N_paths, n_steps))
paths_A[:, 0] = theta
paths_B[:, 0] = theta

for i in range(1, n_steps):
    dW_A = np.random.randn(N_paths) * np.sqrt(dt)
    dW_B = np.random.randn(N_paths) * np.sqrt(dt)
    paths_A[:, i] = paths_A[:, i-1] + kappa_A * (theta - paths_A[:, i-1]) * dt + sigma_A * dW_A
    paths_B[:, i] = paths_B[:, i-1] + kappa_B * (theta - paths_B[:, i-1]) * dt + sigma_B * dW_B

# Helper function to bundle paths for efficient Plotly animation
def get_bundled_paths(paths, t_arr, t_idx):
    curr_t = t_arr[:t_idx+1]
    curr_p = paths[:, :t_idx+1]
    N = paths.shape[0]
    
    t_tiled = np.tile(curr_t, (N, 1))
    nan_col = np.full((N, 1), np.nan)
    
    t_bundled = np.hstack([t_tiled, nan_col]).flatten()
    p_bundled = np.hstack([curr_p, nan_col]).flatten()
    return t_bundled, p_bundled

# Helper to format probability AND Expected Time (Geometric RV)
def format_stats(p, step_size):
    # If probability is mathematically effectively zero
    if p < 1e-16:
        return "Area: ~0%<br>E[VIX > 60]: &infin; yrs"
    
    # E[X] for geometric distribution = 1/p trials. Convert to years.
    expected_years = step_size / p
    
    # Format Area
    if p < 1e-4:
        area_str = f"Area: {p:.2e}"
    else:
        area_str = f"Area: {p*100:.2f}%"
        
    # Format Expected Time
    if expected_years > 1e6:
        time_str = f"E[VIX > 60]: {expected_years:.1e} yrs"
    else:
        time_str = f"E[VIX > 60]: {expected_years:.1f} yrs"
        
    return f"{area_str}<br>{time_str}"

# --- 2. Graphics Setup ---
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "<b>Model A: Incorrect Parameters</b><br>Path Realizations", 
        "<b>Model B: Correct Parameters</b><br>Path Realizations",
        "Model A: 60 is a <b>7.1-Sigma</b> Event", 
        "Model B: 60 is a <b>2.2-Sigma</b> Event"
    ),
    vertical_spacing=0.15, horizontal_spacing=0.08
)

c_A = "#00D2FF"      
c_B = "#FF5500"      
c_path = "rgba(255, 255, 255, 0.15)"
c_alert = "#FF0044"  

# 0 & 1: Static Thresholds on Top
fig.add_trace(go.Scatter(x=[0, T], y=[threshold, threshold], mode='lines', line=dict(color=c_alert, dash='dash')), row=1, col=1)
fig.add_trace(go.Scatter(x=[0, T], y=[threshold, threshold], mode='lines', line=dict(color=c_alert, dash='dash')), row=1, col=2)

# 2 & 3: Animated Paths (Start Empty)
fig.add_trace(go.Scatter(x=[], y=[], mode='lines', line=dict(color=c_path, width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=[], y=[], mode='lines', line=dict(color=c_path, width=1)), row=1, col=2)

# Bottom Left (Model A PDF, Tail, Threshold Line, Text)
fig.add_trace(go.Scatter(x=[], y=[], fill='tozeroy', line=dict(color=c_A)), row=2, col=1)
fig.add_trace(go.Scatter(x=[], y=[], fill='tozeroy', fillcolor='rgba(255,0,68,0.5)', line=dict(width=0)), row=2, col=1)
fig.add_trace(go.Scatter(x=[threshold, threshold], opacity=0.3, y=[0, 0.15], mode='lines', line=dict(color=c_alert, dash='dash')), row=2, col=1)
fig.add_trace(go.Scatter(x=[40], y=[0.06], mode='text', text=[""], textfont=dict(color='white', size=14), textposition='middle right'), row=2, col=1)

# Bottom Right (Model B PDF, Tail, Threshold Line, Text)
fig.add_trace(go.Scatter(x=[], y=[], fill='tozeroy', line=dict(color=c_B)), row=2, col=2)
fig.add_trace(go.Scatter(x=[], y=[], fill='tozeroy', fillcolor='rgba(255,0,68,0.5)', line=dict(width=0)), row=2, col=2)
fig.add_trace(go.Scatter(x=[threshold, threshold], opacity=0.3, y=[0, 0.15], mode='lines', line=dict(color=c_alert, dash='dash')), row=2, col=2)
fig.add_trace(go.Scatter(x=[40], y=[0.06], mode='text', text=[""], textfont=dict(color='white', size=14), textposition='middle right'), row=2, col=2)


# --- 3. Animation Logic ---
frames = []
slider_steps = []
x_grid = np.linspace(-20, 100, 400)
steps_to_anim = range(2, n_steps, 2) # Start at step 2 to avoid 0 variance errors

for t_idx in steps_to_anim:
    frame_name = f"f{t_idx}"
    current_time = t_array[t_idx]
    
    # Calculate exact theoretical variance at time t
    var_A = (sigma_A**2 / (2 * kappa_A)) * (1 - np.exp(-2 * kappa_A * current_time))
    var_B = (sigma_B**2 / (2 * kappa_B)) * (1 - np.exp(-2 * kappa_B * current_time))
    std_A = np.sqrt(var_A)
    std_B = np.sqrt(var_B)
    
    # Integrate Tail CDFs
    prob_A = 1 - norm.cdf(threshold, theta, std_A)
    prob_B = 1 - norm.cdf(threshold, theta, std_B)
    
    # Generate PDFs and shaded tails
    pdf_A = norm.pdf(x_grid, theta, std_A)
    pdf_B = norm.pdf(x_grid, theta, std_B)
    
    tail_A_y = np.where(x_grid >= threshold, pdf_A, 0)
    tail_B_y = np.where(x_grid >= threshold, pdf_B, 0)
    
    # Bundle paths for current frame
    t_A, p_A = get_bundled_paths(paths_A, t_array, t_idx)
    t_B, p_B = get_bundled_paths(paths_B, t_array, t_idx)

    frames.append(go.Frame(
        data=[
            go.Scatter(x=t_A, y=p_A),                             
            go.Scatter(x=t_B, y=p_B),                             
            go.Scatter(x=x_grid, y=pdf_A),                        
            go.Scatter(x=x_grid, y=tail_A_y),                     
            go.Scatter(text=[format_stats(prob_A, dt)]),          # Live Stats A          
            go.Scatter(x=x_grid, y=pdf_B),                        
            go.Scatter(x=x_grid, y=tail_B_y),                     
            go.Scatter(text=[format_stats(prob_B, dt)]),          # Live Stats B
        ],
        traces=[2, 3, 4, 5, 7, 8, 9, 11],
        name=frame_name
    ))
    
    step = {
        "args": [[frame_name], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
        "label": f"{current_time:.1f}y", "method": "animate"
    }
    slider_steps.append(step)

fig.frames = frames

# --- 4. Layout & Styling ---
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    height=800, width=1000, margin=dict(t=80, b=100, l=60, r=60),
    showlegend=False,
    
    updatemenus=[{
        "type": "buttons", "showactive": False,
        "x": 0.05, "y": -0.15, "pad": {"r": 10, "t": 30},
        "bgcolor": "#333", "font": {"color": "white"}, "bordercolor": "#555", "borderwidth": 1,
        "buttons": [
            {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 50, "redraw": False}, "fromcurrent": True}]},
            {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "fromcurrent": True}]}
        ]
    }],
    
    sliders=[{
        "active": 0, "yanchor": "top", "xanchor": "left",
        "currentvalue": {"font": {"size": 16, "color": "white"}, "prefix": "Time: ", "visible": True},
        "pad": {"b": 10, "t": 50}, "len": 0.8, "x": 0.15, "y": -0.15,
        "steps": slider_steps, "bgcolor": "#333", "font": {"color": "white"}
    }]
)

ax_opts = dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', zeroline=False)

fig.update_yaxes(range=[-20, 100], title="Index Value", row=1, col=1, **ax_opts)
fig.update_yaxes(range=[-20, 100], title="Index Value", row=1, col=2, **ax_opts)
fig.update_xaxes(range=[0, T], title="Time (Years)", row=1, col=1, **ax_opts)
fig.update_xaxes(range=[0, T], title="Time (Years)", row=1, col=2, **ax_opts)

fig.update_yaxes(range=[0, 0.12], title="Probability Density", row=2, col=1, **ax_opts)
fig.update_yaxes(range=[0, 0.12], title="Probability Density", row=2, col=2, **ax_opts)
fig.update_xaxes(range=[-20, 100], title="Forecasted Distribution", row=2, col=1, **ax_opts)
fig.update_xaxes(range=[-20, 100], title="Forecasted Distribution", row=2, col=2, **ax_opts)

fig.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Data Generation (Jump Diffusion) ---
np.random.seed(42)
n_steps = 150
H = 15  # Forecast horizon
W = 20  # Lookback window for the rolling model

# Generate trading days
dates = pd.date_range('2024-01-01', periods=n_steps + H)

# Simulate Jump Diffusion
y = np.zeros(n_steps + H)
y[0] = 100.0
mu = 0.4          # Drift
sigma = 1.5       # Diffusion volatility
jump_idx = 80     # The day the regime changes
jump_size = -45.0 # Massive gap down

for i in range(1, n_steps + H):
    dy = mu + np.random.normal(0, sigma)
    if i == jump_idx:
        dy += jump_size
    y[i] = y[i-1] + dy

# --- Precompute Models & Text Annotations ---
exp_fits_y, exp_fits_x, exp_text = [], [], []
roll_fits_y, roll_fits_x, roll_text = [], [], []

# We start animating from step W
for t in range(W, n_steps + 1):
    # --- Model 1: Expanding Window (Left) ---
    x_exp = np.arange(t)
    y_exp = y[:t]
    slope_exp, int_exp = np.polyfit(x_exp, y_exp, 1)
    
    x_exp_pred = np.arange(0, t + H)
    y_exp_pred = int_exp + slope_exp * x_exp_pred
    
    exp_fits_x.append(dates[0:t+H])
    exp_fits_y.append(y_exp_pred)
    exp_text.append(f"<b>Slope:</b> {slope_exp:.2f}<br><b>Intercept:</b> {int_exp:.2f}")
    
    # --- Model 2: Rolling Window (Right) ---
    x_roll = np.arange(t-W, t)
    y_roll = y[t-W:t]
    slope_roll, int_roll = np.polyfit(x_roll, y_roll, 1)
    
    x_roll_pred = np.arange(t-W, t + H)
    y_roll_pred = int_roll + slope_roll * x_roll_pred
    
    roll_fits_x.append(dates[t-W:t+H])
    roll_fits_y.append(y_roll_pred)
    roll_text.append(f"<b>Slope:</b> {slope_roll:.2f}<br><b>Intercept:</b> {int_roll:.2f}")

# --- Graphics Setup ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "<b>Expanding Window Forecast</b><br>Anchored to old regime data", 
        "<b>Rolling Window Forecast</b><br>Scraps old data, adapts to jump"
    ),
    vertical_spacing=0.15, horizontal_spacing=0.08
)

c_bad = "#FF3366"      # Pinkish red (Failing model)
c_good = "#00FFCC"     # Teal (Adapting model)
c_ts = "rgba(255, 255, 255, 0.9)" # White for actual time series

# Pre-calculate Y limits so we can anchor the text perfectly in the upper right
y_min, y_max = min(y) - 10, max(y) + 20
text_x_pos = dates[-5] # Anchor text near the right edge
text_y_pos = y_max - 5 # Anchor text near the top edge

# Dynamic Traces: Actual Time Series
fig.add_trace(go.Scatter(x=dates[:W], y=y[:W], mode='lines', 
                         line=dict(color=c_ts, width=2), name='Actual Data'), row=1, col=1) # Trace 0
fig.add_trace(go.Scatter(x=dates[:W], y=y[:W], mode='lines', 
                         line=dict(color=c_ts, width=2), name='Actual Data'), row=1, col=2) # Trace 1

# Dynamic Traces: Regression Forecasts
fig.add_trace(go.Scatter(x=exp_fits_x[0], y=exp_fits_y[0], mode='lines', 
                         line=dict(color=c_bad, width=3, dash='dot'), name='Expanding Fit'), row=1, col=1) # Trace 2
fig.add_trace(go.Scatter(x=roll_fits_x[0], y=roll_fits_y[0], mode='lines', 
                         line=dict(color=c_good, width=3, dash='dot'), name='Rolling Fit'), row=1, col=2) # Trace 3

# Dynamic Traces: Text Annotations (Pinned to fixed coordinates)
fig.add_trace(go.Scatter(x=[text_x_pos], y=[text_y_pos], mode='text', text=[exp_text[0]], 
                         textposition="bottom left", textfont=dict(color=c_bad, size=14, family="monospace")), 
              row=1, col=1) # Trace 4
fig.add_trace(go.Scatter(x=[text_x_pos], y=[text_y_pos], mode='text', text=[roll_text[0]], 
                         textposition="bottom left", textfont=dict(color=c_good, size=14, family="monospace")), 
              row=1, col=2) # Trace 5

# --- Animation Logic ---
frames = []
slider_steps = []

for i, t in enumerate(range(W, n_steps + 1)):
    frame_name = f"f{t}"
    
    frames.append(go.Frame(
        data=[
            go.Scatter(x=dates[:t], y=y[:t]),               # 0: Update TS left
            go.Scatter(x=dates[:t], y=y[:t]),               # 1: Update TS right
            go.Scatter(x=exp_fits_x[i], y=exp_fits_y[i]),   # 2: Update fit left
            go.Scatter(x=roll_fits_x[i], y=roll_fits_y[i]), # 3: Update fit right
            go.Scatter(text=[exp_text[i]]),                 # 4: Update text left
            go.Scatter(text=[roll_text[i]])                 # 5: Update text right
        ],
        traces=[0, 1, 2, 3, 4, 5],
        name=frame_name
    ))
    
    step = {
        "args": [[frame_name], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
        "label": dates[t].strftime('%Y-%m-%d'),
        "method": "animate"
    }
    slider_steps.append(step)

fig.frames = frames

# --- Layout & Styling ---
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=600, width=1200,
    margin=dict(t=80, b=100, l=60, r=60),
    showlegend=False,
    
    updatemenus=[{
        "type": "buttons",
        "showactive": False,
        "x": 0.05, "y": -0.20,
        "pad": {"r": 10, "t": 30},
        "bgcolor": "#333",
        "font": {"color": "white"},
        "bordercolor": "#555",
        "borderwidth": 1,
        "buttons": [
            {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 80, "redraw": False}, "fromcurrent": True}]},
            {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "fromcurrent": True}]}
        ]
    }],
    
    sliders=[{
        "active": 0,
        "yanchor": "top", "xanchor": "left",
        "currentvalue": {"font": {"size": 16, "color": "white"}, "prefix": "Date: ", "visible": True},
        "transition": {"duration": 0, "easing": "linear"},
        "pad": {"b": 10, "t": 50},
        "len": 0.8, "x": 0.15, "y": -0.20,
        "steps": slider_steps,
        "bgcolor": "#333",
        "font": {"color": "white"}
    }]
)

# Fix axis ranges to keep the view stable during animation
ax_opts = dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', zeroline=False)

fig.update_xaxes(range=[dates[0], dates[-1]], title="Date", **ax_opts)
fig.update_yaxes(range=[y_min, y_max], title="Asset Price", row=1, col=1, **ax_opts)
fig.update_yaxes(range=[y_min, y_max], title="Asset Price", row=1, col=2, **ax_opts)

fig.show()
```

```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# --- 1. Parameters & Simulation ---
np.random.seed(42)
n_steps = 100
T = 1.0
dates = np.linspace(0, T, n_steps)

# True Price: Drifts around 100 consistently (No Regime Change)
true_price = np.ones(n_steps) * 100
true_price += np.cumsum(np.random.normal(0, 0.3, n_steps)) 

# Noisy Quotes (Illiquid Market)
R_true = 10.0 # Lowered slightly so outliers stand out more
quotes = true_price + np.random.normal(0, np.sqrt(R_true), n_steps)

# INJECT MASSIVE OUTLIERS (e.g., flash crash or bad prints)
quotes[50] -= 30
quotes[51] -= 25

# Create a "Market Center" array for the density plot to follow the outliers
market_center = true_price.copy()
market_center[50] = quotes[50]
market_center[51] = quotes[51]

# --- 2. Kalman Filter Implementations ---
# Shared base parameters
R_assumed = 40.0  # High assumed sensor noise
Q_assumed = 0.1   # Low assumed process noise (trusts model)

def run_kalman_filter(quotes, R, Q, inject_uncertainty_at=None):
    x_est = []
    P_est = []
    
    # Initial state
    x = 100.0
    P = 1.0
    
    for i, z in enumerate(quotes):
        # 1. Predict (Prior)
        x_pred = x
        P_pred = P + Q
        
        # Inject massive uncertainty at specified steps (e.g., when it gets "fooled" by large errors)
        if inject_uncertainty_at and i in inject_uncertainty_at:
            P_pred += 150.0 # P is large!
            
        # 2. Update (Posterior)
        K = P_pred / (P_pred + R) # Kalman Gain
        x = x_pred + K * (z - x_pred)
        P = (1 - K) * P_pred
        
        x_est.append(x)
        P_est.append(P)
        
    return np.array(x_est), np.array(P_est)

# Scenario A: Strict Model (ignores outliers, holds strong)
x_low, P_low = run_kalman_filter(quotes, R=R_assumed, Q=Q_assumed)

# Scenario B: Dynamic adjustment gets FOOLED by outliers
# It panics at t=50 (bad drop) and has to panic again at t=52 (recovery)
x_high, P_high = run_kalman_filter(quotes, R=R_assumed, Q=Q_assumed, inject_uncertainty_at=[50, 52])


# --- 3. Graphics Setup ---
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "<b>Strict Model (Low K)</b><br>Ignores Bad Prints (Optimal)", 
        "<b>Adaptive Filter (Dynamic K)</b><br>Overreacts to Outliers (Suboptimal)",
        "Belief vs Market (Strict)", 
        "Belief vs Market (Adaptive)"
    ),
    vertical_spacing=0.15, horizontal_spacing=0.08
)

# Colors
c_low = "#00D2FF"       # Cyan for Strict
c_low_ghost = "rgba(0, 210, 255, 0.15)"
c_high = "#FF5500"      # Orange for Adaptive
c_high_ghost = "rgba(255, 85, 0, 0.15)"
c_quote_dist = "rgba(200, 200, 200, 0.2)" # Grey for True Market Noise
c_quote = "rgba(255, 255, 255, 0.6)"
c_transparent = "rgba(0,0,0,0)"

# Static Traces: True Price [Indices 0, 1]
fig.add_trace(go.Scatter(x=dates, y=true_price, mode='lines', line=dict(color='gray', width=2, dash='dash'), name='True Price'), row=1, col=1)
fig.add_trace(go.Scatter(x=dates, y=true_price, mode='lines', line=dict(color='gray', width=2, dash='dash'), name='True Price'), row=1, col=2)

# Dynamic Traces Initialization: Quotes (2, 3) and KF Lines (4, 5)
fig.add_trace(go.Scatter(x=[dates[0]], y=[quotes[0]], mode='markers', marker=dict(color=c_quote, size=6), name='Quotes'), row=1, col=1)
fig.add_trace(go.Scatter(x=[dates[0]], y=[quotes[0]], mode='markers', marker=dict(color=c_quote, size=6), name='Quotes'), row=1, col=2)

fig.add_trace(go.Scatter(x=[dates[0]], y=[x_low[0]], mode='lines', line=dict(color=c_low, width=3), name='KF Strict'), row=1, col=1)
fig.add_trace(go.Scatter(x=[dates[0]], y=[x_high[0]], mode='lines', line=dict(color=c_high, width=3), name='KF Adaptive'), row=1, col=2)

# Dynamic Traces Initialization: Density Plots [Indices 6 through 11]
x_grid = np.linspace(60, 115, 300)
y_zero = np.zeros_like(x_grid)

# Left Plot Densities
fig.add_trace(go.Scatter(x=x_grid, y=y_zero, fill='tozeroy', line=dict(width=0, color=c_transparent), fillcolor=c_quote_dist, name='Market Noise'), row=2, col=1) # 6
fig.add_trace(go.Scatter(x=x_grid, y=y_zero, fill='tozeroy', line=dict(width=0, color=c_transparent), fillcolor=c_high_ghost, name='Ghost Adaptive'), row=2, col=1) # 7
fig.add_trace(go.Scatter(x=x_grid, y=y_zero, fill='tozeroy', line=dict(color=c_low), name='Main Strict'), row=2, col=1) # 8

# Right Plot Densities
fig.add_trace(go.Scatter(x=x_grid, y=y_zero, fill='tozeroy', line=dict(width=0, color=c_transparent), fillcolor=c_quote_dist, name='Market Noise'), row=2, col=2) # 9
fig.add_trace(go.Scatter(x=x_grid, y=y_zero, fill='tozeroy', line=dict(width=0, color=c_transparent), fillcolor=c_low_ghost, name='Ghost Strict'), row=2, col=2) # 10
fig.add_trace(go.Scatter(x=x_grid, y=y_zero, fill='tozeroy', line=dict(color=c_high), name='Main Adaptive'), row=2, col=2) # 11


# --- 4. Animation Logic ---
frames = []
slider_steps = []
steps_to_anim = range(0, n_steps, 2) # Skip every other frame for performance

for t in steps_to_anim:
    frame_name = f"f{t}"
    
    # Slice current data
    curr_dates = dates[:t+1]
    curr_quotes = quotes[:t+1]
    curr_x_low = x_low[:t+1]
    curr_x_high = x_high[:t+1]
    
    # Calculate Current PDFs (Market center follows the outliers briefly to show the bad signal)
    pdf_market = norm.pdf(x_grid, market_center[t], np.sqrt(R_true))
    pdf_low = norm.pdf(x_grid, x_low[t], np.sqrt(P_low[t]))
    pdf_high = norm.pdf(x_grid, x_high[t], np.sqrt(P_high[t]))

    frames.append(go.Frame(
        data=[
            go.Scatter(x=curr_dates, y=curr_quotes),  # 2
            go.Scatter(x=curr_dates, y=curr_quotes),  # 3
            go.Scatter(x=curr_dates, y=curr_x_low),   # 4
            go.Scatter(x=curr_dates, y=curr_x_high),  # 5
            go.Scatter(y=pdf_market),                 # 6 (Market Noise Left)
            go.Scatter(y=pdf_high),                   # 7 (Ghost High on Left)
            go.Scatter(y=pdf_low),                    # 8 (Main Low on Left)
            go.Scatter(y=pdf_market),                 # 9 (Market Noise Right)
            go.Scatter(y=pdf_low),                    # 10 (Ghost Low on Right)
            go.Scatter(y=pdf_high)                    # 11 (Main High on Right)
        ],
        traces=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 
        name=frame_name
    ))
    
    step = {
        "args": [[frame_name], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
        "label": str(t),
        "method": "animate"
    }
    slider_steps.append(step)

# Add final frame if not included
if n_steps-1 not in steps_to_anim:
    pass

fig.frames = frames

# --- 5. Layout & Styling ---
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=800, width=1000,
    margin=dict(t=80, b=100, l=60, r=60),
    showlegend=False,
    
    updatemenus=[{
        "type": "buttons",
        "showactive": False,
        "x": 0.05, "y": -0.15,
        "pad": {"r": 10, "t": 30},
        "bgcolor": "#333",
        "font": {"color": "white"},
        "bordercolor": "#555",
        "borderwidth": 1,
        "buttons": [
            {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 50, "redraw": False}, "fromcurrent": True}]},
            {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "fromcurrent": True}]}
        ]
    }],
    
    sliders=[{
        "active": 0,
        "yanchor": "top", "xanchor": "left",
        "currentvalue": {"font": {"size": 16, "color": "white"}, "prefix": "Step: ", "visible": True},
        "transition": {"duration": 0, "easing": "linear"},
        "pad": {"b": 10, "t": 50},
        "len": 0.8, "x": 0.15, "y": -0.15,
        "steps": slider_steps,
        "bgcolor": "#333",
        "font": {"color": "white"}
    }]
)

# Axis Styling
ax_opts = dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', zeroline=False)
price_range = [60, 115] 

fig.update_yaxes(range=price_range, title="Price ($)", row=1, col=1, **ax_opts)
fig.update_yaxes(range=price_range, title="Price ($)", row=1, col=2, **ax_opts)
fig.update_xaxes(range=[0, T], title="Time (Years)", row=1, col=1, **ax_opts)
fig.update_xaxes(range=[0, T], title="Time (Years)", row=1, col=2, **ax_opts)

fig.update_yaxes(range=[0, 0.4], title="Density (Confidence)", row=2, col=1, **ax_opts)
fig.update_yaxes(range=[0, 0.4], title="Density (Confidence)", row=2, col=2, **ax_opts)
fig.update_xaxes(range=price_range, title="Estimated True Price", row=2, col=1, **ax_opts)
fig.update_xaxes(range=price_range, title="Estimated True Price", row=2, col=2, **ax_opts)

fig.show()
```

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm

# 1. Load Data
# Assuming vix.csv has columns: Date, SPY, VIX
df = pd.read_csv('vix.csv', parse_dates=['Date'])
df = df.sort_values('Date')

# 2. Prepare AR(1) Variables
# We regress VIX_t (y) on VIX_{t-1} (x)
df['VIX_lag'] = df['VIX'].shift(1)
df = df.dropna()

y = df['VIX']
x = sm.add_constant(df['VIX_lag']) # Adds the intercept (alpha)

# 3. Fit Linear Regression
model = sm.OLS(y, x).fit()
alpha = model.params['const']
beta = model.params['VIX_lag']
resid_std = np.std(model.resid)

# 4. Convert AR(1) to OU Parameters
# Assume daily data, so delta_t = 1/252 (annualized) 
# or delta_t = 1 (step-wise)
dt = 1/252 

kappa = -np.log(beta) / dt
theta = alpha / (1 - beta)
sigma = resid_std * np.sqrt(-2 * np.log(beta) / (dt * (1 - beta**2)))

print(f"--- Calibrated OU Parameters ---")
print(f"Mean Reversion (kappa): {kappa:.4f}")
print(f"Long-term Mean (theta): {theta:.4f}")
print(f"Vol-of-Vol (sigma):     {sigma:.4f}")
print(f"-------------------------------")
print(f"R-squared: {model.rsquared:.4f}")
```

```python
# Assuming results from Step 1 Calibration
# theta was calibrated as your long-term mean
# vix_quotes is your array of VIX prices from the CSV

# 1. Initialize State to first quote
x_hat = quotes[0] 

# 2. Initialize Covariance 
# We start with some uncertainty (e.g., 1.0)
P = 1.0 

# 3. Define the measurement noise (R)
# Unlike Q, which we calibrated, R is your 'trust' in the quotes.
# Let's assume a 1% standard deviation for sensor noise
R = (0.01 * x_hat)**2 

print(f"Filter Initialized:")
print(f"Initial State Estimate: {x_hat}")
print(f"Initial Uncertainty (P): {P}")
```

```python
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm

# 1. Load Data
df = pd.read_csv('vix.csv')
df['Date'] = pd.to_datetime(df['Date'])
n_steps = len(df)
dates = df['Date']
vix_quotes = df['VIX'].values

# 2. Calibration (AR1 Model)
df_lag = df.copy()
df_lag['VIX_prev'] = df_lag['VIX'].shift(1)
df_lag = df_lag.dropna()

y_cal = df_lag['VIX']
x_cal = sm.add_constant(df_lag['VIX_prev'])
res = sm.OLS(y_cal, x_cal).fit()

alpha_cal = res.params['const']
beta_cal = res.params['VIX_prev']
Q_cal = np.var(res.resid) 
R_assumed = 2.0           

# 3. Kalman Filter Loop
x_est, P_est, K_hist = [], [], []
x, P = vix_quotes[0], 5.0 

for z in vix_quotes:
    x_pred = beta_cal * x + alpha_cal
    P_pred = (beta_cal**2) * P + Q_cal
    innovation = z - x_pred
    K = P_pred / (P_pred + R_assumed)
    x = x_pred + K * innovation
    P = (1 - K) * P_pred
    x_est.append(x)
    P_est.append(P)
    K_hist.append(K)

x_est, P_est, K_hist = np.array(x_est), np.array(P_est), np.array(K_hist)

# 4. Graphics Setup
fig = make_subplots(
    rows=2, cols=1,
    row_heights=[0.65, 0.35],
    subplot_titles=(
        "<b>VIX Kalman Filter: OU Dynamics & 3-Step Forecast</b>", 
        "<b>Filter Dynamics: Gain (K) & Uncertainty (P)</b>"
    ),
    vertical_spacing=0.22 
)

c_quote = "rgba(255, 255, 255, 0.5)"
c_est = "#00D2FF"      
c_conf = "rgba(0, 210, 255, 0.15)"
c_k = "#FF5500"
c_forecast = "#FFD700" 

# --- INITIALIZE WITH FIRST DATA POINT (Fixes the "Blank Chart" issue) ---
t0 = 0
upper0 = [x_est[t0] + 2 * np.sqrt(P_est[t0])]
lower0 = [x_est[t0] - 2 * np.sqrt(P_est[t0])]
band_x0 = [dates[t0], dates[t0]]
band_y0 = upper0 + lower0

# Trace 0: CI
fig.add_trace(go.Scatter(x=band_x0, y=band_y0, fill='toself', fillcolor=c_conf, line=dict(width=0), hoverinfo='skip'), row=1, col=1)
# Trace 1: Quotes
fig.add_trace(go.Scatter(x=[dates[t0]], y=[vix_quotes[t0]], mode='markers', marker=dict(color=c_quote, size=5)), row=1, col=1)
# Trace 2: Estimate
fig.add_trace(go.Scatter(x=[dates[t0]], y=[x_est[t0]], mode='lines', line=dict(color=c_est, width=3)), row=1, col=1)
# Trace 3: Forecast (Initial 1 point)
fig.add_trace(go.Scatter(x=[dates[t0]], y=[x_est[t0]], mode='lines+markers', line=dict(color=c_forecast, width=2, dash='dot')), row=1, col=1)
# Trace 4: K
fig.add_trace(go.Scatter(x=[dates[t0]], y=[K_hist[t0]], mode='lines', line=dict(color=c_k, width=2)), row=2, col=1)
# Trace 5: P
fig.add_trace(go.Scatter(x=[dates[t0]], y=[P_est[t0]], mode='lines', line=dict(color=c_est, width=2, dash='dot')), row=2, col=1)

# 5. Animation Logic
frames = []
steps_to_anim = range(0, n_steps - 3, 1) 

for t in steps_to_anim:
    frame_name = f"f{t}"
    curr_dates = dates[:t+1]
    curr_x = x_est[:t+1]
    curr_P = P_est[:t+1]
    
    upper_bound = curr_x + 2 * np.sqrt(curr_P)
    lower_bound = curr_x - 2 * np.sqrt(curr_P)
    band_x = np.concatenate([curr_dates, curr_dates[::-1]])
    band_y = np.concatenate([upper_bound, lower_bound[::-1]])

    f_dates = dates[t:t+4] 
    f_values = [x_est[t]]
    for k in range(1, 4):
        f_values.append(beta_cal * f_values[-1] + alpha_cal)

    frames.append(go.Frame(
        data=[
            go.Scatter(x=band_x, y=band_y),
            go.Scatter(x=curr_dates, y=vix_quotes[:t+1]),
            go.Scatter(x=curr_dates, y=curr_x),
            go.Scatter(x=f_dates, y=f_values),
            go.Scatter(x=curr_dates, y=K_hist[:t+1]),
            go.Scatter(x=curr_dates, y=curr_P),
        ],
        traces=[0, 1, 2, 3, 4, 5],
        name=frame_name
    ))

fig.frames = frames

# 6. Layout & Styling
slider_steps = [{"args": [[f"f{t}"], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}], "label": str(t), "method": "animate"} for t in steps_to_anim]

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    height=850, width=1100, 
    margin=dict(t=100, b=100, l=80, r=80),
    showlegend=False,
    updatemenus=[{
        "type": "buttons", "showactive": False,
        "x": 0.05, "y": -0.12, "buttons": [
            {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 30, "redraw": False}, "fromcurrent": True}]},
            {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]}
        ]
    }],
    sliders=[{"steps": slider_steps, "x": 0.15, "y": -0.12, "len": 0.82}]
)

ax_opts = dict(showgrid=True, gridcolor='rgba(255,255,255,0.08)', zeroline=False)
fig.update_xaxes(range=[dates.min(), dates.max()], **ax_opts)
fig.update_yaxes(range=[vix_quotes.min()-2, vix_quotes.max()+2], title="VIX Level", row=1, col=1, **ax_opts)
fig.update_yaxes(title_text="Gain (K)", color=c_k, range=[0, 1.0], row=2, col=1, **ax_opts)

fig.update_layout(
    yaxis3=dict(
        title="Uncertainty (P)", color=c_est,
        overlaying="y2", side="right",
        range=[0, P_est.max() * 1.1], showgrid=False,
        anchor="x2"
    )
)
fig.data[5].update(yaxis='y3')

fig.show()
```

```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# --- 1. Parameters & Simulation ---
np.random.seed(42)
n_steps = 100
T = 1.0
dates = np.linspace(0, T, n_steps)

# True Price: Drifts around 100, drops to 75 at t=50 (Regime Change)
true_price = np.ones(n_steps) * 100
true_price += np.cumsum(np.random.normal(0, 0.3, n_steps)) # Slight drift
true_price[50:] -= 25 # The massive regime change

# Noisy Quotes (Illiquid Market)
R_true = 25.0 # True variance of the noise
quotes = true_price + np.random.normal(0, np.sqrt(R_true), n_steps)

# --- 2. Kalman Filter Implementations ---
# Shared base parameters
R_assumed = 40.0  # High assumed sensor noise
Q_assumed = 0.1   # Low assumed process noise (trusts model)

def run_kalman_filter(quotes, R, Q, is_adaptive=False):
    x_est = []
    P_est = []
    
    # Initial state
    x = 100.0
    P = 1.0
    
    for i, z in enumerate(quotes):
        # 1. Predict (Prior)
        x_pred = x
        P_pred = P + Q
        
        # --- THE ADAPTIVE "DUAL" LOGIC ---
        # Calculate the "Innovation" (Residual Error)
        innovation = z - x_pred
        
        # Calculate the expected variance of this innovation
        # (Model Uncertainty + Sensor Uncertainty)
        innovation_variance = P_pred + R
        
        if is_adaptive:
            # Check if the error is mathematically extreme (e.g., > 3 standard deviations)
            # 3-sigma variance threshold is 3^2 = 9
            if innovation**2 > 9 * innovation_variance:
                # Shock detected! The filter realizes it is wrong and artificially inflates P
                P_pred += 150.0 
        
        # 2. Update (Posterior)
        K = P_pred / (P_pred + R) # Kalman Gain
        x = x_pred + K * innovation
        P = (1 - K) * P_pred
        
        x_est.append(x)
        P_est.append(P)
        
    return np.array(x_est), np.array(P_est)

# Scenario A: Strict Model (ignores innovation testing, misses regime change)
x_low, P_low = run_kalman_filter(quotes, R=R_assumed, Q=Q_assumed, is_adaptive=False)

# Scenario B: Autonomous Dynamic adjustment (detects shock at t=50 automatically)
x_high, P_high = run_kalman_filter(quotes, R=R_assumed, Q=Q_assumed, is_adaptive=True)


# --- 3. Graphics Setup ---
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "<b>Strict Model (Low K)</b><br>High R, Ignores Noise", 
        "<b>Adaptive Filter (Dynamic K)</b><br>Auto-Detects Regime Change",
        "Belief vs Market (Strict)", 
        "Belief vs Market (Adaptive)"
    ),
    vertical_spacing=0.15, horizontal_spacing=0.08
)

# Colors
c_low = "#00D2FF"       # Cyan for Strict
c_low_ghost = "rgba(0, 210, 255, 0.15)"
c_high = "#FF5500"      # Orange for Adaptive
c_high_ghost = "rgba(255, 85, 0, 0.15)"
c_quote_dist = "rgba(200, 200, 200, 0.2)" # Grey for True Market Noise
c_quote = "rgba(255, 255, 255, 0.6)"
c_transparent = "rgba(0,0,0,0)"

# Static Traces: True Price [Indices 0, 1]
fig.add_trace(go.Scatter(x=dates, y=true_price, mode='lines', line=dict(color='gray', width=2, dash='dash'), name='True Price'), row=1, col=1)
fig.add_trace(go.Scatter(x=dates, y=true_price, mode='lines', line=dict(color='gray', width=2, dash='dash'), name='True Price'), row=1, col=2)

# Dynamic Traces Initialization: Quotes (2, 3) and KF Lines (4, 5)
fig.add_trace(go.Scatter(x=[dates[0]], y=[quotes[0]], mode='markers', marker=dict(color=c_quote, size=6), name='Quotes'), row=1, col=1)
fig.add_trace(go.Scatter(x=[dates[0]], y=[quotes[0]], mode='markers', marker=dict(color=c_quote, size=6), name='Quotes'), row=1, col=2)

fig.add_trace(go.Scatter(x=[dates[0]], y=[x_low[0]], mode='lines', line=dict(color=c_low, width=3), name='KF Strict'), row=1, col=1)
fig.add_trace(go.Scatter(x=[dates[0]], y=[x_high[0]], mode='lines', line=dict(color=c_high, width=3), name='KF Adaptive'), row=1, col=2)

# Dynamic Traces Initialization: Density Plots [Indices 6 through 11]
x_grid = np.linspace(60, 115, 300)
y_zero = np.zeros_like(x_grid)

# Left Plot Densities
fig.add_trace(go.Scatter(x=x_grid, y=y_zero, fill='tozeroy', line=dict(width=0, color=c_transparent), fillcolor=c_quote_dist, name='Market Noise'), row=2, col=1) # 6
fig.add_trace(go.Scatter(x=x_grid, y=y_zero, fill='tozeroy', line=dict(width=0, color=c_transparent), fillcolor=c_high_ghost, name='Ghost Adaptive'), row=2, col=1) # 7
fig.add_trace(go.Scatter(x=x_grid, y=y_zero, fill='tozeroy', line=dict(color=c_low), name='Main Strict'), row=2, col=1) # 8

# Right Plot Densities
fig.add_trace(go.Scatter(x=x_grid, y=y_zero, fill='tozeroy', line=dict(width=0, color=c_transparent), fillcolor=c_quote_dist, name='Market Noise'), row=2, col=2) # 9
fig.add_trace(go.Scatter(x=x_grid, y=y_zero, fill='tozeroy', line=dict(width=0, color=c_transparent), fillcolor=c_low_ghost, name='Ghost Strict'), row=2, col=2) # 10
fig.add_trace(go.Scatter(x=x_grid, y=y_zero, fill='tozeroy', line=dict(color=c_high), name='Main Adaptive'), row=2, col=2) # 11


# --- 4. Animation Logic ---
frames = []
slider_steps = []
steps_to_anim = range(0, n_steps, 2) # Skip every other frame for performance

for t in steps_to_anim:
    frame_name = f"f{t}"
    
    # Slice current data
    curr_dates = dates[:t+1]
    curr_quotes = quotes[:t+1]
    curr_x_low = x_low[:t+1]
    curr_x_high = x_high[:t+1]
    
    # Calculate Current PDFs
    pdf_market = norm.pdf(x_grid, true_price[t], np.sqrt(R_true))
    pdf_low = norm.pdf(x_grid, x_low[t], np.sqrt(P_low[t]))
    pdf_high = norm.pdf(x_grid, x_high[t], np.sqrt(P_high[t]))

    frames.append(go.Frame(
        data=[
            go.Scatter(x=curr_dates, y=curr_quotes),  # 2
            go.Scatter(x=curr_dates, y=curr_quotes),  # 3
            go.Scatter(x=curr_dates, y=curr_x_low),   # 4
            go.Scatter(x=curr_dates, y=curr_x_high),  # 5
            go.Scatter(y=pdf_market),                 # 6 (Market Noise Left)
            go.Scatter(y=pdf_high),                   # 7 (Ghost High on Left)
            go.Scatter(y=pdf_low),                    # 8 (Main Low on Left)
            go.Scatter(y=pdf_market),                 # 9 (Market Noise Right)
            go.Scatter(y=pdf_low),                    # 10 (Ghost Low on Right)
            go.Scatter(y=pdf_high)                    # 11 (Main High on Right)
        ],
        traces=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 
        name=frame_name
    ))
    
    step = {
        "args": [[frame_name], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
        "label": str(t),
        "method": "animate"
    }
    slider_steps.append(step)

# Add final frame if not included
if n_steps-1 not in steps_to_anim:
    pass

fig.frames = frames

# --- 5. Layout & Styling ---
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=800, width=1000,
    margin=dict(t=80, b=100, l=60, r=60),
    showlegend=False,
    
    updatemenus=[{
        "type": "buttons",
        "showactive": False,
        "x": 0.05, "y": -0.15,
        "pad": {"r": 10, "t": 30},
        "bgcolor": "#333",
        "font": {"color": "white"},
        "bordercolor": "#555",
        "borderwidth": 1,
        "buttons": [
            {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 50, "redraw": False}, "fromcurrent": True}]},
            {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "fromcurrent": True}]}
        ]
    }],
    
    sliders=[{
        "active": 0,
        "yanchor": "top", "xanchor": "left",
        "currentvalue": {"font": {"size": 16, "color": "white"}, "prefix": "Step: ", "visible": True},
        "transition": {"duration": 0, "easing": "linear"},
        "pad": {"b": 10, "t": 50},
        "len": 0.8, "x": 0.15, "y": -0.15,
        "steps": slider_steps,
        "bgcolor": "#333",
        "font": {"color": "white"}
    }]
)

# Axis Styling
ax_opts = dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', zeroline=False)
price_range = [60, 115] 

fig.update_yaxes(range=price_range, title="Price ($)", row=1, col=1, **ax_opts)
fig.update_yaxes(range=price_range, title="Price ($)", row=1, col=2, **ax_opts)
fig.update_xaxes(range=[0, T], title="Time (Years)", row=1, col=1, **ax_opts)
fig.update_xaxes(range=[0, T], title="Time (Years)", row=1, col=2, **ax_opts)

fig.update_yaxes(range=[0, 0.4], title="Density (Confidence)", row=2, col=1, **ax_opts)
fig.update_yaxes(range=[0, 0.4], title="Density (Confidence)", row=2, col=2, **ax_opts)
fig.update_xaxes(range=price_range, title="Estimated True Price", row=2, col=1, **ax_opts)
fig.update_xaxes(range=price_range, title="Estimated True Price", row=2, col=2, **ax_opts)

fig.show()
```


---

## 12. Hidden Markov Models for Quant Finance (Lecture 51)

`2025 Video Lectures/51. Hidden Markov Models for Quant Finance/hidden_markov_models.ipynb` — Hidden Markov Models for market regimes.

```python
%%html
<style>
/* Overwrite the hard-coded white background for ipywidgets */
.cell-output-ipywidget-background {
    background-color: transparent !important;
}
/* Set widget foreground text and color to match the VS Code dark theme */
:root {
    --jp-widgets-color: var(--vscode-editor-foreground);
    --jp-widgets-font-size: var(--vscode-editor-font-size);
}
</style>
```

```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Generate points for normal distribution curve
x = np.linspace(-4, 4, 1000)
y = 1/np.sqrt(2*np.pi) * np.exp(-x**2/2)

# Draw a random sample from standard normal
random_draw = np.random.normal(0, 1)

# Create figure with secondary y-axis
fig = make_subplots(rows=2, cols=1, vertical_spacing=0.15, subplot_titles=("Stock Returns as a Normal Random Variable", "Empirical Rule of the Normal Distribution"))

# Add normal distribution curve to first subplot
fig.add_trace(
    go.Scatter(
        x=x,
        y=y,
        mode='lines',
        line=dict(color='rgba(0, 255, 255, 1)', width=2),
        name='Standard Normal PDF',
        showlegend=False
    ),
    row=1, col=1
)

# Add vertical line at random draw to first subplot
fig.add_trace(
    go.Scatter(
        x=[random_draw, random_draw],
        y=[-0.1, 0.5],
        mode='lines',
        line=dict(
            color='rgba(255, 0, 255, 1)', 
            width=2,
            dash='dash'
        ),
        name=f'Random Draw: {random_draw:.3f}',
        showlegend=False
    ),
    row=1, col=1
)

# Add empirical rule plot
mu = 0
sigma = 1
x_emp = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 500)
y_emp = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_emp - mu) / sigma) ** 2)

# Add main curve to second subplot
fig.add_trace(
    go.Scatter(
        x=x_emp,
        y=y_emp,
        mode='lines',
        line=dict(color='rgba(0, 255, 255, 1)', width=2),
        name='Normal Distribution',
        showlegend=False
    ),
    row=2, col=1
)

# Define ranges for empirical rule with neon colors
ranges = [
    (mu - 1 * sigma, mu + 1 * sigma, '34%', 'rgba(255, 0, 255, 0.3)'),
    (mu - 2 * sigma, mu + 2 * sigma, '47.5%', 'rgba(0, 255, 0, 0.3)'),
    (mu - 3 * sigma, mu + 3 * sigma, '49%', 'rgba(255, 255, 0, 0.3)')
]

# Add shaded regions for empirical rule
for start, end, label, color in reversed(ranges):
    mask = (x_emp >= start) & (x_emp <= end)
    x_fill = x_emp[mask]
    y_fill = y_emp[mask]
    
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x_fill, x_fill[::-1]]),
            y=np.concatenate([y_fill, np.zeros_like(y_fill)]),
            fill='toself',
            fillcolor=color,
            opacity=0.4,
            line=dict(width=0),
            name=f'{label} Rule',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Add percentage text annotations at the edges of each range
    fig.add_annotation(
        x=start + 0.5,  # Moved right by 0.5
        y=0.2,
        text=label,
        showarrow=False,
        font=dict(size=14, color='white'),
        row=2,
        col=1
    )
    fig.add_annotation(
        x=end - 0.5,  # Moved left by 0.5
        y=0.2, 
        text=label,
        showarrow=False,
        font=dict(size=14, color='white'),
        row=2,
        col=1
    )

# Update layout
fig.update_layout(
    height=700,
    width=1100,
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes for both subplots
for i in [1, 2]:
    fig.update_xaxes(
        title='X',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        row=i, col=1
    )
    fig.update_yaxes(
        title='Probability Density',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        range=[-0.1, 0.5],
        row=i, col=1
    )

fig.show()
```

```python
# Read in NVDA returns data
import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('NVDA_returns.csv')

# Calculate daily returns
df['Returns'] = df['Close'].pct_change()

# Remove the first row which will have NaN return
df = df.dropna()

# Get sample mean and std as MLE estimates
mu_hat = df['Returns'].mean()
sigma_hat = df['Returns'].std()

# Generate points for fitted normal distribution
x_fit = np.linspace(mu_hat - 4*sigma_hat, mu_hat + 4*sigma_hat, 1000)
y_fit = 1/(sigma_hat * np.sqrt(2*np.pi)) * np.exp(-(x_fit - mu_hat)**2/(2*sigma_hat**2))

# Create figure with secondary y-axis
fig = make_subplots(rows=2, cols=1, vertical_spacing=0.15, 
                    subplot_titles=("NVDA Returns with Fitted Normal Distribution", 
                                  "Empirical Rule of the Normal Distribution"))

# Add histogram of returns
fig.add_trace(
    go.Histogram(
        x=df['Returns'],
        name='NVDA Returns',
        nbinsx=50,
        histnorm='probability density',
        marker_color='rgba(0, 255, 255, 0.5)'
    ),
    row=1, col=1
)

# Add fitted normal distribution curve
fig.add_trace(
    go.Scatter(
        x=x_fit,
        y=y_fit,
        mode='lines',
        line=dict(color='rgba(255, 0, 255, 1)', width=2),
        name='Fitted Normal'
    ),
    row=1, col=1
)

# Add vertical lines for standard deviations in first subplot
for n in [-4, -3, -2, -1, 1, 2, 3, 4]:
    fig.add_shape(
        type="line",
        x0=mu_hat + n*sigma_hat,
        x1=mu_hat + n*sigma_hat,
        y0=0,
        y1=30,
        line=dict(color='rgba(128,128,128,0.5)', width=1, dash='dash'),
        row=1, col=1
    )

# Add empirical rule plot
mu = 0
sigma = 1
x_emp = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 500)
y_emp = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_emp - mu) / sigma) ** 2)

# Add main curve to second subplot
fig.add_trace(
    go.Scatter(
        x=x_emp,
        y=y_emp,
        mode='lines',
        line=dict(color='rgba(0, 255, 255, 1)', width=2),
        name='Normal Distribution',
        showlegend=False
    ),
    row=2, col=1
)

# Define ranges for empirical rule with neon colors
ranges = [
    (mu - 1 * sigma, mu + 1 * sigma, '34%', 'rgba(255, 0, 255, 0.3)'),
    (mu - 2 * sigma, mu + 2 * sigma, '47.5%', 'rgba(0, 255, 0, 0.3)'),
    (mu - 3 * sigma, mu + 3 * sigma, '49%', 'rgba(255, 255, 0, 0.3)')
]

# Add shaded regions for empirical rule
for start, end, label, color in reversed(ranges):
    mask = (x_emp >= start) & (x_emp <= end)
    x_fill = x_emp[mask]
    y_fill = y_emp[mask]
    
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x_fill, x_fill[::-1]]),
            y=np.concatenate([y_fill, np.zeros_like(y_fill)]),
            fill='toself',
            fillcolor=color,
            opacity=0.4,
            line=dict(width=0),
            name=f'{label} Rule',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Add percentage text annotations
    fig.add_annotation(
        x=start + 0.5,
        y=0.2,
        text=label,
        showarrow=False,
        font=dict(size=14, color='white'),
        row=2,
        col=1
    )
    fig.add_annotation(
        x=end - 0.5,
        y=0.2,
        text=label,
        showarrow=False,
        font=dict(size=14, color='white'),
        row=2,
        col=1
    )
    
    # Add vertical lines for standard deviations
    fig.add_shape(
        type="line",
        x0=start, x1=start,
        y0=-0.1, y1=1.0,
        line=dict(color='rgba(128,128,128,0.5)', width=1, dash='dash'),
        row=2, col=1
    )
    fig.add_shape(
        type="line",
        x0=end, x1=end,
        y0=-0.1, y1=1.0,
        line=dict(color='rgba(128,128,128,0.5)', width=1, dash='dash'),
        row=2, col=1
    )

# Update layout
fig.update_layout(
    height=800,
    width=1100,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes for both subplots
fig.update_xaxes(
    title='Returns',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    row=1, col=1
)
fig.update_yaxes(
    title='Density',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    row=1, col=1
)

# Update second subplot axes
fig.update_xaxes(
    title='X',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    row=2, col=1
)
fig.update_yaxes(
    title='Probability Density',
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    range=[-0.1, 0.5],
    row=2, col=1
)

fig.show()
```

```python
from scipy.stats import norm

# Calculate probability of observing something > 4 standard deviations
prob_4_sigma = 1 - norm.cdf(4)
print(f"Probability of observing something > 4σ in a normal distribution: {prob_4_sigma:.10f}")
print(f"Or approximately 1 in {int(1/prob_4_sigma):,} observations")

# Calculate in terms of years assuming daily observations (252 trading days per year)
years = (1/prob_4_sigma) / 252
print(f"With daily observations, this would occur on average once every {years:.1f} years")
```

```python
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Generate sample data
np.random.seed(42)
n_samples = 30000  # Increased base number of samples

# Create time-varying normal distributions with varying variance
n_frames = 60
means = np.concatenate([
    np.linspace(0, 2, n_frames//3),
    np.linspace(2, -2, n_frames//3),
    np.linspace(-2, 0, n_frames//3)
])

# Variance increases with distance from mean=0
variances = 0.5 + (np.abs(means) * 0.5)**2

x_range = np.linspace(-6, 6, 300)  # Increased x range
frames = []
all_samples = []

for mean, var in zip(means, variances):
    # Draw significantly more samples when closer to mean=0
    scaling_factor = 1 / (1 + np.abs(mean)**2)  # Quadratic scaling for more dramatic effect
    n_samples_current = int(n_samples * scaling_factor)
    X = np.random.normal(mean, np.sqrt(var), n_samples_current)
    kde_X = stats.gaussian_kde(X)
    
    # Generate samples from current distribution
    samples = np.random.normal(mean, np.sqrt(var), n_samples_current//500)  # Increased sample retention
    all_samples.extend(samples)
    
    frames.append(
        go.Frame(
            data=[
                go.Scatter(
                    x=x_range,
                    y=kde_X(x_range),
                    mode='lines',
                    line=dict(color='rgba(255, 0, 255, 1)', width=2),
                    name='Varying Normal'
                ),
                go.Histogram(
                    x=all_samples,
                    nbinsx=50,  # Increased number of bins
                    name='Sample Distribution',
                    marker_color='rgba(0, 255, 255, 0.6)'
                )
            ]
        )
    )

# Create base figure with subplots
fig = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4], subplot_titles=('Unobservable Data Generating Distribution', 'Observed (Empirical) Stock Returns'))

# Add initial traces
fig.add_trace(frames[0].data[0], row=1, col=1)
fig.add_trace(frames[0].data[1], row=2, col=1)

# Add frames
fig.frames = frames

# Add animation buttons with increased repetitions
fig.update_layout(
    updatemenus=[{
        'type': 'buttons',
        'showactive': False,
        'buttons': [{
            'label': 'Play',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 50, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0},
                'mode': 'immediate',
                'loop': True  # Enable looping
            }]
        }]
    }]
)

# Update layout
fig.update_layout(
    height=700,
    width=1000,
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    title='Normal Distribution with Varying Mean and Variance'
)

# Update axes for both subplots with expanded range
fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    range=[-6, 6],  # Increased range
    row=1, col=1
)
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    row=1, col=1
)

fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    range=[-6, 6],  # Increased range
    row=2, col=1
)
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    range=[0, 200],  # Increased range to accommodate more samples
    row=2, col=1
)

fig.show()
```

```python
# Generate an Ornstein-Uhlenbeck process for the variance path
def ou_process(theta=2.0, mu=1.0, sigma=2.0, dt=0.1, n_steps=n_frames):  # Increased sigma from 0.5 to 2.0
    x = np.zeros(n_steps)
    x[0] = mu
    for t in range(1, n_steps):
        dx = theta * (mu - x[t-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
        x[t] = x[t-1] + dx
    return np.maximum(x, 0.1)  # Ensure variance stays positive

# Generate variance path and means
time_points = np.linspace(0, n_frames/10, n_frames)
variance_path = ou_process()
means = np.concatenate([
    np.linspace(0, 2, n_frames//3),
    np.linspace(2, -2, n_frames//3),
    np.linspace(-2, 0, n_frames//3)
])

# Create base figure with three subplots
fig = make_subplots(
    rows=3, 
    cols=1, 
    row_heights=[0.2, 0.4, 0.4], 
    subplot_titles=(
        'Latent Variance Process',
        'Unobservable Data Generating Distribution', 
        'Observed (Empirical) Stock Returns'
    )
)

frames = []
all_samples = []

for i in range(n_frames):
    # Draw samples using current mean and OU variance
    n_samples_current = int(n_samples * (1 / (1 + np.abs(means[i])**2)))
    X = np.random.normal(means[i], np.sqrt(variance_path[i]), n_samples_current)
    kde_X = stats.gaussian_kde(X)
    
    # Generate samples
    samples = np.random.normal(means[i], np.sqrt(variance_path[i]), n_samples_current//500)
    all_samples.extend(samples)
    
    frames.append(
        go.Frame(
            data=[
                # Variance path
                go.Scatter(
                    x=time_points[:i+1],
                    y=variance_path[:i+1],
                    mode='lines',
                    line=dict(color='rgba(0, 255, 255, 1)', width=2),
                    name='Variance Path'
                ),
                # Distribution
                go.Scatter(
                    x=x_range,
                    y=kde_X(x_range),
                    mode='lines',
                    line=dict(color='rgba(255, 0, 255, 1)', width=2),
                    name='Varying Normal'
                ),
                # Histogram
                go.Histogram(
                    x=all_samples,
                    nbinsx=50,
                    name='Sample Distribution',
                    marker_color='rgba(0, 255, 255, 0.6)'
                )
            ]
        )
    )

# Add initial traces
fig.add_trace(frames[0].data[0], row=1, col=1)
fig.add_trace(frames[0].data[1], row=2, col=1)
fig.add_trace(frames[0].data[2], row=3, col=1)

# Add frames
fig.frames = frames

# Add animation buttons
fig.update_layout(
    updatemenus=[{
        'type': 'buttons',
        'showactive': False,
        'buttons': [{
            'label': 'Play',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 50, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0},
                'mode': 'immediate',
                'loop': True
            }]
        }]
    }]
)

# Update layout
fig.update_layout(
    height=700,
    width=1000,
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    title='Normal Distribution with Varying Mean and OU-Process Variance'
)

# Update axes for all subplots
for row in [1, 2, 3]:
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        row=row, 
        col=1
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        row=row, 
        col=1
    )

# Specific range updates
fig.update_xaxes(range=[0, n_frames/10], row=1, col=1)  # Time range for variance path
fig.update_yaxes(range=[0, 5], row=1, col=1)  # Increased variance range from 3 to 5
fig.update_xaxes(range=[-6, 6], row=2, col=1)  # Distribution x-range
fig.update_yaxes(range=[0, 1], row=2, col=1)  # Added y-range for distribution plot
fig.update_xaxes(range=[-6, 6], row=3, col=1)  # Histogram x-range
fig.update_yaxes(range=[0, 300], row=3, col=1)  # Histogram y-range

fig.show()
```

```python
# Read in NVDA data and calculate returns
df = pd.read_csv('NVDA_returns.csv')
df['Date'] = pd.to_datetime(df['Date'])  # Convert to datetime
df = df.sort_values('Date')  # Sort chronologically
df['Returns'] = df['Close'].pct_change(fill_method=None)
df = df.dropna()

# Filter for 2024-2025 only
df = df[(df['Date'].dt.year >= 2024) & (df['Date'].dt.year <= 2025)]

# Calculate realized volatility using two different windows
df['RV_20'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)  # 20-day rolling vol
df['RV_60'] = df['Returns'].rolling(window=60).std() * np.sqrt(252)  # 60-day rolling vol

# Drop initial NaN values from warmup period
df = df.dropna()

# Create figure
fig = go.Figure()

# Add 20-day volatility
fig.add_trace(
    go.Scatter(
        x=df['Date'],
        y=df['RV_20'],
        name='20-Day Rolling Volatility',
        line=dict(color='rgba(0, 255, 0, 1)', width=2)  # Neon green
    )
)

# Add 60-day volatility
fig.add_trace(
    go.Scatter(
        x=df['Date'],
        y=df['RV_60'],
        name='60-Day Rolling Volatility',
        line=dict(color='rgba(255, 0, 255, 1)', width=2)  # Neon pink
    )
)

# Update layout
fig.update_layout(
    title='NVDA Realized Volatility: 20-Day vs 60-Day Rolling Windows',
    xaxis_title='Date',
    yaxis_title='Annualized Volatility',
    height=500,
    width=1000,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes
fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)',
    dtick='M3'  # Show ticks every quarter
)

fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128,128,128,0.2)',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='rgba(128,128,128,0.5)'
)

fig.show()
```

```python
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

# Read the data
df = pd.read_csv('NVDA_returns.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date', ascending=True)  # Sort by date ascending before calculating returns
df['Returns'] = df['Close'].pct_change()
df = df.dropna()

# Create two time periods
period1 = df[(df['Date'] >= '2025-01-01') & (df['Date'] <= '2025-04-08')]
period2 = df[(df['Date'] >= '2024-09-01') & (df['Date'] <= '2024-12-31')]

# Calculate means
mean1 = period1['Returns'].mean()
mean2 = period2['Returns'].mean()

# Fit KDE for both periods
kde1 = gaussian_kde(period1['Returns'])
kde2 = gaussian_kde(period2['Returns'])

# Generate points for KDE plots
x1 = np.linspace(period1['Returns'].min(), period1['Returns'].max(), 1000)
y1 = kde1(x1)

x2 = np.linspace(period2['Returns'].min(), period2['Returns'].max(), 1000)
y2 = kde2(x2)

# Create subplots
fig = make_subplots(rows=1, cols=2, 
                    subplot_titles=("Jan-May 2025 Returns Distribution (High Volatility)", 
                                  "Sept-Dec 2024 Returns Distribution (Low Volatility)"))

# First subplot
fig.add_trace(
    go.Scatter(
        x=x1,
        y=y1,
        mode='lines',
        line=dict(color='rgba(255, 0, 255, 1)', width=2),
        name='KDE 2025'
    ),
    row=1, col=1
)

# Add mean line for first subplot
fig.add_shape(
    type="line",
    x0=mean1,
    x1=mean1,
    y0=0,
    y1=max(y1)*1.2,  # Extend beyond the plot
    line=dict(
        color='red' if mean1 < 0 else 'green',
        width=2,
        dash='dash'
    ),
    row=1, col=1
)

# Second subplot
fig.add_trace(
    go.Scatter(
        x=x2,
        y=y2,
        mode='lines',
        line=dict(color='rgba(255, 0, 255, 1)', width=2),
        name='KDE 2024'
    ),
    row=1, col=2
)

# Add mean line for second subplot
fig.add_shape(
    type="line",
    x0=mean2,
    x1=mean2,
    y0=0,
    y1=max(y2)*1.2,  # Extend beyond the plot
    line=dict(
        color='red' if mean2 < 0 else 'green',
        width=2,
        dash='dash'
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    height=500,
    width=1100,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes
for i in [1, 2]:
    fig.update_xaxes(
        title='Returns',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        row=1, col=i
    )
    fig.update_yaxes(
        title='Density',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)',
        row=1, col=i
    )

fig.show()
```

```python
import numpy as np
import plotly.graph_objects as go

df['Date'] = pd.to_datetime(df['Date'])

df = df.sort_values('Date', ascending=True)

# Filter data for 2024-2025
df = df[(df['Date'] >= '2024-01-01') & (df['Date'] <= '2025-12-31')]


# --- Calculate rolling volatility ---
window = 20  # 20-day rolling window
df['Volatility'] = df['Returns'].rolling(window=window).std() * np.sqrt(252)  # Annualized

# Volatility thresholds
vol_33 = df['Volatility'].quantile(0.33)
vol_66 = df['Volatility'].quantile(0.66)

# Volatility regimes
df['Regime'] = 'Medium'
df.loc[df['Volatility'] <= vol_33, 'Regime'] = 'Low'
df.loc[df['Volatility'] >= vol_66, 'Regime'] = 'High'

# --- Create figure ---
fig = go.Figure()

# Regime colors
regime_colors = {
    'Low': 'rgb(0,255,0)',
    'Medium': 'rgb(255,255,0)',
    'High': 'rgb(255,0,0)'
}

# --- Build continuous segments ---
dates = df['Date'].values
prices = df['Close'].values
regimes = df['Regime'].values

for i in range(len(df) - 1):
    x = [dates[i], dates[i+1]]
    y = [prices[i], prices[i+1]]
    regime = regimes[i]

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color=regime_colors[regime], width=2),
            name=f"{regime} Volatility",
            showlegend=(i == 0 or regime not in regimes[:i])  # Show each legend item once
        )
    )

# --- Layout ---
fig.update_layout(
    title='NVDA Price Colored by Volatility Regimes',
    height=600,
    width=1100,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    xaxis=dict(
        title='Date',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)'
    ),
    yaxis=dict(
        title='Price',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)'
    )
)

fig.show()
```

```python
# Calculate transition probabilities between regimes
def get_transition_matrix(regimes):
    states = ['Low', 'Medium', 'High']
    n_states = len(states)
    
    # Initialize transition counts matrix
    counts = np.zeros((n_states, n_states))
    
    # Count transitions
    for i in range(len(regimes)-1):
        current = regimes.iloc[i]  # Use iloc for integer-based indexing
        next_state = regimes.iloc[i+1]
        current_idx = states.index(current)
        next_idx = states.index(next_state)
        counts[current_idx, next_idx] += 1
    
    # Convert to probabilities
    probabilities = counts / counts.sum(axis=1, keepdims=True)
    return probabilities, states

# Get transition matrix
trans_matrix, states = get_transition_matrix(df['Regime'])

# Calculate regime statistics
regime_stats = {}
for regime in ['Low', 'Medium', 'High']:
    mask = df['Regime'] == regime
    returns = df.loc[mask, 'Returns']
    
    regime_stats[regime] = {
        'mean': returns.mean(),
        'std': returns.std(),
        'count': len(returns)
    }

# Print results
print("\nTransition Probabilities Matrix:")
print("From/To      Low    Medium    High")
for i, state in enumerate(states):
    print(f"{state:10} {trans_matrix[i,0]:.3f}  {trans_matrix[i,1]:.3f}   {trans_matrix[i,2]:.3f}")

print("\nRegime Statistics (Daily):")
for regime, stats in regime_stats.items():
    print(f"\n{regime} Volatility Regime:")
    print(f"Mean Return: {stats['mean']*100:.2f}%")
    print(f"Std Dev: {stats['std']*100:.2f}%")
    print(f"Number of Days: {stats['count']}")

# Plot regime distributions
x = np.linspace(-0.1, 0.1, 1000)
fig = go.Figure()

for regime in ['Low', 'Medium', 'High']:
    mu = regime_stats[regime]['mean']
    sigma = regime_stats[regime]['std']
    y = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2))
    
    fig.add_trace(
        go.Scatter(
            x=x*100, # Convert to percentage
            y=y,
            name=f"{regime} Volatility",
            line=dict(color=regime_colors[regime])
        )
    )

fig.update_layout(
    title='Return Distributions by Volatility Regime',
    xaxis_title='Daily Return (%)',
    yaxis_title='Density',
    height=500,
    width=900,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    xaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)'
    ),
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)'
    )
)

fig.show()
```

```python
# Simulate returns using the Markov Chain and regime-specific distributions
n_steps = 10000
current_state = np.random.choice(states)  # Random initial state
simulated_returns = []

for _ in range(n_steps):
    # Get current regime parameters
    mu = regime_stats[current_state]['mean']
    sigma = regime_stats[current_state]['std']
    
    # Generate return from regime-specific distribution
    ret = np.random.normal(mu, sigma)
    simulated_returns.append(ret)
    
    # Transition to next state
    current_idx = states.index(current_state)
    current_state = states[np.random.choice(len(states), p=trans_matrix[current_idx])]

# Plot simulated returns vs normal distribution
simulated_returns = np.array(simulated_returns)
mu_sim = np.mean(simulated_returns)
sigma_sim = np.std(simulated_returns)

# Create histogram of simulated returns
fig = go.Figure()

# Add histogram of simulated returns
fig.add_trace(go.Histogram(
    x=simulated_returns*100,
    name='Simulated Returns',
    nbinsx=50,
    histnorm='probability density',
    opacity=0.75
))

# Add normal distribution curve
x = np.linspace(min(simulated_returns)*100, max(simulated_returns)*100, 1000)
y = 1/(sigma_sim*100 * np.sqrt(2 * np.pi)) * np.exp(-((x - mu_sim*100)**2) / (2 * (sigma_sim*100)**2))

fig.add_trace(go.Scatter(
    x=x,
    y=y,
    name='Normal Distribution',
    line=dict(color='red', width=2)
))

fig.update_layout(
    title='Simulated Returns Distribution vs Normal Distribution',
    xaxis_title='Return (%)',
    yaxis_title='Density',
    height=500,
    width=900,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    xaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)'
    ),
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)'
    )
)

fig.show()

print("\nSimulated Returns Statistics:")
print(f"Mean: {mu_sim*100:.2f}%")
print(f"Std Dev: {sigma_sim*100:.2f}%")
print(f"Excess Kurtosis: {((simulated_returns - np.mean(simulated_returns))**4).mean() / (np.std(simulated_returns)**4) - 3:.3f}")
```

```python
# Implement 3-state HMM from scratch

import numpy as np
from scipy.stats import norm
np.random.seed(42)

# Prepare the data
X = df['Returns'].values

# Initialize parameters
N = 3  # number of states
T = len(X)  # number of observations

# Random initialization
pi = np.ones(N) / N  # initial state probabilities
A = np.random.rand(N, N)  # transition matrix
A = A / A.sum(axis=1)[:, np.newaxis]

# Initialize emission parameters (mean and std for each state)
mu = np.array([-0.01, 0, 0.01])  # initial guesses for means
sigma = np.array([0.01, 0.02, 0.03])  # initial guesses for standard deviations

def forward_pass(X, pi, A, mu, sigma):
    alpha = np.zeros((T, N))
    # Initialize first timestep
    for j in range(N):
        alpha[0,j] = pi[j] * norm.pdf(X[0], mu[j], sigma[j])
    
    # Forward recursion
    for t in range(1, T):
        for j in range(N):
            alpha[t,j] = norm.pdf(X[t], mu[j], sigma[j]) * np.sum(alpha[t-1,:] * A[:,j])
            
    return alpha

def backward_pass(X, A, mu, sigma):
    beta = np.zeros((T, N))
    # Initialize last timestep
    beta[T-1,:] = 1
    
    # Backward recursion
    for t in range(T-2, -1, -1):
        for i in range(N):
            beta[t,i] = np.sum([A[i,j] * norm.pdf(X[t+1], mu[j], sigma[j]) * beta[t+1,j] for j in range(N)])
            
    return beta

# Run EM algorithm
max_iter = 100
prev_log_likelihood = -np.inf

for iteration in range(max_iter):
    # E-step
    alpha = forward_pass(X, pi, A, mu, sigma)
    beta = backward_pass(X, A, mu, sigma)
    
    # Calculate gamma (state probabilities)
    gamma = alpha * beta
    gamma = gamma / gamma.sum(axis=1)[:,np.newaxis]
    
    # Calculate xi (transition probabilities)
    xi = np.zeros((T-1, N, N))
    for t in range(T-1):
        for i in range(N):
            for j in range(N):
                xi[t,i,j] = alpha[t,i] * A[i,j] * norm.pdf(X[t+1], mu[j], sigma[j]) * beta[t+1,j]
        xi[t] = xi[t] / xi[t].sum()
    
    # M-step
    # Update initial probabilities
    pi = gamma[0]
    
    # Update transition matrix
    A = xi.sum(axis=0) / gamma[:-1].sum(axis=0)[:,np.newaxis]
    
    # Update emission parameters
    for j in range(N):
        mu[j] = np.sum(gamma[:,j] * X) / np.sum(gamma[:,j])
        sigma[j] = np.sqrt(np.sum(gamma[:,j] * (X - mu[j])**2) / np.sum(gamma[:,j]))
    
    # Check convergence
    log_likelihood = np.sum(np.log(np.sum(alpha * beta, axis=1)))
    if abs(log_likelihood - prev_log_likelihood) < 1e-6:
        break
    prev_log_likelihood = log_likelihood

# Get the hidden states
hidden_states = np.argmax(gamma, axis=1)

# Map numeric states to latent state labels
state_volatilities = np.array([np.std(X[hidden_states == i]) for i in range(N)])
state_order = np.argsort(state_volatilities)
regime_map = {
    state_order[0]: 'Latent State 1',
    state_order[1]: 'Latent State 2',
    state_order[2]: 'Latent State 3'
}
regimes = [regime_map[state] for state in hidden_states]

# Calculate regime statistics
regime_stats = {}
for regime in ['Latent State 1', 'Latent State 2', 'Latent State 3']:
    mask = [r == regime for r in regimes]
    returns = df.loc[mask, 'Returns']
    
    regime_stats[regime] = {
        'mean': returns.mean(),
        'std': returns.std(),
        'count': len(returns)
    }

# Print results
print("\nTransition Probabilities Matrix:")
print("From/To      Latent State 1    Latent State 2    Latent State 3")
states = ['Latent State 1', 'Latent State 2', 'Latent State 3']
for i, state in enumerate(states):
    print(f"{state:15} {A[i,0]:.3f}         {A[i,1]:.3f}          {A[i,2]:.3f}")

print("\nLatent State Statistics (Daily):")
for regime, stats in regime_stats.items():
    print(f"\n{regime}:")
    print(f"Mean Return: {stats['mean']*100:.2f}%")
    print(f"Std Dev: {stats['std']*100:.2f}%")
    print(f"Number of Days: {stats['count']}")

# Plot regime distributions
x = np.linspace(-0.1, 0.1, 1000)
fig = go.Figure()

for regime in ['Latent State 1', 'Latent State 2', 'Latent State 3']:
    mu_r = regime_stats[regime]['mean']
    sigma_r = regime_stats[regime]['std']
    y = 1/(sigma_r * np.sqrt(2 * np.pi)) * np.exp(-(x - mu_r)**2 / (2 * sigma_r**2))
    
    fig.add_trace(
        go.Scatter(
            x=x*100,
            y=y,
            name=regime,
            line=dict(color=f'rgb({50 + int(regime[-1])*50}, {100 + int(regime[-1])*50}, {150 + int(regime[-1])*50})')
        )
    )

fig.update_layout(
    title='Return Distributions by Latent State (HMM)',
    xaxis_title='Daily Return (%)',
    yaxis_title='Density',
    height=500,
    width=900,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    xaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)'
    ),
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)'
    )
)

fig.show()
```


---

## 13. Black-Scholes Implied Volatility in 3 Minutes (Lecture 89)

`2026 Video Lectures/89. Black-Scholes Implied Volatility in 3 Minutes/bsiv.ipynb` — Solve Black-Scholes implied volatility numerically.

```python
%%html
<style>
/* Overwrite the hard-coded white background for ipywidgets */
.cell-output-ipywidget-background {
    background-color: transparent !important;
}
/* Set widget foreground text and color to match the VS Code dark theme */
:root {
    --jp-widgets-color: var(--vscode-editor-foreground);
    --jp-widgets-font-size: var(--vscode-editor-font-size);
}
</style>
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. Parameters & Data Generation ---
S0 = 100        # Initial price
K = 105         # Strike price
T = 1.0         # Time to maturity (1 year)
r = 0.05        # Risk-free rate
vol = 0.25      # Volatility
n_steps = 100   # Number of simulation steps
dt = T / n_steps

def call_payoff(S, K):
    return np.maximum(S - K, 0)

# Generate a Price Path
np.random.seed(np.random.randint(0, 1000))
rets = np.random.normal((r - 0.5 * vol**2) * dt, vol * np.sqrt(dt), n_steps)
path = S0 * np.exp(np.cumsum(rets))
path = np.insert(path, 0, S0)
dates = pd.date_range(start='2025-01-01', periods=n_steps+1, freq='B')

S_range = np.linspace(60, 160, 200)
payoff_vals = call_payoff(S_range, K)

# Colors - Neon Theme
off_white = "#e0e0e0"
strike_color = "#ff4b4b"
payoff_color = "#ff43ff"   # Neon Pink
path_color = "#00f2ff"     # Neon Cyan

# --- 2. Figure Setup ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("European Call Payoff Diagram", "Underlying Price Simulation"),
    horizontal_spacing=0.12
)

# Base traces
fig.add_trace(go.Scatter(x=S_range, y=payoff_vals, line=dict(color=payoff_color, width=3)), row=1, col=1)
fig.add_trace(go.Scatter(x=[K, K], y=[0, 60], line=dict(color=strike_color, width=1, dash='dash')), row=1, col=1)
fig.add_trace(go.Scatter(x=[S0, S0], y=[0, 60], line=dict(color=off_white, width=2), mode='lines'), row=1, col=1)
fig.add_trace(go.Scatter(x=[dates[0], dates[-1]], y=[K, K], line=dict(color=strike_color, width=1, dash='dash')), row=1, col=2)
fig.add_trace(go.Scatter(x=[dates[0]], y=[S0], line=dict(color=path_color, width=2.5), mode='lines'), row=1, col=2)

# --- 3. Animation Frames & Slider Steps ---
frames = []
slider_steps = []

for i in range(n_steps + 1):
    current_price = path[i]
    frame_name = f"f{i}"
    
    # Create Frame
    frames.append(go.Frame(
        data=[
            go.Scatter(x=S_range, y=payoff_vals, line=dict(color=payoff_color, width=3)),
            go.Scatter(x=[K, K], y=[0, 60], line=dict(color=strike_color, width=1, dash='dash')),
            go.Scatter(x=[current_price, current_price], y=[0, 60], line=dict(color=off_white, width=2), mode='lines'),
            go.Scatter(x=[dates[0], dates[-1]], y=[K, K], line=dict(color=strike_color, width=1, dash='dash')),
            go.Scatter(x=dates[:i+1], y=path[:i+1], line=dict(color=path_color, width=2.5), mode='lines')
        ],
        name=frame_name
    ))
    
    # Create Slider Step
    step = {
        "args": [
            [frame_name],
            {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "fromcurrent": True}
        ],
        "label": str(i),
        "method": "animate"
    }
    slider_steps.append(step)

fig.frames = frames

# --- 4. Layout with Buttons and Slider ---
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=650, width=1200,
    margin=dict(t=100, b=150),
    showlegend=False,
    updatemenus=[{
        "type": "buttons",
        "buttons": [
            {
                "label": "▶ Play",
                "method": "animate",
                "args": [None, {"frame": {"duration": 20, "redraw": False}, "fromcurrent": True}]
            },
            {
                "label": "⏸ Pause",
                "method": "animate",
                "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "fromcurrent": True}]
            }
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "showactive": False,
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top"
    }],
    sliders=[{
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 14, "color": off_white},
            "prefix": "Step: ",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 0},
        "pad": {"b": 10, "t": 50},
        "len": 0.85,
        "x": 0.15,
        "y": 0,
        "steps": slider_steps
    }]
)

axis_style = dict(
    showgrid=True, gridcolor='rgba(255,255,255,0.15)', # slightly brighter grid for neon feel
    tickfont=dict(color=off_white), linecolor=off_white,
    zeroline=False, title_font=dict(color=off_white)
)

fig.update_xaxes(axis_style, range=[60, 160], title_text="Asset Price", row=1, col=1)
fig.update_yaxes(axis_style, range=[-2, 60], title_text="Payoff ($)", row=1, col=1)
fig.update_xaxes(axis_style, range=[dates[0], dates[-1]], title_text="Date", row=1, col=2)
fig.update_yaxes(axis_style, range=[60, 160], title_text="Asset Price ($)", row=1, col=2)

fig.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# --- 1. Parameters & Data Generation ---
S0 = 100        # Initial price
K = 105         # Strike price
T = 1.0         # Time to maturity (1 year)
r = 0.05        # Risk-free rate
vol = 0.25      # Volatility
n_steps = 100   # Number of simulation steps
dt = T / n_steps

def call_payoff(S, K):
    return np.maximum(S - K, 0)

def black_scholes_call(S, K, T, r, sigma):
    # Avoid division by zero as T -> 0
    if T <= 1e-5:
        return np.maximum(S - K, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = (S * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
    return call_price

# Generate a Price Path
np.random.seed(42)
rets = np.random.normal((r - 0.5 * vol**2) * dt, vol * np.sqrt(dt), n_steps)
path = S0 * np.exp(np.cumsum(rets))
path = np.insert(path, 0, S0)
dates = pd.date_range(start='2025-01-01', periods=n_steps+1, freq='B')

S_range = np.linspace(60, 160, 200)
payoff_vals = call_payoff(S_range, K)

# Colors - Neon Theme
off_white = "#e0e0e0"
strike_color = "#ff4b4b"
payoff_color = "#ff43ff"   # Neon Pink
bs_color = "#efff14"       # Neon Yellow
path_color = "#00f2ff"     # Neon Cyan

# --- 2. Figure Setup ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("European Call Pricing vs Payoff", "Underlying Price Simulation"),
    horizontal_spacing=0.12
)

# Initial Black-Scholes Curve (at t=0)
bs_initial = black_scholes_call(S_range, K, T, r, vol)

# Base traces
# 1. Payoff
fig.add_trace(go.Scatter(x=S_range, y=payoff_vals, name="Payoff",
                         line=dict(color=payoff_color, width=3)), row=1, col=1)

# 2. Black-Scholes Curve
fig.add_trace(go.Scatter(x=S_range, y=bs_initial, name="BS Price",
                         line=dict(color=bs_color, width=2, dash='dash')), row=1, col=1)

# 3. Strike Line (Hidden from legend to avoid clutter)
fig.add_trace(go.Scatter(x=[K, K], y=[0, 60], showlegend=False,
                         line=dict(color=strike_color, width=1, dash='dash')), row=1, col=1)

# 4. Current Price Marker (UPDATED: Added Label)
fig.add_trace(go.Scatter(x=[S0, S0], y=[0, 60], name="Underlying Price", showlegend=True,
                         line=dict(color=off_white, width=2), mode='lines'), row=1, col=1)

# 5. Strike Line Col 2
fig.add_trace(go.Scatter(x=[dates[0], dates[-1]], y=[K, K], showlegend=False,
                         line=dict(color=strike_color, width=1, dash='dash')), row=1, col=2)

# 6. Path
fig.add_trace(go.Scatter(x=[dates[0]], y=[S0], showlegend=False,
                         line=dict(color=path_color, width=2.5), mode='lines'), row=1, col=2)

# --- 3. Animation Frames & Slider Steps ---
frames = []
slider_steps = []

for i in range(n_steps + 1):
    current_price = path[i]
    T_remaining = T - (i * dt) 
    
    bs_current = black_scholes_call(S_range, K, T_remaining, r, vol)
    frame_name = f"f{i}"
    
    frames.append(go.Frame(
        data=[
            # Col 1: Payoff
            go.Scatter(x=S_range, y=payoff_vals, line=dict(color=payoff_color, width=3)),
            # Col 1: BS Curve
            go.Scatter(x=S_range, y=bs_current, line=dict(color=bs_color, width=2, dash='dash')),
            # Col 1: Strike Line
            go.Scatter(x=[K, K], y=[0, 60], line=dict(color=strike_color, width=1, dash='dash')),
            # Col 1: Current Price (Moving)
            go.Scatter(x=[current_price, current_price], y=[0, 60], line=dict(color=off_white, width=2), mode='lines'),
            
            # Col 2: Strike Line
            go.Scatter(x=[dates[0], dates[-1]], y=[K, K], line=dict(color=strike_color, width=1, dash='dash')),
            # Col 2: Path
            go.Scatter(x=dates[:i+1], y=path[:i+1], line=dict(color=path_color, width=2.5), mode='lines')
        ],
        name=frame_name
    ))
    
    step = {
        "args": [[frame_name], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "fromcurrent": True}],
        "label": str(i),
        "method": "animate"
    }
    slider_steps.append(step)

fig.frames = frames

# --- 4. Layout with Buttons and Slider ---
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=650, width=1200,
    # UPDATED: Increased top margin to make room for higher legend
    margin=dict(t=120, b=150), 
    
    # UPDATED: Raised legend position (y=1.12)
    legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
    
    updatemenus=[{
        "type": "buttons",
        "buttons": [
            {
                "label": "▶ Play",
                "method": "animate",
                "args": [None, {"frame": {"duration": 20, "redraw": False}, "fromcurrent": True}]
            },
            {
                "label": "⏸ Pause",
                "method": "animate",
                "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "fromcurrent": True}]
            }
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "showactive": False,
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top"
    }],
    sliders=[{
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 14, "color": off_white},
            "prefix": "Step: ",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 0},
        "pad": {"b": 10, "t": 50},
        "len": 0.85,
        "x": 0.15,
        "y": 0,
        "steps": slider_steps
    }]
)

axis_style = dict(
    showgrid=True, gridcolor='rgba(255,255,255,0.15)',
    tickfont=dict(color=off_white), linecolor=off_white,
    zeroline=False, title_font=dict(color=off_white)
)

fig.update_xaxes(axis_style, range=[60, 160], title_text="Asset Price", row=1, col=1)
fig.update_yaxes(axis_style, range=[-2, 60], title_text="Option Price ($)", row=1, col=1)
fig.update_xaxes(axis_style, range=[dates[0], dates[-1]], title_text="Date", row=1, col=2)
fig.update_yaxes(axis_style, range=[60, 160], title_text="Asset Price ($)", row=1, col=2)

fig.show()
```

```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# --- 1. Parameters & Market Truth ---
S0 = 100        # Current Spot Price
K = 100         # Strike Price (ATM)
T = 1.0         # Time to maturity
r = 0.05        # Risk-free rate

# The "True" Market parameters
true_vol = 0.25 
# Calculate the fixed "Market Price" target
market_price = (S0 * norm.cdf((np.log(S0/K) + (r + 0.5*true_vol**2)*T)/(true_vol*np.sqrt(T)))) - \
               (K * np.exp(-r*T) * norm.cdf((np.log(S0/K) + (r + 0.5*true_vol**2)*T)/(true_vol*np.sqrt(T)) - true_vol*np.sqrt(T)))

# Solver settings (Slower Animation)
start_vol = 0.65   
n_frames = 100     # INCREASED for smoother slow motion

# --- 2. Helper Functions ---
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return (S * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))

# --- 3. Data Generation ---

# A. Left Plot Data (Error Function)
vol_range = np.linspace(0.1, 0.8, 100)
model_prices = np.array([black_scholes_call(S0, K, T, r, v) for v in vol_range])
error_values = (model_prices - market_price)**2

# B. Right Plot Data (Fixed BS Curve)
S_range = np.linspace(60, 140, 100)
bs_curve_true = black_scholes_call(S_range, K, T, r, true_vol)

# C. Animation Path (Slower Decay)
steps = np.arange(n_frames)
decay_rate = 0.06 # Slower decay rate
simulated_vols = true_vol + (start_vol - true_vol) * np.exp(-decay_rate * steps)

# --- 4. Visual Styling ---
off_white = "#e0e0e0"
neon_cyan = "#00f2ff"    
neon_pink = "#ff43ff"    
neon_yellow = "#efff14"  
grid_color = "rgba(255,255,255,0.1)"

# --- 5. Figure Setup ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "<b>Objective Function: Minimize Error</b><br>(Squared Pricing Error vs Vol)", 
        "<b>Price Convergence</b><br>(Candidate Price Line vs Market)"
    ),
    horizontal_spacing=0.12
)

# === INITIAL TRACES ===
# Left: Error Curve (Static)
fig.add_trace(go.Scatter(x=vol_range, y=error_values, name="Error Function",
                         line=dict(color=neon_pink, width=3)), row=1, col=1)

# Left: Solver Ball (Dynamic)
initial_error = (black_scholes_call(S0, K, T, r, start_vol) - market_price)**2
fig.add_trace(go.Scatter(x=[start_vol], y=[initial_error], name="Current Error",
                         mode='markers', marker=dict(color=neon_cyan, size=15, symbol='circle-open', line=dict(width=3))), row=1, col=1)

# Right: Fixed BS Curve (Static)
fig.add_trace(go.Scatter(x=S_range, y=bs_curve_true, name="Market Structure (Fixed)",
                         line=dict(color=neon_yellow, width=2, dash='dash')), row=1, col=2)

# Right: Market Price Star (Static)
fig.add_trace(go.Scatter(x=[S0], y=[market_price], name="Market Price",
                         mode='markers', marker=dict(color=neon_yellow, size=18, symbol='star')), row=1, col=2)

# Right: Candidate Price Line (Dynamic)
initial_model_price = black_scholes_call(S0, K, T, r, start_vol)
fig.add_trace(go.Scatter(x=[S_range[0], S_range[-1]], y=[initial_model_price, initial_model_price], 
                         name="Candidate Model Price",
                         line=dict(color=neon_cyan, width=3)), row=1, col=2)

# --- 6. Animation Frames ---
frames = []
slider_steps = []

for i, vol_i in enumerate(simulated_vols):
    current_price = black_scholes_call(S0, K, T, r, vol_i)
    current_error = (current_price - market_price)**2
    
    frame_name = f"f{i}"
    
    frames.append(go.Frame(
        data=[
            # Update Ball
            go.Scatter(x=[vol_i], y=[current_error], mode='markers', marker=dict(color=neon_cyan, size=15, symbol='circle-open', line=dict(width=3))),
            # Update Line
            go.Scatter(x=[S_range[0], S_range[-1]], y=[current_price, current_price], line=dict(color=neon_cyan, width=3)),
        ],
        name=frame_name,
        traces=[1, 4] 
    ))
    
    step = {
        "args": [[frame_name], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "fromcurrent": True}],
        "label": f"{vol_i:.2f}",
        "method": "animate"
    }
    slider_steps.append(step)

fig.frames = frames

# --- 7. Layout Adjustments ---
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=700, width=1200,
    
    # Increased bottom margin to fit Legend -> Slider -> Buttons
    margin=dict(t=100, b=200),
    
    # 1. LEGEND POSITION (Below charts, above slider)
    # y coordinate is relative to the plot area. Negative values push it down.
    legend=dict(
        orientation="h", 
        y=-0.15, 
        x=0.5, 
        xanchor="center",
        yanchor="top"
    ),
    
    updatemenus=[{
        "type": "buttons",
        "buttons": [
            {
                "label": "▶ Run Solver",
                "method": "animate",
                # 2. SLOWER ANIMATION: duration=60 (was 20)
                "args": [None, {"frame": {"duration": 60, "redraw": False}, "fromcurrent": True}]
            },
            {
                "label": "⏸ Pause",
                "method": "animate",
                "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "fromcurrent": True}]
            }
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 10},
        "showactive": False,
        "x": 0.1,
        "xanchor": "right",
        # Push buttons down below the legend and slider
        "y": -0.4, 
        "yanchor": "top"
    }],
    
    sliders=[{
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 14, "color": off_white},
            "prefix": "Vol Guess: ",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 0},
        # Push slider down below the legend
        "pad": {"b": 10, "t": 10},
        "len": 0.85,
        "x": 0.15,
        "y": -0.4, 
        "steps": slider_steps
    }]
)

axis_style = dict(
    showgrid=True, gridcolor=grid_color,
    tickfont=dict(color=off_white), linecolor=off_white,
    zeroline=False, title_font=dict(color=off_white)
)

fig.update_xaxes(axis_style, title_text="Volatility (σ)", row=1, col=1)
fig.update_yaxes(axis_style, title_text="Squared Error (Price - Market)²", row=1, col=1)

fig.update_xaxes(axis_style, range=[60, 140], title_text="Spot Price ($)", row=1, col=2)
fig.update_yaxes(axis_style, range=[0, 40], title_text="Option Price ($)", row=1, col=2)

fig.show()
```

```python
import numpy as np
import plotly.graph_objects as go

# --- Load SPX Option Data ---
spx_data = []
with open("spx.csv") as f:
    for line in f:
        fields = line.strip().split(",")
        if len(fields) == 5:
            strike, ttm, iv, callput, weight = map(float, fields)
            spx_data.append({"K": strike, "T": ttm, "iv": iv, "callput": int(callput)})
        # Skip the first line (market params)

# --- Filter for Front-Month (Shortest T) ---
all_ttms = sorted(set(d['T'] for d in spx_data))
front_ttm = min([t for t in all_ttms if t > 0])
front_month = [d for d in spx_data if np.isclose(d['T'], front_ttm, atol=1e-4)]

strikes = []
ivs = []
for d in front_month:
    strikes.append(d["K"])
    ivs.append(d["iv"])

strikes = np.array(strikes)
ivs = np.array(ivs)
sort_idx = np.argsort(strikes)
strikes = strikes[sort_idx]
ivs = ivs[sort_idx]

# --- Neon Style Settings (consistent with animation above) ---
off_white = "#e0e0e0"
neon_cyan = "#00f2ff"
grid_color = "rgba(255,255,255,0.1)"

# --- Plot: Implied Volatility Skew (Front Month) ---
fig_skew = go.Figure()

fig_skew.add_trace(go.Scatter(
    x=strikes, y=ivs,
    mode="markers+lines",
    line=dict(color=neon_cyan, width=3),
    marker=dict(size=15, color=neon_cyan, line=dict(width=3, color="black")),
    name="Front Month Skew",
    connectgaps=True
))

fig_skew.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=500,
    width=800,
    margin=dict(t=75, b=80),
    title="<b>SPX Implied Volatility Skew - Front Month</b>",
    # Font settings to better match the larger animation chart
    font=dict(color=off_white, size=20, family="Arial"),
    xaxis=dict(
        title=dict(text="Strike Price ($)", font=dict(color=off_white, size=18)),
        tickfont=dict(color=off_white, size=16),
        showgrid=True, gridcolor=grid_color,
        linecolor=off_white,
        zeroline=False
    ),
    yaxis=dict(
        title=dict(text="Implied Volatility (σ)", font=dict(color=off_white, size=18)),
        tickfont=dict(color=off_white, size=16),
        showgrid=True, gridcolor=grid_color,
        linecolor=off_white,
        zeroline=False
    ),
    showlegend=False
)

fig_skew.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- Load CSV Data (spot, r, q in first line) ---
csv_path = "spx.csv"
with open(csv_path, 'r') as f:
    spot, r, q = map(float, f.readline().strip().split(','))
    df = pd.read_csv(f, header=None)
    df.columns = ['Strike', 'Maturity', 'IV', 'IsCall', 'w']

# --- Process surface for 3D plotting ---
pivot = df.pivot_table(index='Maturity', columns='Strike', values='IV').sort_index().sort_index(axis=1)
pivot = pivot.interpolate('linear', axis=0).bfill().ffill()
X, Y = np.meshgrid(pivot.columns, pivot.index)
Z = pivot.values

# --- Professional, Clean 3D Plot Style ---
# Transparent main background now
axis_params = dict(
    showbackground=True,
    backgroundcolor="rgba(15,19,26,0.74)",   # slightly see-through dark, still readable grid/axes
    showgrid=True,
    gridcolor="rgba(90,115,145,0.34)",    # gentle blue-grey gridlines
    gridwidth=1.05,
    zeroline=False,
    linecolor="rgba(180,255,255,0.39)",  # subtle cyan axis lines
    linewidth=2.0,
    ticks="outside",
    tickcolor="rgba(180,244,243,0.43)",
    showspikes=False,
)

surface = go.Surface(
    x=X, y=Y, z=Z,
    colorscale=[
        [0.0, '#182731'],
        [0.13, '#226aa7'],
        [0.36, '#44f0ff'],
        [0.60, '#a0ddf6'],
        [0.82, '#f1fdff'],
        [1.0, '#ffffff'],
    ],
    opacity=1.0,
    showscale=True,
    colorbar=dict(
        title=dict(
            text="Implied Volatility",
            font=dict(color="#bee9ec", size=14, family="Consolas, monospace"),
        ),
        tickfont=dict(color="#a0e8ff", size=11, family="Consolas, monospace"),
        len=0.74,
        thickness=14,
        bgcolor='#191e26',
        outlinewidth=1.8,
        outlinecolor="#40d2f0",
        x=1.0,
    )
)

fig = go.Figure(data=[surface])

fig.update_layout(
    title=dict(
        text="Black-Scholes Implied Volatility Surface",
        font=dict(color="#dafaff", size=21, family="Consolas, monospace"),
        x=0.55, xanchor="center",
    ),
    autosize=True,
    height=560, width=960,
    margin=dict(l=44, r=35, b=40, t=64),
    font=dict(family="Consolas, monospace", color="#9fd3e8", size=13),
    paper_bgcolor='rgba(0,0,0,0)',         # transparent main background
    plot_bgcolor='rgba(0,0,0,0)',         # transparent plot/scene background
    scene=dict(
        xaxis=dict(
            title=dict(text="Strike", font=dict(color="#c3f8ff", size=17, family="Consolas, monospace")),
            tickfont=dict(color="#b3ebf8", size=13, family="Consolas, monospace"),
            **axis_params
        ),
        yaxis=dict(
            title=dict(text="Maturity (Years)", font=dict(color="#c3f8ff", size=16, family="Consolas, monospace")),
            tickfont=dict(color="#b3ebf8", size=13, family="Consolas, monospace"),
            **axis_params
        ),
        zaxis=dict(
            title=dict(text="Implied Volatility", font=dict(color="#c3f8ff", size=16, family="Consolas, monospace")),
            tickfont=dict(color="#b3ebf8", size=12, family="Consolas, monospace"),
            **axis_params,
        ),
        bgcolor="rgba(0,0,0,0)",  # fully transparent 3d scene background
        camera=dict(eye=dict(x=1.5, y=1.28, z=0.8))
    ),
)

fig.show()
```


---

## 14. Trading with the Black-Scholes Implied Volatility Surface (Lecture 30)

`2025 Video Lectures/30. Trading with the Black-Scholes Implied Volatility Surface/the_implied_volatility_surface.ipynb` — Build and explore the implied volatility surface.

```python
import numpy as np
from scipy.stats import norm

def black_scholes_call(S, K, r, sigma, T):
    """
    Calculate Black-Scholes price for a European call option.
    
    Parameters:
    S (float): Current stock price
    K (float): Strike price
    r (float): Risk-free interest rate
    sigma (float): Volatility
    T (float): Time to expiration (in years)
    
    Returns:
    float: Call option price
    """
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    call = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return call
```

```python
S = 100 # Spot is given by the market/contract
K = 100 # Strike is given by the market/contract
r = 0.05 # Risk-free rate is given by the market/contract
T = 1 # Time to maturity is given by the market/contract

# Thanks to supply and demand, the market price of the call option is given by the market/contract
C = 9.36 # Call option price is given by the market/contract

# Volatility is not observable!  We can proxy for it using a variety of measures but it isn't directly available.
sigma = 0.2 # Volatility is NOT given by the market/contract!

print("The Black-Scholes call option price is: $", np.round(black_scholes_call(S, K, r, sigma, T), 2))
```

```python
S = 100 # Spot is given by the market/contract
K = 90 # Strike is given by the market/contract
r = 0.05 # Risk-free rate is given by the market/contract
T = 1 # Time to maturity is given by the market/contract

# Thanks to supply and demand, the market price of the call option is given by the market/contract
C = 17.50 # Call option price is given by the market/contract

# Volatility is not observable!  We can proxy for it using a variety of measures but it isn't directly available.
sigma = 0.25 # Volatility is NOT given by the market/contract!

print("The Black-Scholes call option price is: $", np.round(black_scholes_call(S, K, r, sigma, T), 2))
```

```python
from scipy.optimize import newton

def implied_volatility(C, S, K, r, T, sigma_init=0.2):
    """
    Calculate implied volatility using Newton-Raphson method.
    
    Parameters:
    C (float): Market price of call option
    S (float): Current stock price
    K (float): Strike price
    r (float): Risk-free interest rate
    T (float): Time to expiration (in years)
    sigma_init (float): Initial volatility guess
    
    Returns:
    float: Implied volatility
    """
    def objective(sigma):
        return (black_scholes_call(S, K, r, sigma, T) - C)**2
    
    implied_vol = newton(objective, sigma_init)
    return implied_vol
```

```python
# Let's visualize how the Newton-Raphson method converges to the implied volatility
import numpy as np
import matplotlib.pyplot as plt

def plot_implied_vol_convergence(C, S, K, r, T, sigma_init=0.2):
    """Plot the convergence steps of Newton-Raphson method for implied volatility"""
    def objective(sigma):
        return (black_scholes_call(S, K, r, sigma, T) - C)**2
    
    def derivative(sigma):
        h = 1e-7
        return (objective(sigma + h) - objective(sigma)) / h
    
    # Track convergence steps
    sigmas = []
    errors = []
    sigma = sigma_init
    
    for i in range(10):  # Max 10 iterations
        sigmas.append(sigma)
        error = abs(objective(sigma))
        errors.append(error)
        
        if error < 1e-10:
            break
            
        # Newton-Raphson update
        sigma = sigma - objective(sigma)/derivative(sigma)
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.semilogy(range(len(errors)), errors, 'bo-')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Iteration')
    plt.ylabel('Absolute Error ($)')
    plt.title('Newton-Raphson Convergence for Implied Volatility')
    
    plt.show()

# Plot the convergence for our example
plot_implied_vol_convergence(C=12.15, S=100, K=100, r=0.05, T=1)
```

```python
# We observe from the market:
S = 100  # Current stock price
K = 100  # Strike price
r = 0.05  # Risk-free interest rate
T = 1     # Time to expiration (in years)
C = 10.15  # Market price of call option

# Calculate implied volatility for our example
implied_vol = implied_volatility(C, S, K, r, T)
print(f"The implied volatility is: {implied_vol:.1%}")

# Verify the result by calculating option price with implied vol
price_check = black_scholes_call(S, K, r, implied_vol, T)
print(f"Original market price: ${C:.2f}")
print(f"Price using implied vol: ${price_check:.2f}")
```

```python
import matplotlib.pyplot as plt
import numpy as np

# Create figure with four subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))

# Plot f(x) = 2x
x = np.linspace(0, 5, 100)
y = 2*x

ax1.plot(x, y, 'b-', label='f(x) = 2x')
ax1.plot([2], [4], 'ro', label='f(2) = 4')
ax1.plot([2, 2], [0, 4], 'r--', alpha=0.3)
ax1.plot([0, 2], [4, 4], 'r--', alpha=0.3)
ax1.set_title('Original Function: f(x) = 2x')
ax1.grid(True)
ax1.legend()
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')

# Plot inverse function f⁻¹(x) = x/2
ax2.plot(x, x/2, 'g-', label='f⁻¹(x) = x/2')
ax2.plot([4], [2], 'ro', label='f⁻¹(4) = 2')
ax2.plot([4, 4], [0, 2], 'r--', alpha=0.3)
ax2.plot([0, 4], [2, 2], 'r--', alpha=0.3)
ax2.set_title('Inverse Function: f⁻¹(x) = x/2')
ax2.grid(True)
ax2.legend()
ax2.set_xlabel('x')
ax2.set_ylabel('f⁻¹(x)')

# Plot Black-Scholes call price vs volatility
vol = np.linspace(0.1, 0.8, 100)
S = 100
K = 100
r = 0.05
T = 1
price_at_30 = black_scholes_call(S, K, r, 0.3, T)
prices = [black_scholes_call(S, K, r, v, T) for v in vol]

ax3.plot(vol, prices, 'b-', label='BS Call Price')
ax3.plot([0.3], [price_at_30], 'ro', label=f'σ=30% → ${price_at_30:.2f}')
ax3.plot([0.3, 0.3], [0, price_at_30], 'r--', alpha=0.3)
ax3.plot([0, 0.3], [price_at_30, price_at_30], 'r--', alpha=0.3)
ax3.set_title('Black-Scholes: Price vs Volatility')
ax3.grid(True)
ax3.legend()
ax3.set_xlabel('Volatility (σ)')
ax3.set_ylabel('Option Price ($)')

# Plot implied volatility vs option price by flipping axes
ax4.plot(prices, vol, 'g-', label='Implied Volatility')
ax4.plot([price_at_30], [0.3], 'ro', label=f'${price_at_30:.2f} → σ=30%')
ax4.plot([price_at_30, price_at_30], [0, 0.3], 'r--', alpha=0.3)
ax4.plot([0, price_at_30], [0.3, 0.3], 'r--', alpha=0.3)
ax4.set_title('Inverse: Implied Vol vs Price')
ax4.grid(True)
ax4.legend()
ax4.set_xlabel('Option Price ($)')
ax4.set_ylabel('Implied Volatility (σ)')

plt.tight_layout()
plt.show()
```

```python
import numpy as np
import plotly.graph_objects as go

# Data for the volatility surface
strikes = [90, 95, 100, 105, 110]
maturities = [1/12, 3/12, 6/12, 1, 2]  # in years
maturity_labels = ['1 Month', '3 Months', '6 Months', '1 Year', '2 Years']

# Volatility values (in %)
vols = np.array([
    [28.0, 24.5, 22.0, 20.5, 19.5],  # 1 month
    [27.5, 24.0, 21.8, 20.3, 19.3],  # 3 months
    [27.0, 23.5, 21.5, 20.0, 19.0],  # 6 months
    [26.5, 23.0, 21.2, 19.8, 18.8],  # 1 year
    [26.0, 22.5, 21.0, 19.5, 18.5]   # 2 years
])

# Create meshgrid for 3D surface
X, Y = np.meshgrid(strikes, maturities)

# Create the 3D surface plot
fig = go.Figure(data=[go.Surface(
    x=X,
    y=Y,
    z=vols,
    colorscale='viridis'
)])

# Update layout
fig.update_layout(
    title='Implied Volatility Surface',
    scene=dict(
        xaxis_title='Strike Price',
        yaxis_title='Time to Maturity (Years)',
        zaxis_title='Implied Volatility (%)',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.2)
        )
    ),
    width=800,
    height=600,
    template='plotly_dark',
    paper_bgcolor='rgb(30,30,30)',
    plot_bgcolor='rgb(30,30,30)'
)

# Show the plot
fig.show()
```

```python
# Create subplots
fig = go.Figure()
fig2 = go.Figure()

# Plot volatility skew for each maturity
for i, (maturity, label) in enumerate(zip(maturities, maturity_labels)):
    fig.add_trace(
        go.Scatter(
            x=strikes,
            y=vols[i],
            name=label,
            mode='lines+markers'
        )
    )

# Plot term structure for each strike
for i, strike in enumerate(strikes):
    fig2.add_trace(
        go.Scatter(
            x=maturities,
            y=vols[:,i],
            name=f'Strike {strike}',
            mode='lines+markers'
        )
    )

# Update layouts
fig.update_layout(
    title='Volatility Skew Across Maturities',
    xaxis_title='Strike Price',
    yaxis_title='Implied Volatility (%)',
    width=500,
    height=400,
    showlegend=True
)

fig2.update_layout(
    title='Volatility Term Structure by Strike', 
    xaxis_title='Time to Maturity (Years)',
    yaxis_title='Implied Volatility (%)',
    width=500,
    height=400,
    showlegend=True
)

# Display plots side by side using subplot
from plotly.subplots import make_subplots
combined_fig = make_subplots(rows=1, cols=2, subplot_titles=("Volatility Skew", "Term Structure of Volatility"))

# Add traces from both figures
for trace in fig.data:
    combined_fig.add_trace(trace, row=1, col=1)
for trace in fig2.data:
    combined_fig.add_trace(trace, row=1, col=2)

# Update layout of combined figure
combined_fig.update_layout(
    width=1000,
    height=400,
    template='plotly_dark',
    paper_bgcolor='rgb(30,30,30)',
    plot_bgcolor='rgb(30,30,30)',
    showlegend=True
)

# Show combined plot
combined_fig.show()
```

```python
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Parameters from the example
S0 = 100  # Current stock price
K1, K2 = 90, 110  # Strike prices
r = 0.02  # Risk-free rate
T = 1.0  # Time to maturity
n_paths = 10000
n_steps = 252  # Daily steps for a year

# Market vs our volatilities
sigma_market_K1 = 0.25  # 25% IV for K=90
sigma_market_K2 = 0.20  # 20% IV for K=110
sigma_our = 0.225  # Our constant 22.5% vol

# Black-Scholes call price function
def bs_call(S, K, r, sigma, T):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

# Calculate initial prices
our_price_K1 = bs_call(S0, K1, r, sigma_our, T)
market_price_K1 = bs_call(S0, K1, r, sigma_market_K1, T)
our_price_K2 = bs_call(S0, K2, r, sigma_our, T)
market_price_K2 = bs_call(S0, K2, r, sigma_market_K2, T)

# Generate stock price paths using true market volatility structure
dt = T/n_steps
t = np.linspace(0, T, n_steps+1)
W = np.random.standard_normal((n_paths, n_steps+1))
W[:, 0] = 0

# Use average volatility for demonstration
sigma_realized = (sigma_market_K1 + sigma_market_K2)/2
S = np.zeros((n_paths, n_steps+1))
S[:, 0] = S0

for t in range(1, n_steps+1):
    S[:, t] = S[:, t-1] * np.exp((r - 0.5*sigma_realized**2)*dt + 
                                sigma_realized*np.sqrt(dt)*W[:, t])

# Calculate P&L paths
desk_pnl = np.zeros(n_paths)
trader_pnl = np.zeros(n_paths)

for i in range(n_paths):
    # Desk sells K1 call too cheap and buys K2 call too expensive
    desk_pnl[i] = -(max(S[i, -1] - K1, 0) - our_price_K1)  # Loss on K1
    desk_pnl[i] += (max(S[i, -1] - K2, 0) - our_price_K2)  # Loss on K2
    
    # Trader has opposite position
    trader_pnl[i] = -desk_pnl[i]

# Plot results
plt.figure(figsize=(12, 6))
plt.hist(desk_pnl, bins=50, alpha=0.5, label='Desk P&L', color='red')
plt.hist(trader_pnl, bins=50, alpha=0.5, label='Trader P&L', color='green')
plt.axvline(x=np.mean(desk_pnl), color='red', linestyle='--', 
            label=f'Avg Desk P&L: ${np.mean(desk_pnl):.2f}')
plt.axvline(x=np.mean(trader_pnl), color='green', linestyle='--',
            label=f'Avg Trader P&L: ${np.mean(trader_pnl):.2f}')
plt.title('P&L Distribution from Volatility Surface Arbitrage')
plt.xlabel('P&L ($)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()

print(f"Probability of Desk Loss: {np.mean(desk_pnl < 0):.1%}")
print(f"Probability of Counterparty Loss: {np.mean(trader_pnl < 0):.1%}")
print(f"Our Desk EV: ${np.mean(desk_pnl):.2f}")
print(f"Counterparty Desk EV: ${np.mean(trader_pnl):.2f}")
```

```python
# Simulate wealth paths over many trades
n_trades = 1000
n_paths = 100
initial_wealth = 1000

# Set parameters from statistics
p_desk_loss = 0.504
desk_ev = -0.25
p_trader_loss = 0.496 
trader_ev = 0.25

# Create arrays to store wealth paths
desk_wealth = np.zeros((n_paths, n_trades+1))
trader_wealth = np.zeros((n_paths, n_trades+1))
desk_wealth[:, 0] = initial_wealth
trader_wealth[:, 0] = initial_wealth

# Simulate paths
for path in range(n_paths):
    for trade in range(n_trades):
        # Generate random outcomes based on probabilities
        desk_outcome = np.random.choice([-1, 1], p=[p_desk_loss, 1-p_desk_loss])
        # Scale outcome to match EV
        if desk_outcome == 1:
            desk_gain = abs(desk_ev/p_desk_loss)
        else:
            desk_gain = -abs(desk_ev/(1-p_desk_loss))
            
        desk_wealth[path, trade+1] = desk_wealth[path, trade] + desk_gain
        trader_wealth[path, trade+1] = trader_wealth[path, trade] - desk_gain

# Plot results
plt.figure(figsize=(12, 6))
trade_nums = np.arange(n_trades+1)

# Plot all paths with low opacity
for path in range(n_paths):
    plt.plot(trade_nums, desk_wealth[path, :], 'r-', alpha=0.1)
    plt.plot(trade_nums, trader_wealth[path, :], 'g-', alpha=0.1)

# Plot means with high opacity
plt.plot(trade_nums, np.mean(desk_wealth, axis=0), 'r-', 
         label='Average Desk Wealth', linewidth=2)
plt.plot(trade_nums, np.mean(trader_wealth, axis=0), 'g-', 
         label='Average Counterparty Desk Wealth', linewidth=2)

plt.title('Wealth Evolution Over Multiple Trades')
plt.xlabel('Number of Trades')
plt.ylabel('Wealth ($)')
plt.legend()
plt.grid(True)
plt.show()

# Print final wealth statistics
print(f"Average Final Desk Wealth: ${np.mean(desk_wealth[:, -1]):.2f}")
print(f"Average Final Counterparty Desk Wealth: ${np.mean(trader_wealth[:, -1]):.2f}")
```

```python
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Heston model parameters
v0 = 0.04      # initial variance
kappa = 2.0    # mean reversion speed
theta = 0.04   # long-run variance
sigma = 0.3    # volatility of variance
rho = -0.7     # correlation
S0 = 100       # initial stock price
T = 1.0        # time horizon
N = 252        # number of time steps
dt = T/N       # time step size

# Generate correlated Brownian motions
Z1 = np.random.standard_normal(N)
Z2 = rho * Z1 + np.sqrt(1-rho**2) * np.random.standard_normal(N)

# Initialize arrays
S = np.zeros(N+1)
v = np.zeros(N+1)
t = np.linspace(0, T, N+1)

# Set initial values
S[0] = S0
v[0] = v0

# Simulate paths
for i in range(N):
    v[i+1] = v[i] + kappa*(theta - v[i])*dt + sigma*np.sqrt(v[i]*dt)*Z2[i]
    v[i+1] = max(v[i+1], 0)  # ensure variance stays positive
    S[i+1] = S[i] * np.exp((0.05 - 0.5*v[i])*dt + np.sqrt(v[i]*dt)*Z1[i])

# Plot results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

# Stock price path
ax1.plot(t, S, 'b-', label='Stock Price')
ax1.set_title('Heston Model Stock Price Path')
ax1.set_xlabel('Time')
ax1.set_ylabel('Stock Price')
ax1.grid(True)
ax1.legend()

# Volatility path
ax2.plot(t, np.sqrt(v), 'r-', label='Volatility')
ax2.set_title('Heston Model Volatility Path')
ax2.set_xlabel('Time')
ax2.set_ylabel('Volatility')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()
# Calculate option prices and implied vols for two strikes
K1, K2 = 90, 110  # strikes
T_opt = 0.25      # 3-month option

# Calculate terminal stock prices for many paths
n_paths = 10000
Z1_many = np.random.standard_normal((n_paths, N))
Z2_many = rho * Z1_many + np.sqrt(1-rho**2) * np.random.standard_normal((n_paths, N))

S_paths = np.zeros((n_paths, N+1))
v_paths = np.zeros((n_paths, N+1))
S_paths[:,0] = S0
v_paths[:,0] = v0

# Simulate many paths
for i in range(N):
    v_paths[:,i+1] = v_paths[:,i] + kappa*(theta - v_paths[:,i])*dt + sigma*np.sqrt(v_paths[:,i]*dt)*Z2_many[:,i]
    v_paths[:,i+1] = np.maximum(v_paths[:,i+1], 0)
    S_paths[:,i+1] = S_paths[:,i] * np.exp((0.05 - 0.5*v_paths[:,i])*dt + np.sqrt(v_paths[:,i]*dt)*Z1_many[:,i])

# Calculate option prices from MC simulation
idx_T = int(T_opt/dt)
put_price_K1 = np.mean(np.maximum(K1 - S_paths[:,idx_T], 0)) * np.exp(-0.05*T_opt)
call_price_K2 = np.mean(np.maximum(S_paths[:,idx_T] - K2, 0)) * np.exp(-0.05*T_opt)

# Calculate implied volatilities using Newton's method
from scipy.optimize import newton

def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def bs_put(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def implied_vol_put(P, S, K, T, r):
    f = lambda sigma: bs_put(S, K, T, r, sigma) - P
    return newton(f, x0=0.2)

def implied_vol_call(C, S, K, T, r):
    f = lambda sigma: bs_call(S, K, T, r, sigma) - C
    return newton(f, x0=0.2)

impl_vol_K1 = implied_vol_put(put_price_K1, S0, K1, T_opt, 0.05)
impl_vol_K2 = implied_vol_call(call_price_K2, S0, K2, T_opt, 0.05)

print(f"Strike {K1}: Put Price = {put_price_K1:.4f}, Implied Vol = {impl_vol_K1:.4f}")
print(f"Strike {K2}: Call Price = {call_price_K2:.4f}, Implied Vol = {impl_vol_K2:.4f}")

print("\nHeston Pricing Functional Market/Contract Parameters:")
print(f"Initial Stock Price (S0): {S0}")
print(f"Risk-free Rate: 5%")
print(f"Time to Maturity: {T_opt} years")
print(f"Initial Variance (v0): {v0}")
print(f"Mean Reversion Speed (kappa): {kappa}")
print(f"Long-run Variance (theta): {theta}")
print(f"Volatility of Variance (sigma): {sigma}")
print(f"Correlation (rho): {rho}")
```


---

## 15. How to Price Options with Monte Carlo Simulation (Lecture 73)

`2025 Video Lectures/73. How to Price Options with Monte Carlo Simulation/htpowmc.ipynb` — Price options with Monte Carlo simulation.

```python
%%html
<style>
/* Overwrite the hard-coded white background for ipywidgets */
.cell-output-ipywidget-background {
    background-color: transparent !important;
}
/* Set widget foreground text and color to match the VS Code dark theme */
:root {
    --jp-widgets-color: var(--vscode-editor-foreground);
    --jp-widgets-font-size: var(--vscode-editor-font-size);
}
</style>
```

```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Black-Scholes Analytical Call Value as function of Spot ---
from scipy.stats import norm

def black_scholes_call_price(S, K, T, r, sigma):
    """Black-Scholes price for a European Call option."""
    S = np.asarray(S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price

# --- Parameters ---
S0 = 100     # Initial spot
K = 105      # Strike
T = 1.0      # Time to maturity (years)
r = 0.04     # Risk-free rate
sigma = 0.24 # Volatility

# --- Spot Range for Black-Scholes Plot ---
S_range = np.linspace(60, 140, 100)
bs_values = black_scholes_call_price(S_range, K, T, r, sigma)

# --- Simulate Sample Paths of Geometric Brownian Motion (risk-neutral under Q) ---

N_paths = 8    # Number of paths to plot
N_steps = 120  # Time steps
dt = T / N_steps

np.random.seed(1)
dW = np.random.normal(0, np.sqrt(dt), size=(N_paths, N_steps))
W = np.cumsum(dW, axis=1)
t_grid = np.linspace(0, T, N_steps+1)
S_paths = np.zeros((N_paths, N_steps+1))
S_paths[:,0] = S0
for i in range(N_paths):
    X = (r - 0.5*sigma**2) * t_grid[1:] + sigma*W[i]
    S_paths[i,1:] = S0 * np.exp(X)

# --- Make Figure: Left = Black-Scholes vs Spot, Right = Brownian Sample Paths ---
fig = make_subplots(
    rows=1, cols=2, 
    column_widths=[0.53, 0.47],
    subplot_titles=("", "") # remove left/right titles
)

# --- Left Panel: Black-Scholes Curve ---
fig.add_trace(
    go.Scatter(
        x=S_range,
        y=bs_values,
        mode="lines",
        line=dict(color="#23AFFF", width=4),
        name="BS Value"
    ),
    row=1, col=1
)

# Highlight S0 vertical
fig.add_trace(
    go.Scatter(
        x=[S0, S0],
        y=[0, black_scholes_call_price(S0, K, T, r, sigma)],
        mode="lines",
        line=dict(color="#FFC300", dash="dot", width=3),
        showlegend=False
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=[S0],
        y=[black_scholes_call_price(S0, K, T, r, sigma)],
        mode="markers+text",
        marker=dict(size=10, color="#FFC300"),
        text=[f"${black_scholes_call_price(S0, K, T, r, sigma):.2f}$"],
        textposition="top right",
        showlegend=False
    ),
    row=1, col=1
)

fig.update_xaxes(
    title_text="Spot Price S0",
    row=1, col=1,
    range=[58, 142],
    showgrid=True,
    gridcolor='rgba(128,128,128,0.18)'
)
fig.update_yaxes(
    title_text="Option Value C(S0, 0)",
    row=1, col=1,
    showgrid=True,
    gridcolor='rgba(128,128,128,0.18)'
)

# --- Right Panel: Sample Paths w/ Strike and Maturity ---

for i in range(N_paths):
    fig.add_trace(
        go.Scatter(
            x=t_grid,
            y=S_paths[i],
            mode="lines",
            line=dict(width=2, color="#9B59B6"),
            opacity=0.74,
            showlegend=False
        ),
        row=1, col=2
    )

# Strike: Horizontal
fig.add_trace(
    go.Scatter(
        x=[0, T],
        y=[K, K],
        mode="lines",
        line=dict(color="#FFD700", width=3, dash="dash"),
        name="Strike K"
    ),
    row=1, col=2
)

# Maturity: Vertical
fig.add_trace(
    go.Scatter(
        x=[T, T],
        y=[min(S_paths.min(), S0*0.4), max(S_paths.max(), S0*1.6)],
        mode="lines",
        line=dict(color="#12C95A", width=3, dash="dot"),
        name="Maturity T"
    ),
    row=1, col=2
)

fig.update_xaxes(
    title_text="Time t",
    row=1, col=2,
    range=[-0.01, T + 0.02],
    showgrid=True,
    gridcolor='rgba(128,128,128,0.18)'
)
fig.update_yaxes(
    title_text="Spot Price St",
    row=1, col=2,
    range=[min(50, S_paths.min()-5), max(150, S_paths.max()+5)],
    showgrid=True,
    gridcolor='rgba(128,128,128,0.18)'
)

# --- General Style ---
fig.update_layout(
    height=500,
    width=1150,  # wider chart
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#FAF7E0', size=16),
    title=dict(
        text="<br>Black-Scholes Value and Monte Carlo Simulations<br>"
             "<span style='font-size:18px;color:#b7ead8'>Analytical Value & Simulated Paths (Risk-Neutral)</span><br>",
        y=0.98,
        x=0.5,
        font=dict(color="#FAFAFA", size=25)
    ),
    margin=dict(l=44, r=26, t=95, b=62),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.18,
        xanchor="center",
        x=0.5,
        font=dict(size=16)
    )
)

fig.show()
```

```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Dice PMF Setup ---
dice_faces = np.arange(1, 7)
pmf = np.full(6, 1/6)  # Fair dice, uniform PMF
products = dice_faces * pmf
expectation = products.sum()  # Should be 3.5 for a fair 6-sided die

# --- Helper to Generate Animation Frame at Given Step ---
def make_pmf_expectation_fig(step):
    value = dice_faces[step]
    prob = pmf[step]
    product = products[step]
    partial_sum = products[:step+1].sum()
    highlight_sum = partial_sum
    # For right bar, show sum up to current step
    right_bar_val = partial_sum

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.52, 0.48],
        subplot_titles=(
            # Add extra line breaks before and after to increase vertical padding
            "<br>Dice Probability (blue) and Face Value (gold)<br>",
            "<br>Accumulating Expectation (Sum up to current step)<br>"
        ),
        specs=[[{"secondary_y": True}, {}]]
    )

    # --- Left: Twin Bar Chart with Two y-axes (dots/labels removed) ---
    pmf_bar_colors = ["#23AFFF"]*6
    pmf_bar_colors[step] = "#104AF9"
    face_bar_colors = ["#D8A040"]*6
    face_bar_colors[step] = "#FFE066"

    # ADD HIGHLIGHT (thicker border) for current step
    # Set highlight border color to green for both
    highlight_border_color = "#12C95A"  # vibrant green
    pmf_bar_line_colors = ["rgba(0,0,0,0)"] * 6
    pmf_bar_line_colors[step] = highlight_border_color
    pmf_bar_line_widths = [0]*6
    pmf_bar_line_widths[step] = 4

    face_bar_line_colors = ["rgba(0,0,0,0)"] * 6
    face_bar_line_colors[step] = highlight_border_color
    face_bar_line_widths = [0]*6
    face_bar_line_widths[step] = 4

    # Probability bar
    fig.add_trace(
        go.Bar(
            x=dice_faces,
            y=pmf,
            marker=dict(
                color=pmf_bar_colors,
                line=dict(color=pmf_bar_line_colors, width=pmf_bar_line_widths)
            ),
            width=0.38,
            name="Probability",
            showlegend=False,
            offset=-0.16
        ),
        row=1, col=1, secondary_y=False
    )
    # Face value bar
    fig.add_trace(
        go.Bar(
            x=dice_faces,
            y=dice_faces,
            marker=dict(
                color=face_bar_colors,
                line=dict(color=face_bar_line_colors, width=face_bar_line_widths)
            ),
            width=0.34,
            name="Face Value",
            showlegend=False,
            offset=0.16
        ),
        row=1, col=1, secondary_y=True
    )

    # Dots and labels on left chart have been removed

    fig.update_xaxes(
        title_text="Dice Face Value",
        tickvals=dice_faces,
        range=[0.18, 7.1],  # expand left margin and give more space on xlim (was [0.5, 6.5])
        showgrid=False, row=1, col=1
    )
    fig.update_yaxes(
        title_text="Probability", range=[0, 0.22], row=1, col=1,
        secondary_y=False, showgrid=False,
        title_font=dict(color="#23AFFF"), tickfont=dict(color="#23AFFF")
    )
    fig.update_yaxes(
        title_text="Face Value", range=[0.5, 6.5], row=1, col=1,
        secondary_y=True, title_font=dict(color="#D8A040"), tickfont=dict(color="#D8A040"),
        showgrid=False
    )

    # --- Right: Accumulating green bar for partial sum ---
    # Highlight the right bar if this is the last step
    right_bar_marker = dict(color="#14D68B")
    if step == len(dice_faces)-1:
        right_bar_marker["line"] = dict(color="#12C95A", width=4)  # Use green for highlight
    else:
        right_bar_marker["line"] = dict(color="rgba(0,0,0,0)", width=0)

    fig.add_trace(
        go.Bar(
            x=["Sum"],
            y=[right_bar_val],
            marker=right_bar_marker,
            width=0.44,
            name="Partial Sum",
            showlegend=False
        ),
        row=1, col=2
    )

    # Draw final expectation as a faint 'target' horizontal dashed line for illustration
    fig.add_trace(
        go.Scatter(
            x=["Sum","Sum"],
            y=[expectation, expectation],
            mode="lines",
            line=dict(color="#2CFAE6", dash="dash", width=2),
            showlegend=False,
            hoverinfo="none"
        ),
        row=1, col=2
    )
    # Also show value numerically atop the bar
    fig.add_trace(
        go.Scatter(
            x=["Sum"],
            y=[right_bar_val],
            mode="text",
            text=[f"{right_bar_val:.2f}"],
            textposition="top center",
            showlegend=False
        ),
        row=1, col=2
    )

    fig.update_xaxes(
        title_text="", row=1, col=2, tickvals=["Sum"]
    )
    fig.update_yaxes(
        title_text="Accumulated Value",
        row=1, col=2,
        range=[0, 4],
        showgrid=True,
        gridcolor='rgba(128,128,128,0.18)'
    )

    # --- Figure Layout ---
    fig.update_layout(
        height=470,                                 # Taller for more title-to-chart space (was 420)
        width=1200,
        plot_bgcolor='rgba(0,0,0,0)',   # Fully transparent
        paper_bgcolor='rgba(0,0,0,0)',  # Fully transparent
        font=dict(color='#FAF7E0', size=15),
        title=dict(
            # Add line breaks before and after to give more vertical padding
            text=(
                "<br>"  # top vertical padding
                f"Expectations using a Die<br>Accumulated sum: {right_bar_val:.2f} → Expectation (total) = 3.5"
                "<br><br>"  # bottom vertical padding
            ),
            y= .99,                    # Lower the charts slightly from subtitle (was 0.81)
            font=dict(color="#FAFAFA", size=25, family=None),  # Larger size, off-white
            x=0.5,
            xanchor="center",
            yanchor="top"
        ),
        # More top margin (t) and more bottom margin (b) for visually loose spacing
        margin=dict(l=42, r=24, b=85, t=160),  # More space top/bottom (was t=125, b=60)
        barmode='overlay'
    )

    # Also add more vertical space between subplots and any annotations by customizing the subplot_row_heights and spacing/annotations if needed (not strictly necessary for this case).

    return fig

# --- Create Animation Frames for Each Dice Face ---
pmf_frames = [
    go.Frame(
        data=make_pmf_expectation_fig(step).data,
        name=str(step),
        layout=make_pmf_expectation_fig(step).layout
    )
    for step in range(len(dice_faces))
]

fig_pmf = make_pmf_expectation_fig(0)
fig_pmf.frames = pmf_frames

# --- Play Button ---
fig_pmf.update_layout(
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.32,  # Lower play button for more margin (was y=-0.27)
        'showactive': False,
        'buttons': [{
            'label': 'Play',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 700, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        }]
    }]
)

fig_pmf.show()
```

```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Empirical Proportion + Empirical Mean LLN Animation Setup ---

dice_faces = np.arange(1, 7)
pmf = np.full(6, 1/6)
expectation = 3.5

np.random.seed(42)
N = 250  # Number of samples ("rolls") to simulate

samples = np.random.choice(dice_faces, size=N, replace=True)
empirical_counts = np.zeros((N, len(dice_faces)))
empirical_probs = np.zeros((N, len(dice_faces)))
empirical_means = np.zeros(N)
for i in range(N):
    rolls_so_far = samples[:i+1]
    for k, face in enumerate(dice_faces):
        empirical_counts[i, k] = np.sum(rolls_so_far == face)
        empirical_probs[i, k] = empirical_counts[i, k] / (i+1)
    empirical_means[i] = np.mean(rolls_so_far)

def make_lln_empirical_fig(step):
    # step is 0-based (number of data points so far is step+1)
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.54, 0.46],
        subplot_titles=(
            "<br>Empirical Proportions vs Theoretical PMF<br>(as Number of Rolls Increases)",
            "<br>Empirical Mean vs Theoretical Mean<br>(Law of Large Numbers in Action)"
        )
    )

    # --- Left: Bar chart, theoretical PMF, and empirical freq ---
    # Theoretical PMF bars (static)
    fig.add_trace(
        go.Bar(
            x=dice_faces - 0.12,  # shift left a touch
            y=pmf,
            width=0.24,
            name='Theoretical $P(X=x)$',
            marker_color='#23AFFF',
            opacity=0.45,
            showlegend=True if step == 0 else False
        ),
        row=1, col=1
    )

    # Empirical frequency bars so far
    fig.add_trace(
        go.Bar(
            x=dice_faces + 0.12,  # shift right a touch
            y=empirical_probs[step],
            width=0.24,
            name='Empirical Proportion',
            marker_color='#FFE066',
            opacity=1.0,
            showlegend=True if step==0 else False,
            text=[f"{v:.2f}" for v in empirical_probs[step]],
            textposition="outside"
        ),
        row=1, col=1
    )

    # -- Removed roll count annotation here --

    fig.update_xaxes(
        title_text="Dice Face Value",
        tickvals=dice_faces,
        range=[0.7, 6.3],
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="Probability",
        range=[0, 0.44],
        row=1, col=1,
        tickformat=".2f"
    )

    # --- Right: Bar chart for mean --- 
    # Means displayed as two bars side by side
    means_x = ["Theoretical Mean", "Empirical Mean"]
    means_y = [expectation, empirical_means[step]]

    fig.add_trace(
        go.Bar(
            x=means_x,
            y=means_y,
            width=0.44,
            name="Mean Values",
            marker_color=["#23AFFF", "#FFE066"],
            opacity=0.9,
            showlegend=False,
            text=[f"{expectation:.3f}", f"{empirical_means[step]:.3f}"],
            textposition="outside"
        ),
        row=1, col=2
    )

    # Optionally show convergence path (scatter curve for empirical mean)
    if step > 1:
        fig.add_trace(
            go.Scatter(
                x=["Empirical Mean"]*step,
                y=empirical_means[:step],
                mode="lines+markers",
                line=dict(color='#BB9900', width=2, dash='dot'),
                marker=dict(size=3, color='#FFD700'),
                name="Empirical Mean\n(trace)",
                showlegend=False,
                hoverinfo="y+x"
            ),
            row=1, col=2
        )

    # Horizontal target band for expectation, for visibility
    fig.add_shape(
        type="rect",
        xref="x2", yref="y2",
        x0=-0.5, x1=1.5, y0=expectation-0.03, y1=expectation+0.03,
        fillcolor="rgba(30,250,240,0.09)",
        line_width=0, row=1, col=2
    )

    fig.update_xaxes(
        title_text="",
        row=1, col=2,
        tickvals=means_x
    )
    fig.update_yaxes(
        title_text="Sample Mean Value",
        range=[1, 6],
        row=1, col=2
    )

    # --- Layout ---
    fig.update_layout(
        height=450, width=1000,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FAF7E0', size=15),
        title=dict(
            text="<br>Law of Large Numbers: Empirical Frequencies & Mean Converge to Theoretical Values<br>",
            y= .99, font=dict(color="#FAFAFA", size=23),
            x=0.5, xanchor="center", yanchor="top"
        ),
        margin=dict(l=45, r=22, b=80, t=98),
        barmode='group'   # group bars for means
    )
    return fig

# --- Create Animation Frames ---
lln_frames = [
    go.Frame(
        data=make_lln_empirical_fig(n).data,
        name=str(n),
        layout=make_lln_empirical_fig(n).layout
    )
    for n in range(N)
]

fig_lln = make_lln_empirical_fig(0)
fig_lln.frames = lln_frames

# --- Animation play button (no slider) ---
fig_lln.update_layout(
    updatemenus=[
        {
            'type': 'buttons',
            'x': 0.48, 'y': -0.22,
            'showactive': False,
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 30, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 0}
                    }]
                }
            ]
        }
    ]
)

fig_lln.show()
```

```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from numpy import trapezoid

def simps(y, x):
    return trapezoid(y, x)

# --- Normal Distribution Setup ---
mu = 0
sigma = 1
x = np.linspace(-4, 4, 400)  # Higher resolution for smoothness
pdf = norm.pdf(x, mu, sigma)
dx = x[1] - x[0]
norm_expectation = simps(x * pdf, x)  # Should be 0 for standard normal

# --- Animation sweep setup ---
n_steps = 120
step_indices = np.linspace(0, len(x)-1, n_steps, dtype=int)

def make_normal_integration_fig(step_idx):
    cur_idx = step_indices[step_idx]
    x_sweep = x[:cur_idx+1]
    pdf_sweep = pdf[:cur_idx+1]
    partial_sum = simps(x_sweep * pdf_sweep, x_sweep)

    # --- Color/style palette (match discrete die style) ---
    left_primary = "#23AFFF"    # vivid blue for PDF
    left_secondary = "#14D68B"  # fill/area (greenish-teal)
    left_accent = "#FFE066"     # highlight sweep line (gold)
    left_density = "#D8A040"    # x*p(x) curve (gold/brown)
    right_bar_main = "#14D68B"  # green for accumulating bar
    right_bar_highlight = "#12C95A"   # border highlight
    faint_target = "#2CFAE6"    # blue/turq target
    text_color = "#FAF7E0"

    # --- Figure and subplots
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.52, 0.48],
        subplot_titles=(
            "Normal PDF (blue) and x*p(x) (gold)",
            "Accumulated Expectation (integral partial sum)"
        ),
        specs=[[{"secondary_y": False}, {}]]
    )

    # --- Left: PDF, x*p(x) (gold), swept area, sweep line ---
    # PDF curve
    fig.add_trace(
        go.Scatter(
            x=x, y=pdf,
            mode='lines',
            line=dict(color=left_primary, width=2.3),
            name="Normal PDF",
            showlegend=False
        ),
        row=1, col=1
    )
    # x*p(x) curve (gold/brown dotted)
    fig.add_trace(
        go.Scatter(
            x=x, y=x*pdf,
            mode='lines',
            line=dict(color=left_density, width=2.0, dash='dot'),
            name="x*p(x)",
            showlegend=False,
            opacity=0.78
        ),
        row=1, col=1
    )
    # Area fill under x*p(x)
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x_sweep, x_sweep[::-1]]),
            y=np.concatenate([x_sweep*pdf_sweep, np.zeros_like(x_sweep)]),
            fill='toself',
            fillcolor='rgba(20, 214, 139, 0.22)',  # match right_bar_main, translucent
            line=dict(color='rgba(20,214,139,0.42)', width=0.9),
            hoverinfo='skip',
            showlegend=False
        ),
        row=1, col=1
    )
    # Sweep vertical line
    fig.add_trace(
        go.Scatter(
            x=[x[cur_idx], x[cur_idx]],
            y=[min(np.min(pdf), np.min(x*pdf)), max(pdf) * 0.99],
            mode="lines",
            line=dict(color=left_accent, width=4, dash='dash'),
            showlegend=False, hoverinfo="skip"
        ),
        row=1, col=1
    )
    # Marker at (x[cur_idx], pdf[cur_idx])
    fig.add_trace(
        go.Scatter(
            x=[x[cur_idx]], y=[pdf[cur_idx]],
            mode="markers",
            marker=dict(size=15, color=left_accent, line=dict(width=2, color="#ffffff")),
            showlegend=False, hoverinfo="skip"
        ),
        row=1, col=1
    )

    # Axes config to resemble discrete version
    y_min = np.min(x*pdf) * 1.14
    y_max = max(pdf)*1.11
    fig.update_xaxes(
        title_text="x",
        tickvals=[-4, -2, 0, 2, 4],
        range=[-4.2, 4.2],
        row=1, col=1,
        showgrid=False,
    )
    fig.update_yaxes(
        title_text="p(x), x·p(x)",
        range=[y_min, y_max],
        row=1, col=1,
        showgrid=False,
        title_font=dict(color="#23AFFF"), tickfont=dict(color="#23AFFF")
    )

    # --- Right: Accumulating bar for the current integral value ---
    right_bar_marker = dict(color=right_bar_main)
    if step_idx == n_steps-1:
        right_bar_marker["line"] = dict(color=right_bar_highlight, width=4)
    else:
        right_bar_marker["line"] = dict(color="rgba(0,0,0,0)", width=0)

    fig.add_trace(
        go.Bar(
            x=["Sum"],
            y=[partial_sum],
            marker=right_bar_marker,
            width=0.48,
            name="Partial Sum",
            showlegend=False
        ),
        row=1, col=2
    )

    # Faint target dashed line for true expectation
    fig.add_trace(
        go.Scatter(
            x=["Sum","Sum"],
            y=[norm_expectation, norm_expectation],
            mode="lines",
            line=dict(color=faint_target, dash="dash", width=2),
            showlegend=False,
            hoverinfo="none"
        ),
        row=1, col=2
    )
    # Numeric annotation on bar
    fig.add_trace(
        go.Scatter(
            x=["Sum"],
            y=[partial_sum],
            mode="text",
            text=[f"{partial_sum:.4f}"],
            textposition="top center",
            showlegend=False
        ),
        row=1, col=2
    )

    fig.update_xaxes(
        title_text="", row=1, col=2, tickvals=["Sum"]
    )
    fig.update_yaxes(
        title_text="Accumulated Value",
        row=1, col=2,
        range=[-0.45, 0.45],
        showgrid=True,
        gridcolor='rgba(128,128,128,0.18)'
    )

    # --- Figure Layout (discrete style, transparent, padding) ---
    fig.update_layout(
        height=470,
        width=900,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FAF7E0', size=15),
        title=dict(
            text=(
                "Normal Expectation Accumulation<br>"
                f"Accumulated sum: {partial_sum:.4f}  (Final: {norm_expectation:.4f})"
            ),
            y=0.93,  # shifted main title down a bit (was .99)
            font=dict(color="#FAFAFA", size=22, family=None),
            x=0.5,
            xanchor="center",
            yanchor="top"
        ),
        margin=dict(l=42, r=24, b=85, t=160),
        barmode='overlay'
    )

    return fig

# --- Build frames for each sweep position ---
integral_frames = [
    go.Frame(
        data=make_normal_integration_fig(step).data,
        name=str(step),
        layout=make_normal_integration_fig(step).layout
    )
    for step in range(n_steps)
]

fig_integral = make_normal_integration_fig(0)
fig_integral.frames = integral_frames

# --- Play Button ---
fig_integral.update_layout(
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.32,
        'showactive': False,
        'buttons': [{
            'label': 'Play',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 50, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        }]
    }]
)

fig_integral.show()
```

```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# --- LLN for the Standard Normal! ---

np.random.seed(42)
N = 250  # Number of standard normal samples

# Generate samples from standard normal
samples = np.random.randn(N)
empirical_means = np.array([np.mean(samples[:i+1]) for i in range(N)])
# empirical_vars = np.array([np.var(samples[:i+1], ddof=0) for i in range(N)])

true_mean = 0.0
# true_var = 1.0

def make_lln_normal_fig(step):
    # step is 0-based (number of data points so far is step+1)
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.56, 0.44],
        subplot_titles=(
            "<br>Histogram of Standard Normal Samples<br> (as Number of Draws Increases)",
            "<br>Empirical Mean Converges to Theoretical Mean"
        )
    )

    data_so_far = samples[:step+1]
    
    # --- Left: Histogram and theoretical density ---
    hist_vals, bins = np.histogram(data_so_far, bins=15, range=[-4, 4], density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=hist_vals,
            width=(bins[1]-bins[0]),
            marker_color="#FFE066",
            name="Empirical Histogram",
            opacity=0.77,
            showlegend=True if step == 0 else False
        ),
        row=1, col=1
    )
    # Theoretical normal curve
    x_grid = np.linspace(-4, 4, 100)
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=norm.pdf(x_grid),
            mode="lines",
            line=dict(width=2, color="#23AFFF"),
            name="Standard Normal PDF",
            showlegend=True if step == 0 else False
        ),
        row=1, col=1
    )

    fig.update_xaxes(
        title_text="Value",
        range=[-4, 4],
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="Density",
        range=[0, np.max(hist_vals)*1.18 if len(hist_vals)>0 else 0.42],
        row=1, col=1
    )

    # --- Right: Bar chart for mean only --- 
    metrics_x = ["Theoretical Mean", "Empirical Mean"]
    metrics_y = [true_mean, empirical_means[step]]

    fig.add_trace(
        go.Bar(
            x=metrics_x,
            y=metrics_y,
            width=0.43,
            marker_color=["#23AFFF","#FFE066"],
            text=[f"{true_mean:.2f}", f"{empirical_means[step]:.2f}"],
            textposition="outside",
            opacity=0.92,
            showlegend=False
        ),
        row=1, col=2
    )

    # Draw convergence path for empirical mean only
    if step > 1:
        fig.add_trace(
            go.Scatter(
                x=["Empirical Mean"]*step,
                y=empirical_means[:step],
                mode="lines+markers",
                line=dict(color='#BB9900', width=2, dash='dot'),
                marker=dict(size=3, color='#FFD700'),
                name="Empirical Mean (trace)",
                showlegend=False,
                hoverinfo="y+x"
            ),
            row=1, col=2
        )

    # Highlight target band for the mean (for visibility)
    fig.add_shape(
        type="rect",
        xref="x2", yref="y2",
        x0=-0.4, x1=1.4, y0=true_mean-0.07, y1=true_mean+0.07,
        fillcolor="rgba(30,250,240,0.09)",
        line_width=0, row=1, col=2
    )

    fig.update_xaxes(
        title_text="",
        row=1, col=2,
        tickvals=metrics_x
    )
    fig.update_yaxes(
        title_text="Value",
        range=[-1.25, 3],
        row=1, col=2
    )

    # --- Layout ---
    fig.update_layout(
        height=450, width=900,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FAF7E0', size=15),
        title=dict(
            text="<br>Law of Large Numbers (LLN) for Standard Normal: Empirical Mean Converges<br>",
            y= .99, font=dict(color="#FAFAFA", size=23),
            x=0.5, xanchor="center", yanchor="top"
        ),
        margin=dict(l=45, r=22, b=80, t=98),
        barmode='group'
    )
    return fig

# --- Create Animation Frames ---
lln_normal_frames = [
    go.Frame(
        data=make_lln_normal_fig(n).data,
        name=str(n),
        layout=make_lln_normal_fig(n).layout
    )
    for n in range(N)
]

fig_lln_normal = make_lln_normal_fig(0)
fig_lln_normal.frames = lln_normal_frames

# --- Animation play button (no slider) ---
fig_lln_normal.update_layout(
    updatemenus=[
        {
            'type': 'buttons',
            'x': 0.48, 'y': -0.22,
            'showactive': False,
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 30, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 0}
                    }]
                }
            ]
        }
    ]
)

fig_lln_normal.show()
```

```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# --- Simulation Parameters ---
n_paths_total = 1000
n_paths_show = 100  # Number of paths to be animated one by one
n_steps = 100
dt = 1
mu = 0
sigma = 1

np.random.seed(42)
# Brownian Motion W_t = sum(increments)
increments = np.random.normal(mu, sigma * np.sqrt(dt), (n_paths_total, n_steps))
W = np.concatenate([np.zeros((n_paths_total, 1)), np.cumsum(increments, axis=1)], axis=1)
time_grid = np.arange(n_steps + 1)

T = n_steps  # final time

# --- Precompute theoretical normal distribution for W_T ~ N(0, T) ---
hist_bins = 30
w_T_all = W[:, T]
# Use a wide range for histogram bins to cover all paths
hist_range = [w_T_all.min() - 1, w_T_all.max() + 1]
bin_edges = np.linspace(hist_range[0], hist_range[1], hist_bins + 1)
hist_y_bins = 0.5 * (bin_edges[1:] + bin_edges[:-1])

theory_pdf = norm.pdf(hist_y_bins, 0, np.sqrt(T))

# Determine maximum density value for fixed x-axis on histogram
max_empirical_density = 0
n_paths_to_check = min(n_paths_show, n_paths_total)
for k in range(1, n_paths_to_check + 1):
    w_T_slice = W[:k, T]
    hist_k = np.histogram(w_T_slice, bins=bin_edges, density=True)
    max_empirical_density = max(max_empirical_density, max(hist_k[0]))

# Hardcode max for right xlim as requested (regardless of pdf/hist maximum)
max_pdf = 0.045  # <-- UPDATED from 0.04 to 0.045 as requested

# --- Figure Setup ---
fig = make_subplots(
    rows=1, cols=2,
    column_widths=[0.7, 0.3],
    subplot_titles=["Simulated Brownian Paths", f"Distribution of $W_T$ (T={T})"],
    horizontal_spacing=0.10
)

# --- Initial Traces (Placeholders for the first frame's data) ---

# Left Chart: All historical paths (muted) and the single new path (highlighted)
# Initial state: no paths shown (empty lists)
fig.add_trace(go.Scatter(
    x=[np.nan], y=[np.nan], mode='lines', line=dict(color='#00ffff', width=1),
    showlegend=False, name="Historical Paths (Muted)"
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=[np.nan], y=[np.nan], mode='lines', line=dict(color='magenta', width=3),
    showlegend=False, name="New Path (Highlighted)"
), row=1, col=1)

# Right Chart: Histogram (empty) and Theoretical PDF (full)
hist_init = go.Bar(
    y=hist_y_bins,
    x=np.zeros_like(hist_y_bins),
    width=(hist_y_bins[1] - hist_y_bins[0]) if len(hist_y_bins) > 1 else 0.5,
    orientation='h',
    marker=dict(color='rgba(100,255,100,0.6)'),
    name='Empirical'
)
fig.add_trace(hist_init, row=1, col=2)

theory_line_init = go.Scatter(
    x=theory_pdf,
    y=hist_y_bins,
    mode='lines',
    line=dict(color='magenta', width=3),
    name='Theoretical'
)
fig.add_trace(theory_line_init, row=1, col=2)

# --- Animation Frames ---
frames = []
path_indices = np.arange(n_paths_show)
for k in range(1, n_paths_show + 1):  # animate adding 1, 2, ..., n_paths_show paths

    # --- Left: Paths (Scatter) ---
    # 1. Historical Paths (1 to k-1) - Muted color

    if k > 1:
        # Get paths 1 to k-1
        paths_muted = W[path_indices[:k-1]]

        # Initialize x_muted and y_muted as float arrays from the beginning
        x_muted = np.tile(time_grid, k - 1).astype(float)
        y_muted = paths_muted.flatten().astype(float)

        # Indices where NaNs should be inserted (after every path)
        insert_indices = np.arange(n_steps + 1, len(x_muted), n_steps + 1)

        # Insert np.nan (float) into the float array
        x_muted = np.insert(x_muted, insert_indices, np.nan)
        y_muted = np.insert(y_muted, insert_indices, np.nan)
    else:
        # For k=1, no historical paths yet. Use sentinel NaNs initialized as floats.
        x_muted, y_muted = np.array([np.nan]), np.array([np.nan])

    scatter_muted = go.Scatter(
        x=x_muted,
        y=y_muted,
        mode='lines',
        line=dict(color='#00ffff', width=1),
        opacity=0.3,
        showlegend=False,
        hoverinfo='none',
    )

    # 2. New Path (k) - Highlighted color
    scatter_new = go.Scatter(
        x=time_grid.astype(float),
        y=W[k-1].astype(float),  # W[k-1] is the k-th path
        mode='lines',
        line=dict(color='magenta', width=3),
        name="New Path",
        showlegend=False,
        hoverinfo='none',
    )

    # --- Right: Distribution (Bar/Scatter) ---
    # 3. Histogram of the current k path endpoints at T
    w_T_slice = W[path_indices[:k], T]
    hist_k = np.histogram(w_T_slice, bins=bin_edges, density=True)
    hist_k_x = hist_k[0]  # Density values

    hist_trace = go.Bar(
        y=hist_y_bins,
        x=hist_k_x,
        orientation='h',
        marker=dict(color='rgba(100,255,100,0.6)'),
        hoverinfo='skip',
        name='Empirical',
    )

    # 4. Theoretical normal distribution for W_T ~ N(0, T)
    theory_line = go.Scatter(
        x=theory_pdf,
        y=hist_y_bins,
        mode='lines',
        line=dict(color='magenta', width=3),
        hoverinfo='skip',
        name='Theoretical',
    )

    # Combine traces for the frame data. The order must match the initial figure setup order.
    frames.append(go.Frame(
        data=[scatter_muted, scatter_new, hist_trace, theory_line],
        name=f"paths{k}",
        # Keep the title dynamically updated per frame
        layout=go.Layout(title_text=f"Convergence of Sample Paths to Theoretical Distribution at Final Time T ({k} paths shown)")
    ))

fig.frames = frames

# --- Layout ---
fig.update_layout(
    height=600, width=1000,
    title_text=f"Convergence of Sample Paths to Theoretical Distribution at Final Time T (0 paths shown)",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=True,
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.1)'),
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.10,
        'xanchor': 'center', 'yanchor': 'top',
        'direction': 'left',
        'showactive': False,
        'buttons': [{
            'label': '▶ Play',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 30, 'redraw': True},  # Smoother animation
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        }]
    }]
)

# Axes styling and limits
for c in [1, 2]:
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=1, col=c)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=1, col=c)

# --- Axes (fixed globally) ---
fig.update_xaxes(title_text='Time (t)', range=[0, n_steps], row=1, col=1)
fig.update_yaxes(title_text='$W_t$', range=[-20, 20], row=1, col=1)
fig.update_xaxes(title_text='Density', range=[0, max_pdf], row=1, col=2)  # <-- FIXED TO 0.045!
fig.update_yaxes(title_text='$W_T$', range=hist_range, row=1, col=2)  # Use calculated range

fig.show()
```

```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Risk-Neutral Heston Model Parameters (Illustrative Example) ---
r = 0.02    # risk-free rate
kappa = 1.5 # mean reversion speed
theta = 0.04 # long run variance
xi = 0.3    # volatility of variance ("vol of vol")
rho = -0.7  # correlation between Brownian motions
v0 = 0.04   # initial variance
S0_1 = 100  # initial price Asset 1
S0_2 = 100  # initial price Asset 2

# Simulation settings
T = 1.0                  # time horizon
n_steps = 100
dt = T / n_steps
n_paths_total = 3000
n_paths_show = 100        # Animate showing up to n_paths_show paths

np.random.seed(42)

# Simulate two Heston processes (Assets 1 and 2), correlated via rho
def simulate_heston_paths(S0, v0, r, kappa, theta, xi, rho, dt, n_steps, n_paths):
    S = np.zeros((n_paths, n_steps + 1))
    v = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    v[:, 0] = v0

    # Pre-generate correlated normals
    Z1_raw = np.random.normal(size=(n_paths, n_steps))
    Z2_raw = np.random.normal(size=(n_paths, n_steps))
    Z1 = Z1_raw
    Z2 = rho * Z1_raw + np.sqrt(1 - rho ** 2) * Z2_raw

    for t in range(n_steps):
        dW1 = np.sqrt(dt) * Z1[:, t]
        dW2 = np.sqrt(dt) * Z2[:, t]
        v_prev = np.maximum(v[:, t], 0)  # Full truncation for variance
        v_sqrt = np.sqrt(v_prev)

        # Euler-Maruyama for volatility
        v[:, t + 1] = v_prev + kappa * (theta - v_prev) * dt + xi * v_sqrt * dW2
        v[:, t + 1] = np.maximum(v[:, t + 1], 0)

        # Euler-Maruyama for price dynamics
        S[:, t + 1] = S[:, t] * np.exp((r - 0.5 * v_prev) * dt + v_sqrt * dW1)
    return S, v

# Precompute Heston price paths (all at once for efficiency)
S1, v1 = simulate_heston_paths(S0_1, v0, r, kappa, theta, xi, rho, dt, n_steps, n_paths_total)
S2, v2 = simulate_heston_paths(S0_2, v0, r, kappa, theta, xi, rho, dt, n_steps, n_paths_total)
time_grid = np.linspace(0, T, n_steps + 1)

# --- Payoff: Best-of Rainbow Call Option ---
K = 100
payoff = lambda ST1, ST2: np.maximum(np.maximum(ST1, ST2) - K, 0)
discount = np.exp(-r * T)
payoff_all = payoff(S1[:, -1], S2[:, -1])
price_all = discount * payoff_all

# --- Compute Cumulative Monte Carlo Means ---
cum_prices = np.cumsum(price_all[:n_paths_show]) / np.arange(1, n_paths_show + 1)
true_mean_price = np.mean(price_all)

# --- Figure Setup ---
fig = make_subplots(
    rows=1, cols=2,
    column_widths=[0.7, 0.3],
    subplot_titles=[
        "Simulated Heston Asset Paths (Rainbow Option)",
        "Convergence of MC Price Estimate"
    ],
    horizontal_spacing=0.10
)

# --- Initial Traces (Placeholders for frame data) ---

# 1. Historical Asset 1 Paths (Muted)
fig.add_trace(go.Scatter(
    x=[np.nan], y=[np.nan], mode='lines',
    line=dict(color='#00ffff', width=1), name='Asset 1 Paths (Historical)', showlegend=True, opacity=0.12
), row=1, col=1)

# 2. Historical Asset 2 Paths (Muted)
fig.add_trace(go.Scatter(
    x=[np.nan], y=[np.nan], mode='lines',
    line=dict(color='magenta', width=1), name='Asset 2 Paths (Historical)', showlegend=True, opacity=0.12
), row=1, col=1)

# 3. Most Recent Asset 1 Path (Highlighted)
fig.add_trace(go.Scatter(
    x=[np.nan], y=[np.nan], mode='lines',
    line=dict(color='#00ffff', width=3), name='Asset 1 Path (New)', showlegend=False
), row=1, col=1)

# 4. Most Recent Asset 2 Path (Highlighted)
fig.add_trace(go.Scatter(
    x=[np.nan], y=[np.nan], mode='lines',
    line=dict(color='magenta', width=3), name='Asset 2 Path (New)', showlegend=False
), row=1, col=1)

# 5. Running MC Price Estimate (Right Chart)
fig.add_trace(go.Scatter(
    x=[np.nan], y=[np.nan],
    mode='lines+markers',
    marker=dict(size=7, color='white'),
    line=dict(color='white', width=3),
    name='MC Mean Price', showlegend=False
), row=1, col=2)

# 6. Theoretical Mean (Reference Line on Right Chart)
fig.add_trace(go.Scatter(
    x=[0, n_paths_show], y=[true_mean_price, true_mean_price],
    mode='lines',
    line=dict(color='yellow', width=2, dash='dash'),
    name='MC True Mean', showlegend=True
), row=1, col=2)

# --- Animation Frames ---
frames = []

for k in range(1, n_paths_show + 1):

    # ------- LEFT CHART: Plot Heston Asset Price Paths --------

    # 1. All previous (historical) Asset 1 paths up to k-1, faded
    if k > 1:
        S1_hist = S1[:k-1]
        time_hist = np.tile(time_grid, k - 1)
        S1h_flat = S1_hist.T.ravel()
        x1_hist = np.insert(time_hist, np.arange(n_steps + 1, (n_steps + 1) * (k - 1), n_steps + 1), np.nan)
        y1_hist = np.insert(S1h_flat, np.arange(n_steps + 1, (n_steps + 1) * (k - 1), n_steps + 1), np.nan)
    else:
        x1_hist, y1_hist = np.array([np.nan]), np.array([np.nan])

    trace1_hist = go.Scatter(
        x=x1_hist,
        y=y1_hist,
        mode='lines',
        line=dict(color='#00ffff', width=1),
        name='Asset 1 Paths (Historical)',
        opacity=0.12,
        showlegend=False,
        hoverinfo='none'
    )

    # 2. All previous (historical) Asset 2 paths up to k-1, faded
    if k > 1:
        S2_hist = S2[:k-1]
        time_hist = np.tile(time_grid, k - 1)
        S2h_flat = S2_hist.T.ravel()
        x2_hist = np.insert(time_hist, np.arange(n_steps + 1, (n_steps + 1) * (k - 1), n_steps + 1), np.nan)
        y2_hist = np.insert(S2h_flat, np.arange(n_steps + 1, (n_steps + 1) * (k - 1), n_steps + 1), np.nan)
    else:
        x2_hist, y2_hist = np.array([np.nan]), np.array([np.nan])

    trace2_hist = go.Scatter(
        x=x2_hist,
        y=y2_hist,
        mode='lines',
        line=dict(color='magenta', width=1),
        name='Asset 2 Paths (Historical)',
        opacity=0.12,
        showlegend=False,
        hoverinfo='none'
    )

    # 3. The latest Asset 1 Heston price path, highlighted
    trace3_new_1 = go.Scatter(
        x=time_grid,
        y=S1[k-1],
        mode='lines',
        line=dict(color='#00ffff', width=3),
        name='Asset 1 Path (New)',
        opacity=1.0,
        showlegend=False,
        hoverinfo='none'
    )

    # 4. The latest Asset 2 Heston price path, highlighted
    trace4_new_2 = go.Scatter(
        x=time_grid,
        y=S2[k-1],
        mode='lines',
        line=dict(color='magenta', width=3),
        name='Asset 2 Path (New)',
        opacity=1.0,
        showlegend=False,
        hoverinfo='none'
    )

    # ------- RIGHT CHART: MC Estimate --------

    # 5. Running MC price estimate (discounted payoff)
    trace5_mc_mean = go.Scatter(
        x=np.arange(1, k + 1),
        y=cum_prices[:k],
        mode='lines+markers',
        marker=dict(size=7, color='white'),
        line=dict(color='white', width=3),
        name='MC Mean',
        hoverinfo='x+y',
        showlegend=False
    )

    frames.append(go.Frame(
        data=[
            trace1_hist,
            trace2_hist,
            trace3_new_1,
            trace4_new_2,
            trace5_mc_mean
        ],
        name=f"step{k}",
        layout=go.Layout(
            title_text=f"Rainbow MC: {k} Sample Path{'s' if k > 1 else ''} (Right: Price Estimate Convergence)"
        )
    ))

fig.frames = frames

# Set axes labels and ranges
fig.update_layout(
    height=650,
    width=1000,
    title_text=f"Rainbow MC: 0 Sample Paths",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=True,
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.1)', itemsizing='constant'),
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.10,
        'xanchor': 'center', 'yanchor': 'top',
        'direction': 'left',
        'showactive': False,
        'buttons': [{
            'label': '▶ Play',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 250, 'redraw': True},
                'fromcurrent': True,
                'transition': {'duration': 0}
            }]
        }]
    }]
)

# Grid and axes for both subplots
for c in [1, 2]:
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=1, col=c)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.3)', row=1, col=c)

# Left chart (Heston price paths) axis scaling
all_prices = np.concatenate([S1[:, :], S2[:, :]])
fig.update_xaxes(title_text='Time $t$', range=[0, T], row=1, col=1)
fig.update_yaxes(
    title_text='Asset Price',
    range=[np.nanmin(all_prices) * 0.8, np.nanmax(all_prices) * 1.2],
    row=1, col=1
)

# Right chart (MC mean estimate) axis scaling
fig.update_xaxes(title_text='MC Sample Size', range=[0, n_paths_show + 1], row=1, col=2)
fig.update_yaxes(
    title_text='Discounted Payoff Estimate',
    range=[
        np.min(cum_prices) * 0.95,
        np.max(cum_prices) * 1.05
    ],
    row=1, col=2
)

fig.show()
```


---

## 16. How to Trade with the Kelly Criterion (Lecture 36)

`2025 Video Lectures/36. How to Trade with the Kelly Criterion/how_to_trade_with_the_kelly_criterion.ipynb` — Kelly criterion bet sizing.

```python
%%html
<style>
/* Overwrite the hard-coded white background for ipywidgets */
.cell-output-ipywidget-background {
    background-color: transparent !important;
}
/* Set widget foreground text and color to match the VS Code dark theme */
:root {
    --jp-widgets-color: var(--vscode-editor-foreground);
    --jp-widgets-font-size: var(--vscode-editor-font-size);
}
</style>
```

```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set random seed for reproducibility
np.random.seed(42)

# Trading system parameters
n_trades = 1000
n_paths = 100
initial_capital = 1000

# Kelly system parameters
win_rate = 0.51
win_pct = 0.05  # 5% return on wins
loss_pct = 0.05  # 5% loss on losses
odds = win_pct/loss_pct  # Net odds ratio
kelly_fraction = (odds * win_rate - (1-win_rate)) / odds  # Kelly criterion formula

# Non-ergodic system parameters (multiplicative returns) 
win_rate_pct = 0.51
bet_pct = 0.75  # Percentage of capital bet size

# Simulate Kelly criterion wealth paths
paths_kelly = np.zeros((n_paths, n_trades + 1))
paths_kelly[:, 0] = initial_capital

# Simulate non-ergodic wealth paths
paths_nonergodic = np.zeros((n_paths, n_trades + 1))
paths_nonergodic[:, 0] = initial_capital

for path in range(n_paths):
    # Kelly system simulation
    capital_kelly = initial_capital
    
    # Non-ergodic system simulation
    capital_nonergodic = initial_capital
    
    for i in range(n_trades):
        # Kelly system trades
        is_win = np.random.random() < win_rate
        bet_amount = capital_kelly * kelly_fraction
        
        if is_win:
            capital_kelly += bet_amount * win_pct
        else:
            capital_kelly -= bet_amount * loss_pct
        paths_kelly[path, i + 1] = capital_kelly
        
        # Non-ergodic system trades
        is_win_pct = np.random.random() < win_rate_pct
        bet_amount = capital_nonergodic * bet_pct
        
        if is_win_pct:
            capital_nonergodic += bet_amount * win_pct
        else:
            capital_nonergodic -= bet_amount * loss_pct
        paths_nonergodic[path, i + 1] = capital_nonergodic

# Calculate mean paths
mean_path_kelly = np.mean(paths_kelly, axis=0)
mean_path_nonergodic = np.mean(paths_nonergodic, axis=0)

# Create figure with subplots
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=(f'Kelly Criterion Paths (f*={kelly_fraction:.2%})',
                                  'Non-Ergodic Wealth Paths'))

# Plot Kelly system paths
for i in range(n_paths):
    fig.add_trace(
        go.Scatter(x=np.arange(n_trades + 1), y=paths_kelly[i],
                   line=dict(color='#00FF00', width=1),
                   opacity=0.1,
                   showlegend=False),
        row=1, col=1
    )

# Plot Kelly system mean path
fig.add_trace(
    go.Scatter(x=np.arange(n_trades + 1), y=mean_path_kelly,
               name='Mean Path (Kelly)',
               line=dict(color='#00FF00', width=4)),
    row=1, col=1
)

# Plot non-ergodic system paths
for i in range(n_paths):
    fig.add_trace(
        go.Scatter(x=np.arange(n_trades + 1), y=paths_nonergodic[i],
                   line=dict(color='#FF00FF', width=1),
                   opacity=0.1,
                   showlegend=False),
        row=1, col=2
    )

# Plot non-ergodic system mean path
fig.add_trace(
    go.Scatter(x=np.arange(n_trades + 1), y=mean_path_nonergodic,
               name='Mean Path (Non-Ergodic)',
               line=dict(color='#FF00FF', width=4)),
    row=1, col=2
)

# Update layout
fig.update_layout(
    showlegend=False,
    width=1000,
    height=500,
    title_text='Trading System Analysis: Kelly Criterion vs Non-Ergodic Systems',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    barmode='overlay',
    legend=dict(
        yanchor="bottom",
        y=0.01,
        xanchor="right",
        x=0.99
    )
)

# Update axes
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Trades',
                 row=1, col=1)
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Trades',
                 row=1, col=2)
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Capital ($)',
                 row=1, col=1)
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Capital ($)',
                 row=1, col=2)

fig.show()
```

```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set random seed for reproducibility
np.random.seed(42)

# Trading system parameters
n_trades = 1000
n_paths = 100
initial_capital = 1000
win_rate = 0.51
avg_win = 20
avg_loss = 20

# Generate win/loss amounts from normal distributions
win_amounts = np.random.normal(avg_win, avg_win/2, n_trades)
loss_amounts = np.random.normal(avg_loss, avg_loss/2, n_trades)

# Generate sample trades
trades = np.random.random(n_trades) < win_rate
pl_outcomes = np.where(trades, win_amounts, -loss_amounts)

# Simulate multiple wealth paths with small fixed bet size
paths_small = np.zeros((n_paths, n_trades + 1))
paths_small[:, 0] = initial_capital
small_bet = 10  # $10 fixed bet size

# Simulate multiple wealth paths with large fixed bet size
paths_large = np.zeros((n_paths, n_trades + 1))
paths_large[:, 0] = initial_capital
large_bet = 30  # $30 fixed bet size

for path in range(n_paths):
    capital_small = initial_capital
    capital_large = initial_capital
    
    for i in range(n_trades):
        # Generate trade outcomes
        is_win = np.random.random() < win_rate
        trade_amount = np.random.normal(avg_win, avg_win/2) if is_win else np.random.normal(avg_loss, avg_loss/2)
        
        # Small fixed bet size path
        trade_pl_small = small_bet * trade_amount if is_win else -small_bet * trade_amount
        capital_small += trade_pl_small
        paths_small[path, i + 1] = capital_small
        
        # Large fixed bet size path
        trade_pl_large = large_bet * trade_amount if is_win else -large_bet * trade_amount
        capital_large += trade_pl_large
        paths_large[path, i + 1] = capital_large

# Calculate mean paths
mean_path_small = np.mean(paths_small, axis=0)
mean_path_large = np.mean(paths_large, axis=0)

# Create figure with subplots
fig = make_subplots(rows=1, cols=3,
                    subplot_titles=('P/L Distribution',
                                  'Wealth Paths (Small Bet)',
                                  'Wealth Paths (Large Bet)'))

# Split P/L outcomes into wins and losses
wins = pl_outcomes[pl_outcomes >= 0]
losses = pl_outcomes[pl_outcomes < 0]

# Plot P/L distribution
fig.add_trace(
    go.Histogram(x=losses,
                 nbinsx=25,
                 name='Losses',
                 marker_color='#FF0000', opacity=.75),
    row=1, col=1
)

fig.add_trace(
    go.Histogram(x=wins,
                 nbinsx=25,
                 name='Wins',
                 marker_color='#00FF00', opacity=.75),
    row=1, col=1
)

# Plot small bet size paths
for i in range(n_paths):
    fig.add_trace(
        go.Scatter(x=np.arange(n_trades + 1), y=paths_small[i],
                   line=dict(color='#00FFFF', width=1),
                   opacity=0.1,
                   showlegend=False),
        row=1, col=2
    )

# Plot small bet size mean path
fig.add_trace(
    go.Scatter(x=np.arange(n_trades + 1), y=mean_path_small,
               name='Mean Path (Small)',
               line=dict(color='#00FFFF', width=4)),
    row=1, col=2
)

# Plot large bet size paths
for i in range(n_paths):
    fig.add_trace(
        go.Scatter(x=np.arange(n_trades + 1), y=paths_large[i],
                   line=dict(color='#FF00FF', width=1),
                   opacity=0.1,
                   showlegend=False),
        row=1, col=3
    )

# Plot large bet size mean path
fig.add_trace(
    go.Scatter(x=np.arange(n_trades + 1), y=mean_path_large,
               name='Mean Path (Large)',
               line=dict(color='#FF00FF', width=4)),
    row=1, col=3
)

# Update layout
fig.update_layout(
    showlegend=False,
    width=1500,
    height=500,
    title_text='Trading System Analysis: Small vs Large Fixed Bet Size',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    barmode='overlay',
    legend=dict(
        yanchor="bottom",
        y=0.01,
        xanchor="right",
        x=0.99
    )
)

# Update axes
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='P/L',
                 row=1, col=1)
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Trades',
                 row=1, col=2)
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Trades',
                 row=1, col=3)
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Frequency',
                 row=1, col=1)
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Capital ($)',
                 row=1, col=2)
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Capital ($)',
                 row=1, col=3)

fig.show()
```

```python
# Calculate average final values for small and large bet systems
small_bet_final_avg = np.mean(paths_small[:,-1])
large_bet_final_avg = np.mean(paths_large[:,-1])

print(f"After {n_trades} trades:")
print(f"Small Bet System Average Value: ${small_bet_final_avg:.2f}")
print(f"Large Bet System Average Value: ${large_bet_final_avg:.2f}")
```

```python
# Set random seed for reproducibility
np.random.seed(42)

# Trading system parameters
n_trades = 1000
n_paths = 100
initial_capital = 1000

# Ergodic system parameters (additive returns)
win_rate = 0.51
avg_win = 20  # Fixed dollar amount
avg_loss = 20  # Fixed dollar amount

# Non-ergodic system parameters (multiplicative returns)
win_rate_pct = 0.51
win_pct = 0.05  # 5% return on wins
loss_pct = 0.05  # 5% loss on losses

# Simulate ergodic wealth paths (additive returns)
paths_ergodic = np.zeros((n_paths, n_trades + 1))
paths_ergodic[:, 0] = initial_capital
bet_size = 20  # Fixed bet size

# Simulate non-ergodic wealth paths (multiplicative returns)
paths_nonergodic = np.zeros((n_paths, n_trades + 1))
paths_nonergodic[:, 0] = initial_capital
bet_pct = 0.75  # Percentage of capital bet size

for path in range(n_paths):
    # Ergodic system simulation
    capital_ergodic = initial_capital
    
    # Non-ergodic system simulation
    capital_nonergodic = initial_capital
    
    for i in range(n_trades):
        # Ergodic system trades (additive returns)
        is_win = np.random.random() < win_rate
        trade_amount = np.random.normal(avg_win, avg_win/2) if is_win else np.random.normal(avg_loss, avg_loss/2)
        
        trade_pl = bet_size * trade_amount if is_win else -bet_size * trade_amount
        capital_ergodic += trade_pl
        paths_ergodic[path, i + 1] = capital_ergodic
        
        # Non-ergodic system trades (multiplicative returns)
        is_win_pct = np.random.random() < win_rate_pct
        
        bet_amount = capital_nonergodic * bet_pct
        if is_win_pct:
            capital_nonergodic += bet_amount * win_pct
        else:
            capital_nonergodic -= bet_amount * loss_pct
        paths_nonergodic[path, i + 1] = capital_nonergodic

# Calculate mean paths
mean_path_ergodic = np.mean(paths_ergodic, axis=0)
mean_path_nonergodic = np.mean(paths_nonergodic, axis=0)

# Create figure with subplots
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=('Ergodic Wealth Paths',
                                  'Non-Ergodic Wealth Paths'))

# Plot ergodic system paths
for i in range(n_paths):
    fig.add_trace(
        go.Scatter(x=np.arange(n_trades + 1), y=paths_ergodic[i],
                   line=dict(color='#00FFFF', width=1),
                   opacity=0.1,
                   showlegend=False),
        row=1, col=1
    )

# Plot ergodic system mean path
fig.add_trace(
    go.Scatter(x=np.arange(n_trades + 1), y=mean_path_ergodic,
               name='Mean Path (Ergodic)',
               line=dict(color='#00FFFF', width=4)),
    row=1, col=1
)

# Plot non-ergodic system paths
for i in range(n_paths):
    fig.add_trace(
        go.Scatter(x=np.arange(n_trades + 1), y=paths_nonergodic[i],
                   line=dict(color='#FF00FF', width=1),
                   opacity=0.1,
                   showlegend=False),
        row=1, col=2
    )

# Plot non-ergodic system mean path
fig.add_trace(
    go.Scatter(x=np.arange(n_trades + 1), y=mean_path_nonergodic,
               name='Mean Path (Non-Ergodic)',
               line=dict(color='#FF00FF', width=4)),
    row=1, col=2
)

# Update layout
fig.update_layout(
    showlegend=False,
    width=1250,
    height=500,
    title_text='Trading System Analysis: Ergodic vs Non-Ergodic Systems',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    barmode='overlay',
    legend=dict(
        yanchor="bottom",
        y=0.01,
        xanchor="right",
        x=0.99
    )
)

# Update axes
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Trades',
                 row=1, col=1)
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Trades',
                 row=1, col=2)
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Capital ($)',
                 row=1, col=1)
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Capital ($)',
                 row=1, col=2)

fig.show()
```

```python
# Calculate mean final wealth for both strategies
mean_final_wealth_ergodic = np.mean(paths_ergodic[:,-1])
mean_final_wealth_nonergodic = np.mean(paths_nonergodic[:,-1])

print(f"Mean final wealth after {n_trades} trades:")
print(f"Ergodic strategy: ${mean_final_wealth_ergodic:,.2f}")
print(f"Non-ergodic strategy: ${mean_final_wealth_nonergodic:,.2f}")
```

```python
# Calculate probability of ending above initial capital
prob_above_initial_ergodic = np.mean(paths_ergodic[:,-1] > initial_capital)
prob_above_initial_nonergodic = np.mean(paths_nonergodic[:,-1] > initial_capital)

print(f"Probability of ending above ${initial_capital:,.0f} after {n_trades} trades:")
print(f"Ergodic strategy: {prob_above_initial_ergodic:.1%}")
print(f"Non-ergodic strategy: {prob_above_initial_nonergodic:.1%}")
```

```python
# Set random seed for reproducibility
np.random.seed(42)

# Trading system parameters
n_trades = 1000
n_paths = 100
initial_capital = 1000

# Kelly system parameters
win_rate = 0.51
win_pct = 0.05  # 5% return on wins
loss_pct = 0.05  # 5% loss on losses
odds = win_pct/loss_pct  # Net odds ratio
kelly_fraction = (odds * win_rate - (1-win_rate)) / odds  # Kelly criterion formula

# Non-ergodic system parameters (multiplicative returns) 
win_rate_pct = 0.51
bet_pct = 0.75  # Percentage of capital bet size

# Simulate Kelly criterion wealth paths
paths_kelly = np.zeros((n_paths, n_trades + 1))
paths_kelly[:, 0] = initial_capital

# Simulate non-ergodic wealth paths
paths_nonergodic = np.zeros((n_paths, n_trades + 1))
paths_nonergodic[:, 0] = initial_capital

for path in range(n_paths):
    # Kelly system simulation
    capital_kelly = initial_capital
    
    # Non-ergodic system simulation
    capital_nonergodic = initial_capital
    
    for i in range(n_trades):
        # Kelly system trades
        is_win = np.random.random() < win_rate
        bet_amount = capital_kelly * kelly_fraction
        
        if is_win:
            capital_kelly += bet_amount * win_pct
        else:
            capital_kelly -= bet_amount * loss_pct
        paths_kelly[path, i + 1] = capital_kelly
        
        # Non-ergodic system trades
        is_win_pct = np.random.random() < win_rate_pct
        bet_amount = capital_nonergodic * bet_pct
        
        if is_win_pct:
            capital_nonergodic += bet_amount * win_pct
        else:
            capital_nonergodic -= bet_amount * loss_pct
        paths_nonergodic[path, i + 1] = capital_nonergodic

# Calculate mean paths
mean_path_kelly = np.mean(paths_kelly, axis=0)
mean_path_nonergodic = np.mean(paths_nonergodic, axis=0)

# Create figure with subplots
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=(f'Kelly Criterion Paths (f*={kelly_fraction:.2%})',
                                  'Non-Ergodic Wealth Paths'))

# Plot Kelly system paths
for i in range(n_paths):
    fig.add_trace(
        go.Scatter(x=np.arange(n_trades + 1), y=paths_kelly[i],
                   line=dict(color='#00FF00', width=1),
                   opacity=0.1,
                   showlegend=False),
        row=1, col=1
    )

# Plot Kelly system mean path
fig.add_trace(
    go.Scatter(x=np.arange(n_trades + 1), y=mean_path_kelly,
               name='Mean Path (Kelly)',
               line=dict(color='#00FF00', width=4)),
    row=1, col=1
)

# Plot non-ergodic system paths
for i in range(n_paths):
    fig.add_trace(
        go.Scatter(x=np.arange(n_trades + 1), y=paths_nonergodic[i],
                   line=dict(color='#FF00FF', width=1),
                   opacity=0.1,
                   showlegend=False),
        row=1, col=2
    )

# Plot non-ergodic system mean path
fig.add_trace(
    go.Scatter(x=np.arange(n_trades + 1), y=mean_path_nonergodic,
               name='Mean Path (Non-Ergodic)',
               line=dict(color='#FF00FF', width=4)),
    row=1, col=2
)

# Update layout
fig.update_layout(
    showlegend=False,
    width=1250,
    height=500,
    title_text='Trading System Analysis: Kelly Criterion vs Non-Ergodic Systems',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    barmode='overlay',
    legend=dict(
        yanchor="bottom",
        y=0.01,
        xanchor="right",
        x=0.99
    )
)

# Update axes
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Trades',
                 row=1, col=1)
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Trades',
                 row=1, col=2)
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Capital ($)',
                 row=1, col=1)
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Capital ($)',
                 row=1, col=2)

fig.show()
```

```python
# Calculate probability of ending above initial capital
kelly_above = np.sum(paths_kelly[:, -1] > initial_capital) / n_paths
kelly_below = np.sum(paths_kelly[:, -1] < initial_capital) / n_paths

nonergodic_above = np.sum(paths_nonergodic[:, -1] > initial_capital) / n_paths
nonergodic_below = np.sum(paths_nonergodic[:, -1] < initial_capital) / n_paths

print("Kelly Criterion System:")
print(f"Probability of ending above initial capital: {kelly_above:.1%}")
print(f"Probability of ending below initial capital: {kelly_below:.1%}")
print("\nNon-Ergodic System:")
print(f"Probability of ending above initial capital: {nonergodic_above:.1%}")
print(f"Probability of ending below initial capital: {nonergodic_below:.1%}")

# Calculate median final wealth
kelly_median = np.median(paths_kelly[:, -1])
nonergodic_median = np.median(paths_nonergodic[:, -1])

print(f"\nMedian final wealth:")
print(f"Kelly Criterion: ${kelly_median:.2f}")
print(f"Non-Ergodic: ${nonergodic_median:.2f}")
```

```python
# Generate OU processes for probabilities and trade sizes
def ou_process(n_steps, mu, theta, sigma, initial):
    dt = 1
    x = np.zeros(n_steps)
    x[0] = initial
    for t in range(1, n_steps):
        dx = theta * (mu - x[t-1]) * dt + sigma * np.random.normal(0, np.sqrt(dt))
        x[t] = x[t-1] + dx
    return x

n_steps = 1000
# Parameters for probability OU processes
p_win_mu, p_win_theta, p_win_sigma = 0.6, 0.1, 0.05
p_lose_mu, p_lose_theta, p_lose_sigma = 0.4, 0.1, 0.05

# Parameters for trade size OU processes
win_size_mu, win_size_theta, win_size_sigma = 2.0, 0.1, 0.2
lose_size_mu, lose_size_theta, lose_size_sigma = 1.0, 0.1, 0.1

# Generate processes
p_win = np.clip(ou_process(n_steps, p_win_mu, p_win_theta, p_win_sigma, p_win_mu), 0.4, 0.8)
p_lose = 1 - p_win
win_size = ou_process(n_steps, win_size_mu, win_size_theta, win_size_sigma, win_size_mu)
lose_size = -np.abs(ou_process(n_steps, lose_size_mu, lose_size_theta, lose_size_sigma, lose_size_mu))

# Create figure with two subplots side by side
fig = make_subplots(rows=1, cols=2, 
                    subplot_titles=("Win/Loss Probabilities", "Average Trade Sizes"))

# Add probability traces to first subplot
fig.add_trace(
    go.Scatter(x=np.arange(n_steps), y=p_win,
               name="Probability of Winning",
               line=dict(color='#00FF00', width=2)),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=np.arange(n_steps), y=p_lose,
               name="Probability of Losing",
               line=dict(color='#FF0000', width=2)),
    row=1, col=1
)

# Add trade size traces to second subplot
fig.add_trace(
    go.Scatter(x=np.arange(n_steps), y=win_size,
               name="Average Winning Trade",
               line=dict(color='#00FFFF', width=2)),
    row=1, col=2
)

fig.add_trace(
    go.Scatter(x=np.arange(n_steps), y=lose_size,
               name="Average Losing Trade",
               line=dict(color='#FF69B4', width=2)),
    row=1, col=2
)

# Update layout
fig.update_layout(
    title_text="Evolution of Trading Parameters Over Time",
    title_x=0.5,
    width=1250,
    height=500,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes
fig.update_xaxes(title_text="Time",
                 showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)')

fig.update_yaxes(title_text="Probability",
                 range=[0, 1],
                 showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 row=1, col=1)

fig.update_yaxes(title_text="Trade Size ($)",
                 showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 row=1, col=2)

fig.show()
```

```python
# Set parameters for correct vs incorrect estimation
true_win_rate = 0.51
true_win_pct = 0.5
true_loss_pct = 0.5
true_odds = true_win_pct/true_loss_pct

# Correct estimation
correct_kelly = (true_odds * true_win_rate - (1-true_win_rate)) / true_odds

# Incorrect estimation (overconfident)
estimated_win_rate = 0.55  # Overestimating win rate
incorrect_kelly = (true_odds * estimated_win_rate - (1-estimated_win_rate)) / true_odds

# Simulate paths
paths_correct = np.zeros((n_paths, n_trades + 1))
paths_incorrect = np.zeros((n_paths, n_trades + 1))
paths_correct[:, 0] = initial_capital
paths_incorrect[:, 0] = initial_capital

for path in range(n_paths):
    capital_correct = initial_capital
    capital_incorrect = initial_capital
    
    for i in range(n_trades):
        # Generate true outcome based on true win rate
        is_win = np.random.random() < true_win_rate
        
        # Correct Kelly system
        bet_amount_correct = capital_correct * correct_kelly
        if is_win:
            capital_correct += bet_amount_correct * true_win_pct
        else:
            capital_correct -= bet_amount_correct * true_loss_pct
        paths_correct[path, i + 1] = capital_correct
        
        # Incorrect Kelly system (using overconfident fraction)
        bet_amount_incorrect = capital_incorrect * incorrect_kelly
        if is_win:
            capital_incorrect += bet_amount_incorrect * true_win_pct
        else:
            capital_incorrect -= bet_amount_incorrect * true_loss_pct
        paths_incorrect[path, i + 1] = capital_incorrect

# Calculate mean paths
mean_path_correct = np.mean(paths_correct, axis=0)
mean_path_incorrect = np.mean(paths_incorrect, axis=0)

# Create figure
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=(f'Correct Kelly Estimation (f*={correct_kelly:.2%})',
                                  f'Incorrect Kelly Estimation (f*={incorrect_kelly:.2%})'))

# Plot correct estimation paths
for i in range(n_paths):
    fig.add_trace(
        go.Scatter(x=np.arange(n_trades + 1), y=paths_correct[i],
                   line=dict(color='#00FF00', width=1),
                   opacity=0.1,
                   showlegend=False),
        row=1, col=1
    )

# Plot correct estimation mean path
fig.add_trace(
    go.Scatter(x=np.arange(n_trades + 1), y=mean_path_correct,
               name='Mean Path (Correct)',
               line=dict(color='#00FF00', width=4)),
    row=1, col=1
)

# Plot incorrect estimation paths
for i in range(n_paths):
    fig.add_trace(
        go.Scatter(x=np.arange(n_trades + 1), y=paths_incorrect[i],
                   line=dict(color='#FF0000', width=1),
                   opacity=0.1,
                   showlegend=False),
        row=1, col=2
    )

# Plot incorrect estimation mean path
fig.add_trace(
    go.Scatter(x=np.arange(n_trades + 1), y=mean_path_incorrect,
               name='Mean Path (Incorrect)',
               line=dict(color='#FF0000', width=4)),
    row=1, col=2
)

# Update layout
fig.update_layout(
    showlegend=False,
    width=1250,
    height=500,
    title_text='Impact of Kelly Criterion Estimation Error',
    title_x=0.5,
    title_font_size=20,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

# Update axes
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Trades')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                 zeroline=True, zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.5)',
                 title_text='Capital ($)')

fig.show()

# Print statistics
print("\nFinal Wealth Statistics:")
print(f"Correct Estimation - Median Final Wealth: ${np.median(paths_correct[:, -1]):.2f}")
print(f"Incorrect Estimation - Median Final Wealth: ${np.median(paths_incorrect[:, -1]):.2f}")
print(f"\nProbability of Ending Above Initial Capital:")
print(f"Correct Estimation: {np.sum(paths_correct[:, -1] > initial_capital) / n_paths:.1%}")
print(f"Incorrect Estimation: {np.sum(paths_incorrect[:, -1] > initial_capital) / n_paths:.1%}")
```


---

## 17. Analyzing Stock Returns with Principal Component Analysis in Python (Lecture 17)

`2025 Video Lectures/17. Analyzing Stock Returns with Principal Component Analysis in Python/PCA-Stock-Returns-main/PCA-Stock-Returns-main/pca_stock_returns_analysis.ipynb` — PCA of stock returns (SVD and covariance eigendecomposition) for risk factors and decorrelation.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
```

```python
# Load all sheets from the Excel file
excel_file = pd.ExcelFile('stock_returns_2021_2022.xlsx')
sheet_names = excel_file.sheet_names

# Initialize empty list to store dataframes
dfs = []

# Read each sheet and add to list
for sheet in sheet_names:
    df_sheet = pd.read_excel(excel_file, sheet_name=sheet)
    df_sheet['Stock'] = sheet  # Add column to identify stock
    dfs.append(df_sheet)

# Concatenate all dataframes
df = pd.concat(dfs, ignore_index=True)

# Group by date and pivot to get stocks as columns
df = df.pivot_table(
    index='Date',
    columns='Stock', 
    values='Close',
    aggfunc='first'
).reset_index()

# Sort by date in ascending order to ensure proper return calculation
df = df.sort_values('Date')

# Calculate log returns for each stock column (excluding Date)
stock_columns = df.columns.difference(['Date'])
# Using shift(1) since data is in ascending order - newer minus older prices
df[stock_columns] = np.log(df[stock_columns] / df[stock_columns].shift(1))

# Drop first row since it will have NaN values after calculating returns
df = df.dropna()

# Display basic information about the dataset
print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
display(df.head())

# Check for missing values
print("\nMissing values:")
display(df.isnull().sum())
```

```python
# Calculate correlation matrix excluding the Date column
correlation_matrix = df.drop('Date', axis=1).corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
im = plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
plt.colorbar(im)

# Add correlation values as text annotations
for i in range(len(correlation_matrix)):
    for j in range(len(correlation_matrix)):
        text = plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                       ha='center', va='center')

# Set ticks and labels
plt.xticks(range(len(correlation_matrix)), correlation_matrix.columns, rotation=45)
plt.yticks(range(len(correlation_matrix)), correlation_matrix.columns)

plt.title('Correlation Matrix of Stock Returns')
plt.tight_layout()
plt.show()

# Calculate average correlation
avg_correlation = (correlation_matrix.sum().sum() - len(correlation_matrix)) / (len(correlation_matrix) * (len(correlation_matrix) - 1))
print(f"Average correlation between stocks: {avg_correlation:.3f}")
```

```python
# Remove SPX from the data and drop the Date column
df_no_spx = df.drop(['SPX 2021_2022', 'Date'], axis=1)

# Standardize the data
scaler = StandardScaler()
returns_scaled = scaler.fit_transform(df_no_spx)

# Apply PCA
pca = PCA()
pca_result = pca.fit_transform(returns_scaled)

# Create DataFrame with principal components
pca_df = pd.DataFrame(
    pca_result,
    columns=[f'PC{i+1}' for i in range(len(df_no_spx.columns))]
)

# Plot explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         np.cumsum(pca.explained_variance_ratio_), 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Scree Plot: Cumulative Explained Variance (excluding SPX)')
plt.grid(True)
plt.show()

# Print explained variance for first 3 components
print("Explained variance ratio for first 3 components (excluding SPX):")
for i, ratio in enumerate(pca.explained_variance_ratio_[:3]):
    print(f"PC{i+1}: {ratio:.3f}")
```

```python
# Get component loadings
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(len(df_no_spx.columns))],
    index=df_no_spx.columns
)

# Plot loadings for first 3 PCs on a single plot
plt.figure(figsize=(12, 8))
for i in range(3):
    plt.plot(loadings.index, loadings[f'PC{i+1}'], 
             marker='o', 
             linewidth=2, 
             markersize=8,
             label=f'PC{i+1}')

plt.title('Component Loadings for First 3 Principal Components')
plt.xlabel('Stocks')
plt.ylabel('Loading Value')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print top contributors for each PC
print("Top contributors to each principal component:")
for i in range(3):
    print(f"\nPC{i+1}:")
    top_contributors = loadings[f'PC{i+1}'].abs().sort_values(ascending=False).head(3)
    for stock, loading in top_contributors.items():
        print(f"{stock}: {loading:.3f}")
```

```python
# Calculate rolling PCA (using 60-day windows)
window_size = 60
rolling_pca = []

for i in range(window_size, len(df_no_spx)):
    window_data = df_no_spx.iloc[i-window_size:i]
    window_scaled = StandardScaler().fit_transform(window_data)
    pca_window = PCA()
    pca_window.fit(window_scaled)
    rolling_pca.append(pca_window.explained_variance_ratio_[:3])

rolling_pca_df = pd.DataFrame(rolling_pca, columns=['PC1', 'PC2', 'PC3'])

# Plot rolling explained variance
plt.figure(figsize=(12, 6))
for col in rolling_pca_df.columns:
    plt.plot(rolling_pca_df.index, rolling_pca_df[col], label=col)
plt.xlabel('Time')
plt.ylabel('Explained Variance Ratio')
plt.title('Rolling Explained Variance Ratio (60-day window)')
plt.legend()
plt.grid(True)
plt.show()
```


---

## 18. Black-Litterman vs. Mean-Variance Portfolio Optimization in Python (Lecture 100)

`2026 Video Lectures/100. Black-Litterman vs. Mean-Variance Portfolio Optimization in Python/blmpfm.ipynb` — Black-Litterman vs mean-variance optimization.

```python
import numpy as np
import pandas as pd
import yfinance as yf  # replaces the Interactive Brokers (ibapi) data scripts


def _yf_close(ticker, start, end=None):
    """Daily adjusted close for one ticker from Yahoo Finance, as a Series indexed by date."""
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    s = raw["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = s.dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.name = ticker
    return s


# Writes efficient_frontier_daily_returns.csv (columns: date, symbol, daily_return) for 99 S&P 500 stocks
_syms = ['GOOGL', 'META', 'DIS', 'NFLX', 'CMCSA', 'T', 'VZ', 'TMUS', 'CHTR', 'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'BKNG', 'PG', 'KO', 'PEP', 'WMT', 'COST', 'MDLZ', 'MO', 'PM', 'CL', 'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'C', 'JNJ', 'UNH', 'LLY', 'ABBV', 'MRK', 'PFE', 'TMO', 'DHR', 'BMY', 'CAT', 'DE', 'UPS', 'HON', 'RTX', 'LMT', 'GE', 'MMM', 'BA', 'AAPL', 'MSFT', 'NVDA', 'AVGO', 'CRM', 'AMD', 'ORCL', 'ADBE', 'CSCO', 'LIN', 'APD', 'SHW', 'ECL', 'NEM', 'FCX', 'DD', 'DOW', 'NUE', 'PLD', 'AMT', 'EQIX', 'SPG', 'PSA', 'WELL', 'DLR', 'O', 'VICI', 'NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'EXC', 'XEL', 'ED']
_px = yf.download(_syms, start="2024-06-01", auto_adjust=True, progress=False)["Close"]
_px.index = pd.to_datetime(_px.index).tz_localize(None)
_ret = _px.pct_change().iloc[1:].dropna(how="all")
_long = _ret.stack().rename("daily_return").rename_axis(["date", "symbol"]).reset_index()
_long.to_csv("efficient_frontier_daily_returns.csv", index=False)

```

```python
%%html
<style>
/* Overwrite the hard-coded white background for ipywidgets */
.cell-output-ipywidget-background {
    background-color: transparent !important;
}
/* Set widget foreground text and color to match the VS Code dark theme */
:root {
    --jp-widgets-color: var(--vscode-editor-foreground);
    --jp-widgets-font-size: var(--vscode-editor-font-size);
}
</style>
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.optimize as sco

# ==========================================
# --- 1. DATA GENERATION & SIMULATION ---
# ==========================================
np.random.seed(42)

# Define our "Universe" of Risky Assets
assets = ['Tech', 'Health', 'Finance', 'Energy', 'Consumer', 'Bonds', 'Gold']
n_assets = len(assets)

# Theoretical Expected Annual Returns and Volatilities
expected_returns = np.array([0.14, 0.10, 0.11, 0.08, 0.09, 0.04, 0.06])
volatilities = np.array([0.22, 0.16, 0.19, 0.25, 0.14, 0.06, 0.15])

# Create a realistic correlation matrix (Bonds/Gold have low correlation to Equities)
corr_matrix = np.array([
    [1.00, 0.60, 0.65, 0.40, 0.70, -0.10, 0.05],
    [0.60, 1.00, 0.55, 0.30, 0.65, -0.05, 0.10],
    [0.65, 0.55, 1.00, 0.50, 0.60, -0.05, 0.05],
    [0.40, 0.30, 0.50, 1.00, 0.40, -0.15, 0.25],
    [0.70, 0.65, 0.60, 0.40, 1.00, 0.00, 0.10],
    [-0.10,-0.05,-0.05,-0.15, 0.00, 1.00, 0.30],
    [0.05, 0.10, 0.05, 0.25, 0.10, 0.30, 1.00]
])

# Covariance Matrix: diag(vol) * corr * diag(vol)
cov_matrix = np.outer(volatilities, volatilities) * corr_matrix

# Simulate Price Paths (Geometric Brownian Motion)
n_days = 252
dt = 1/252
daily_returns = np.random.multivariate_normal(expected_returns * dt, cov_matrix * dt, n_days)
price_paths = np.vstack([np.ones(n_assets), np.cumprod(1 + daily_returns, axis=0)])
dates_array = pd.date_range(start='2024-01-01', periods=n_days+1, freq='B')

# ==========================================
# --- 2. THE EFFICIENT FRONTIER MATH ---
# ==========================================
risk_free_rate = 0.02
n_portfolios = 3000

# Generate Random Portfolios for the Scatter Cloud
random_weights = np.random.dirichlet(np.ones(n_assets), n_portfolios)
port_returns = random_weights @ expected_returns
port_vols = np.sqrt(np.einsum('ij,jk,ik->i', random_weights, cov_matrix, random_weights))
port_sharpes = (port_returns - risk_free_rate) / port_vols

# Sort by Sharpe so higher Sharpe points are drawn last (on top)
sort_idx = np.argsort(port_sharpes)
port_returns = port_returns[sort_idx]
port_vols = port_vols[sort_idx]
port_sharpes = port_sharpes[sort_idx]
random_weights = random_weights[sort_idx]

# Find Tangency Portfolio (Max Sharpe)
max_sharpe_idx = np.argmax(port_sharpes)
opt_ret = port_returns[max_sharpe_idx]
opt_vol = port_vols[max_sharpe_idx]
opt_weights = random_weights[max_sharpe_idx]

# Calculate the True Efficient Frontier Line (Upper boundary)
target_returns = np.linspace(port_returns.min(), port_returns.max(), 50)
ef_vols = []
for tr in target_returns:
    res = sco.minimize(lambda w: np.sqrt(w.T @ cov_matrix @ w), 
                       np.ones(n_assets)/n_assets, method='SLSQP',
                       bounds=[(0, 1) for _ in range(n_assets)],
                       constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                                    {'type': 'eq', 'fun': lambda w: w.T @ expected_returns - tr}])
    ef_vols.append(res.fun)
ef_vols = np.array(ef_vols)

# ==========================================
# --- 3. OPTIMIZED ANIMATION FRAMES ---
# ==========================================
frames = []
step_size = max(1, (n_days + 1) // 100) # Roughly 100 frames for smooth playback
frame_indices = list(range(1, n_days + 1, step_size))
if frame_indices[-1] != n_days:
    frame_indices.append(n_days)

# Colors for the asset paths
path_colors = ['rgba(255,255,255,0.6)', 'rgba(0,255,255,0.6)', 'rgba(255,51,51,0.6)', 
               'rgba(255,255,0,0.6)', 'rgba(153,51,255,0.6)', 'rgba(57,255,20,0.6)', 'rgba(255,153,51,0.6)']

for k in frame_indices:
    t_x = dates_array[:k]
    frame_data = []
    
    # 1. Left Chart: 7 Asset Price Paths
    for i in range(n_assets):
        frame_data.append(go.Scatter(x=t_x, y=price_paths[:k, i], mode='lines', 
                                     line=dict(color=path_colors[i], width=1.5), name=assets[i]))
    
    # Reveal logic for the right side based on time progression
    progress = k / n_days
    num_ports_to_show = int(n_portfolios * progress)
    ef_points_to_show = int(len(target_returns) * progress)
    
    # 2. Right Chart (Top): Random Portfolios Scatter
    frame_data.append(go.Scatter(
        x=port_vols[:num_ports_to_show], y=port_returns[:num_ports_to_show], mode='markers',
        marker=dict(color=port_sharpes[:num_ports_to_show], colorscale='Viridis', 
                    size=4, opacity=0.7, showscale=False), name='Portfolios'
    ))
    
    # 3. Right Chart (Top): True Efficient Frontier Line
    frame_data.append(go.Scatter(
        x=ef_vols[:ef_points_to_show], y=target_returns[:ef_points_to_show], mode='lines',
        line=dict(color='#00ffff', width=3, dash='dash'), name='Efficient Frontier'
    ))
    
    # 4. Right Chart (Top): Tangency Point (Only reveal near the end)
    show_tangency = progress > 0.95
    frame_data.append(go.Scatter(
        x=[opt_vol] if show_tangency else [None], y=[opt_ret] if show_tangency else [None], 
        mode='markers', marker=dict(color='#39ff14', size=14, symbol='star', line=dict(color='white', width=1)), 
        name='Tangency Portfolio'
    ))
    
    # 5. Right Chart (Bottom): Tangency Weights Bar Chart (Grow with progress)
    frame_data.append(go.Bar(
        x=assets, y=opt_weights * (progress if not show_tangency else 1.0), 
        marker_color='#39ff14', name='Weights'
    ))

    frames.append(go.Frame(data=frame_data, name=f"step{k}"))

# ==========================================
# --- 4. FIGURE INITIALIZATION ---
# ==========================================
fig = make_subplots(
    rows=2, cols=2, 
    specs=[[{"rowspan": 2}, {}],
           [None, {}]],
    subplot_titles=[
        "Simulated Asset Price Paths (Growth of $1)", 
        "Markowitz Bullet: Risk vs. Expected Return",
        "Tangency Portfolio Capital Allocation"
    ],
    horizontal_spacing=0.08,
    vertical_spacing=0.15
)

# Add Dummy Traces (MUST perfectly match the 11 traces in the frame loop)
# 7 Asset lines
for i in range(n_assets):
    fig.add_trace(go.Scatter(x=[dates_array[0]], y=[price_paths[0, i]]), row=1, col=1)
# 3 EF Scatter traces (Portfolios, Frontier Line, Tangency Point)
for _ in range(3):
    fig.add_trace(go.Scatter(x=[None], y=[None]), row=1, col=2)
# 1 Bar chart for weights
fig.add_trace(go.Bar(x=assets, y=[0]*n_assets), row=2, col=2)

fig.frames = frames

# ==========================================
# --- 5. LAYOUT & METRICS ANNOTATION ---
# ==========================================
sliders = [dict(
    active=0, currentvalue={"prefix": ""}, pad={"t": 0},
    x=0.15, len=0.85, y=-0.05,
    steps=[dict(
        method="animate",
        args=[[frames[idx].name], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
        label=str(dates_array[k].date())
    ) for idx, k in enumerate(frame_indices)]
)]

fig.update_layout(
    height=700, width=1100,
    title_text="Modern Portfolio Theory: Building the True Efficient Frontier",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
    showlegend=False, sliders=sliders, 
    margin=dict(t=80, b=100, r=50, l=50),
    updatemenus=[{
        'type': 'buttons', 'x': 0.0, 'y': -0.1, 'xanchor': 'left', 'yanchor': 'top', 'direction': 'left', 'showactive': False,
        'buttons': [{'label': '▶ Play Simulation', 'method': 'animate', 'args': [None, {'frame': {'duration': 30, 'redraw': True}, 'transition': {'duration': 0}, 'fromcurrent': True, 'mode': 'immediate'}]}]
    }]
)

# Enforce static axes limits so charts don't bounce during animation
fig.update_xaxes(title_text='Date', range=[dates_array[0], dates_array[-1]], row=1, col=1, gridcolor='rgba(128,128,128,0.2)')
fig.update_yaxes(title_text='Normalized Price', range=[0.6, price_paths.max()*1.05], row=1, col=1, gridcolor='rgba(128,128,128,0.2)')

fig.update_xaxes(title_text='Volatility (Risk, σ)', range=[0, port_vols.max()*1.1], row=1, col=2, gridcolor='rgba(128,128,128,0.2)')
fig.update_yaxes(title_text='Expected Return (μ)', range=[0, port_returns.max()*1.1], row=1, col=2, gridcolor='rgba(128,128,128,0.2)')

fig.update_xaxes(title_text='Asset Class', row=2, col=2, gridcolor='rgba(128,128,128,0.2)')
fig.update_yaxes(title_text='Weight (%)', range=[0, opt_weights.max()*1.2], row=2, col=2, gridcolor='rgba(128,128,128,0.2)')

# --- ADD ANNOTATION OVERLAYS ---
max_sharpe_val = port_sharpes.max()
fig.add_annotation(
    text=f"<b>Max Achievable Sharpe: {max_sharpe_val:.2f}</b>",
    xref="x domain", yref="y domain", x=0.02, y=0.98,
    showarrow=False, align="left", font=dict(color="#39ff14", size=14),
    bgcolor="rgba(0,0,0,0.6)", bordercolor="#39ff14", borderwidth=1, borderpad=6,
    row=1, col=2
)

fig.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.optimize as sco

# ==========================================
# --- 1. DATA DEFINITION (The Two Universes) ---
# ==========================================
np.random.seed(42)

# Universe 1: Equities (High Correlation to each other)
eq_assets = ['Tech', 'Health', 'Finance', 'Consumer']
eq_ret = np.array([0.14, 0.10, 0.11, 0.09])
eq_vol = np.array([0.22, 0.16, 0.19, 0.14])

# Universe 2 Additions: Non-Equities (Low/Negative Correlation to Equities)
non_eq_assets = ['Govt Bonds', 'Corp Bonds', 'Gold']
non_eq_ret = np.array([0.04, 0.06, 0.07])
non_eq_vol = np.array([0.06, 0.09, 0.15])

# Combined "Multi-Asset" Universe
multi_assets = eq_assets + non_eq_assets
multi_ret = np.concatenate((eq_ret, non_eq_ret))
multi_vol = np.concatenate((eq_vol, non_eq_vol))
n_eq = len(eq_assets)
n_multi = len(multi_assets)

# Master Correlation Matrix (Equities highly correlated, Bonds/Gold provide diversification)
corr_matrix = np.array([
    #Tech Health Fin   Cons  Govt  Corp  Gold
    [1.00, 0.65, 0.70, 0.75, -0.15, 0.20, 0.05], # Tech
    [0.65, 1.00, 0.60, 0.70, -0.05, 0.25, 0.10], # Health
    [0.70, 0.60, 1.00, 0.65, -0.10, 0.30, 0.05], # Finance
    [0.75, 0.70, 0.65, 1.00,  0.00, 0.25, 0.15], # Consumer
    [-0.15,-0.05,-0.10, 0.00, 1.00, 0.40, 0.25], # Govt Bonds
    [0.20, 0.25, 0.30, 0.25,  0.40, 1.00, 0.15], # Corp Bonds
    [0.05, 0.10, 0.05, 0.15,  0.25, 0.15, 1.00]  # Gold
])

# Covariance Matrices
cov_multi = np.outer(multi_vol, multi_vol) * corr_matrix
cov_eq = cov_multi[:n_eq, :n_eq] # Slice just the equities portion

risk_free_rate = 0.02
n_portfolios = 2500

# ==========================================
# --- 2. MONTE CARLO & OPTIMIZATION ---
# ==========================================

def simulate_universe(n_assets, expected_returns, cov_matrix):
    weights = np.random.dirichlet(np.ones(n_assets), n_portfolios)
    returns = weights @ expected_returns
    vols = np.sqrt(np.einsum('ij,jk,ik->i', weights, cov_matrix, weights))
    sharpes = (returns - risk_free_rate) / vols
    
    # Sort by Volatility for smooth left-to-right animation drawing
    sort_idx = np.argsort(vols)
    return weights[sort_idx], returns[sort_idx], vols[sort_idx], sharpes[sort_idx]

# Simulate both universes
eq_w, eq_r, eq_v, eq_s = simulate_universe(n_eq, eq_ret, cov_eq)
multi_w, multi_r, multi_v, multi_s = simulate_universe(n_multi, multi_ret, cov_multi)

# Find Tangency Portfolios
eq_max_idx = np.argmax(eq_s)
eq_tan_w, eq_tan_r, eq_tan_v = eq_w[eq_max_idx], eq_r[eq_max_idx], eq_v[eq_max_idx]

multi_max_idx = np.argmax(multi_s)
multi_tan_w, multi_tan_r, multi_tan_v = multi_w[multi_max_idx], multi_r[multi_max_idx], multi_v[multi_max_idx]

# Calculate True Frontier Lines (Upper Boundaries)
def get_frontier(expected_returns, cov_matrix, n_assets, target_returns):
    vols = []
    for tr in target_returns:
        res = sco.minimize(lambda w: np.sqrt(w.T @ cov_matrix @ w), np.ones(n_assets)/n_assets, 
                           method='SLSQP', bounds=[(0, 1) for _ in range(n_assets)],
                           constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                                        {'type': 'eq', 'fun': lambda w: w.T @ expected_returns - tr}])
        vols.append(res.fun)
    return np.array(vols)

eq_targets = np.linspace(eq_r.min(), eq_r.max(), 40)
eq_frontier_vols = get_frontier(eq_ret, cov_eq, n_eq, eq_targets)

multi_targets = np.linspace(multi_r.min(), multi_r.max(), 40)
multi_frontier_vols = get_frontier(multi_ret, cov_multi, n_multi, multi_targets)

# ==========================================
# --- 3. ANIMATION FRAMES ---
# ==========================================
frames = []
n_frames = 100

for i in range(1, n_frames + 1):
    progress = i / n_frames
    
    # Phase timing logic
    eq_progress = min(1.0, progress * 2.5)       # Finishes at frame 40
    multi_progress = min(1.0, max(0, progress - 0.4) * 2.5) # Starts at frame 40, finishes at 80
    
    # Independent reveal logic for the tangency stars
    reveal_eq_star = eq_progress >= 1.0
    reveal_multi_star = multi_progress >= 1.0
    
    eq_pts = int(n_portfolios * eq_progress)
    eq_line_pts = int(len(eq_targets) * eq_progress)
    
    multi_pts = int(n_portfolios * multi_progress)
    multi_line_pts = int(len(multi_targets) * multi_progress)

    frame_data = [
        # 0. Equities Scatter
        go.Scatter(x=eq_v[:eq_pts], y=eq_r[:eq_pts], mode='markers', marker=dict(color='#00E5FF', size=3, opacity=0.4), name='Equity Portfolios'),
        # 1. Equities Frontier Line
        go.Scatter(x=eq_frontier_vols[:eq_line_pts], y=eq_targets[:eq_line_pts], mode='lines', line=dict(color='#00E5FF', width=4), name='Equity EF'),
        # 2. Multi-Asset Scatter
        go.Scatter(x=multi_v[:multi_pts], y=multi_r[:multi_pts], mode='markers', marker=dict(color='#39ff14', size=3, opacity=0.4), name='Multi-Asset Portfolios'),
        # 3. Multi-Asset Frontier Line
        go.Scatter(x=multi_frontier_vols[:multi_line_pts], y=multi_targets[:multi_line_pts], mode='lines', line=dict(color='#39ff14', width=4), name='Multi-Asset EF'),
        
        # 4. Equities Tangency Point (Reveals instantly when eq_progress hits 1.0)
        go.Scatter(x=[eq_tan_v] if reveal_eq_star else [None], y=[eq_tan_r] if reveal_eq_star else [None], 
                   mode='markers', marker=dict(color='#00E5FF', size=16, symbol='star', line=dict(color='white', width=1)), name='Eq Tangency'),
        # 5. Multi-Asset Tangency Point (Reveals instantly when multi_progress hits 1.0)
        go.Scatter(x=[multi_tan_v] if reveal_multi_star else [None], y=[multi_tan_r] if reveal_multi_star else [None], 
                   mode='markers', marker=dict(color='#39ff14', size=16, symbol='star', line=dict(color='white', width=1)), name='Multi Tangency'),
        
        # 6. Multi-Asset Bar Chart (Bottom Left) -> Scales up with multi_progress
        go.Bar(x=multi_assets, y=multi_tan_w * multi_progress, marker_color='#39ff14', name='Multi Weights'),
        # 7. Equities Bar Chart (Bottom Right) -> Scales up with eq_progress
        go.Bar(x=eq_assets, y=eq_tan_w * eq_progress, marker_color='#00E5FF', name='Eq Weights')
    ]
    frames.append(go.Frame(data=frame_data, name=f"step{i}"))

# ==========================================
# --- 4. FIGURE INITIALIZATION & LAYOUT ---
# ==========================================
# 2 Rows, Top Row spans both columns!
fig = make_subplots(
    rows=2, cols=2, 
    specs=[[{"colspan": 2}, None], 
           [{}, {}]],
    subplot_titles=[
        "The Expansion of the Efficient Frontier (Equities vs. Multi-Asset)", 
        "Multi-Asset Tangency Portfolio",
        "Equities Tangency Portfolio"
    ],
    vertical_spacing=0.15
)

# Add 8 Dummy Traces to match frame architecture precisely
for _ in range(6): fig.add_trace(go.Scatter(x=[None], y=[None]), row=1, col=1)
# Add Multi-Asset Bar to Row 2, Col 1 (Bottom Left)
fig.add_trace(go.Bar(x=multi_assets, y=[0]*n_multi), row=2, col=1)
# Add Equities Bar to Row 2, Col 2 (Bottom Right)
fig.add_trace(go.Bar(x=eq_assets, y=[0]*n_eq), row=2, col=2)

fig.frames = frames

# Animation Sliders & Play Button
sliders = [dict(
    active=0, currentvalue={"prefix": "Simulation Phase: "}, pad={"t": 0},
    x=0.15, len=0.85, y=-0.05,
    steps=[dict(method="animate", args=[[f.name], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))], label=str(i)) for i, f in enumerate(frames)]
)]

fig.update_layout(
    height=800, width=1100,
    title_text="Portfolio Optimization: The Mathematical Power of Diversification",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
    showlegend=False, sliders=sliders, 
    margin=dict(t=80, b=100, r=50, l=50),
    updatemenus=[{
        'type': 'buttons', 'x': 0.0, 'y': -0.1, 'xanchor': 'left', 'yanchor': 'top', 'direction': 'left', 'showactive': False,
        'buttons': [{'label': '▶ Play Simulation', 'method': 'animate', 'args': [None, {'frame': {'duration': 40, 'redraw': True}, 'transition': {'duration': 0}, 'fromcurrent': True, 'mode': 'immediate'}]}]
    }]
)

# Lock Axes to prevent bouncing
max_vol = multi_v.max() * 1.05
max_ret = multi_r.max() * 1.1

# Top Chart (Frontiers)
fig.update_xaxes(title_text='Volatility (Risk, σ)', range=[0, max_vol], row=1, col=1, gridcolor='rgba(128,128,128,0.2)')
fig.update_yaxes(title_text='Expected Return (μ)', range=[0, max_ret], row=1, col=1, gridcolor='rgba(128,128,128,0.2)')

# Bottom Left Chart (Multi-Asset Bars)
fig.update_yaxes(title_text='Weight (%)', range=[0, 1.0], row=2, col=1, gridcolor='rgba(128,128,128,0.2)')

# Bottom Right Chart (Equities Bars)
fig.update_yaxes(title_text='Weight (%)', range=[0, 1.0], row=2, col=2, gridcolor='rgba(128,128,128,0.2)')

# Sharpe Annotations
fig.add_annotation(
    text=f"<b>Equities Max Sharpe: {eq_s.max():.2f}</b>",
    xref="x domain", yref="y domain", x=0.02, y=0.98,
    showarrow=False, align="left", font=dict(color="#00E5FF", size=14),
    bgcolor="rgba(0,0,0,0.6)", bordercolor="#00E5FF", borderwidth=1, borderpad=6,
    row=1, col=1
)

fig.add_annotation(
    text=f"<b>Multi-Asset Max Sharpe: {multi_s.max():.2f}</b>",
    xref="x domain", yref="y domain", x=0.02, y=0.88,
    showarrow=False, align="left", font=dict(color="#39ff14", size=14),
    bgcolor="rgba(0,0,0,0.6)", bordercolor="#39ff14", borderwidth=1, borderpad=6,
    row=1, col=1
)

fig.show()
```

```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# --- 1. DATA GENERATION & SIMULATION ---
# ==========================================
np.random.seed(42) # Set seed for reproducible paths

total_years = 10
days_per_year = 252
total_days = total_years * days_per_year
dt = 1 / days_per_year

t_days = np.arange(total_days + 1)
t_years = t_days / days_per_year
n_paths = 10

# --- Path 1: GBM (8% Expected Return, 10% Volatility) ---
mu_1 = 0.08
sigma_1 = 0.10
Z_1 = np.random.normal(0, 1, (total_days, n_paths))
# GBM Formula applied across 10 paths
returns_1 = (mu_1 - 0.5 * sigma_1**2) * dt + sigma_1 * np.sqrt(dt) * Z_1
paths_1 = np.vstack([np.ones(n_paths), np.exp(np.cumsum(returns_1, axis=0))])
# Calculate the cross-sectional mean path
mean_path_1 = np.mean(paths_1, axis=1)

# --- Path 2: GBM (20% Expected Return, 30% Volatility) ---
mu_2 = 0.20
sigma_2 = 0.30
Z_2 = np.random.normal(0, 1, (total_days, n_paths))
# GBM Formula applied across 10 paths
returns_2 = (mu_2 - 0.5 * sigma_2**2) * dt + sigma_2 * np.sqrt(dt) * Z_2
paths_2 = np.vstack([np.ones(n_paths), np.exp(np.cumsum(returns_2, axis=0))])
# Calculate the cross-sectional mean path
mean_path_2 = np.mean(paths_2, axis=1)

# ==========================================
# --- 2. OPTIMIZED ANIMATION FRAMES ---
# ==========================================
frames = []

# Animate in roughly 100 steps (every ~25 trading days) to keep it smooth and fast
step_size = max(1, total_days // 100)
frame_indices = list(range(1, total_days + 1, step_size))
if frame_indices[-1] != total_days:
    frame_indices.append(total_days)

for k in frame_indices:
    current_years = t_years[:k+1]
    frame_data = []
    
    # 1. Left Chart: 10 Faint Paths + 1 Mean Path
    for p in range(n_paths):
        frame_data.append(go.Scatter(
            x=current_years, y=paths_1[:k+1, p], 
            mode='lines', line=dict(color='#00E5FF', width=1), opacity=0.5,
            showlegend=False, hoverinfo='skip'
        ))
    frame_data.append(go.Scatter(
        x=current_years, y=mean_path_1[:k+1], 
        mode='lines', line=dict(color='#ffffff', width=4),
        name='Mean Path 1'
    ))

    # 2. Right Chart: 10 Faint Paths + 1 Mean Path
    for p in range(n_paths):
        frame_data.append(go.Scatter(
            x=current_years, y=paths_2[:k+1, p], 
            mode='lines', line=dict(color='#39ff14', width=1), opacity=0.5,
            showlegend=False, hoverinfo='skip'
        ))
    frame_data.append(go.Scatter(
        x=current_years, y=mean_path_2[:k+1], 
        mode='lines', line=dict(color='#ffffff', width=4),
        name='Mean Path 2'
    ))

    frames.append(go.Frame(data=frame_data, name=f"day{k}"))

# ==========================================
# --- 3. FIGURE INITIALIZATION ---
# ==========================================
fig = make_subplots(
    rows=1, cols=2, 
    subplot_titles=[
        "Path 1: Expected Return 8%, Volatility 10%", 
        "Path 2: Expected Return 20%, Volatility 30%"
    ],
    horizontal_spacing=0.1
)

# Add Dummy Traces (MUST perfectly match the trace structure in the frame loop: 11 left, 11 right)
for _ in range(n_paths):
    fig.add_trace(go.Scatter(x=[0], y=[1.0]), row=1, col=1)
fig.add_trace(go.Scatter(x=[0], y=[1.0]), row=1, col=1) # Left Mean

for _ in range(n_paths):
    fig.add_trace(go.Scatter(x=[0], y=[1.0]), row=1, col=2)
fig.add_trace(go.Scatter(x=[0], y=[1.0]), row=1, col=2) # Right Mean

fig.frames = frames

# ==========================================
# --- 4. LAYOUT & AESTHETICS ---
# ==========================================
sliders = [dict(
    active=0, currentvalue={"prefix": ""}, pad={"t": 0},
    x=0.15, len=0.85, y=-0.05,
    steps=[dict(
        method="animate",
        args=[[frames[idx].name], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
        label=f"{t_years[k]:.1f}"
    ) for idx, k in enumerate(frame_indices)]
)]

fig.update_layout(
    height=700, width=1100,
    title_text="Geometric Brownian Motion: The Drag of High Volatility (10 Sample Paths + Mean)",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
    font=dict(color='white'),
    showlegend=False, sliders=sliders, 
    margin=dict(t=100, b=100, r=50, l=50),
    updatemenus=[{
        'type': 'buttons', 'x': 0.0, 'y': -0.1, 'xanchor': 'left', 'yanchor': 'top', 'direction': 'left', 'showactive': False,
        'buttons': [{'label': '▶ Play Simulation', 'method': 'animate', 'args': [None, {'frame': {'duration': 30, 'redraw': True}, 'transition': {'duration': 0}, 'fromcurrent': True, 'mode': 'immediate'}]}]
    }]
)

# Enforce static axes limits (0 to 10 on both charts)
fig.update_xaxes(title_text='Years', range=[0, total_years], gridcolor='rgba(128,128,128,0.2)', row=1, col=1)
fig.update_yaxes(title_text='Growth of $1.00', range=[0, 10], gridcolor='rgba(128,128,128,0.2)', row=1, col=1)

fig.update_xaxes(title_text='Years', range=[0, total_years], gridcolor='rgba(128,128,128,0.2)', row=1, col=2)
fig.update_yaxes(title_text='Growth of $1.00', range=[0, 10], gridcolor='rgba(128,128,128,0.2)', row=1, col=2)

# --- ADD FINAL VALUE ANNOTATIONS ---
# Calculate the realized geometric compound rate of the MEAN path
geom_1 = ((mean_path_1[-1])**(1/total_years) - 1) * 100
fig.add_annotation(
    text=f"<b>Mean Final Value: ${mean_path_1[-1]:.2f}</b><br>Realized Mean CAGR: {geom_1:.2f}%<br>(Target Return: {mu_1*100:.1f}%)",
    xref="x domain", yref="y domain", x=0.02, y=0.95,
    showarrow=False, align="left", font=dict(color="#ffffff", size=14),
    bgcolor="rgba(0,0,0,0.6)", bordercolor="#00E5FF", borderwidth=1, borderpad=6,
    row=1, col=1
)

geom_2 = ((mean_path_2[-1])**(1/total_years) - 1) * 100
fig.add_annotation(
    text=f"<b>Mean Final Value: ${mean_path_2[-1]:.2f}</b><br>Realized Mean CAGR: {geom_2:.2f}%<br>(Target Return: {mu_2*100:.1f}%)",
    xref="x domain", yref="y domain", x=0.02, y=0.95,
    showarrow=False, align="left", font=dict(color="#ffffff", size=14),
    bgcolor="rgba(0,0,0,0.6)", bordercolor="#39ff14", borderwidth=1, borderpad=6,
    row=1, col=2
)

fig.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.optimize as sco

# ==========================================
# --- 1. DATA IMPORT & PROCESSING ---
# ==========================================
np.random.seed(42)

# Load empirical data
df = pd.read_csv('efficient_frontier_daily_returns.csv')

# Ensure date is a datetime object
df['date'] = pd.to_datetime(df['date'])

# Pivot the table to get a matrix of daily returns: Rows = Dates, Columns = Symbols
df_ret = df.pivot(index='date', columns='symbol', values='daily_return').dropna()

assets = df_ret.columns.tolist()
n_assets = len(assets)

# Annualize historical expected returns and covariance matrix (assuming 252 trading days)
expected_returns = df_ret.mean().values * 252
cov_matrix = df_ret.cov().values * 252

# ==========================================
# --- 2. THE EFFICIENT FRONTIER MATH ---
# ==========================================
risk_free_rate = 0.02
n_portfolios = 6000 # Increased for a dense, beautiful cloud

# THE FIX: Using a Dirichlet alpha of 0.15 instead of 1.0. 
# This forces the random number generator to create more concentrated portfolios,
# allowing the scatter cloud to naturally expand and "hug" the true efficient frontier.
random_weights = np.random.dirichlet(np.ones(n_assets) * 0.15, n_portfolios)

port_returns = random_weights @ expected_returns
port_vols = np.sqrt(np.einsum('ij,jk,ik->i', random_weights, cov_matrix, random_weights))
port_sharpes = (port_returns - risk_free_rate) / port_vols

# Sort by Sharpe so higher Sharpe points are drawn last (on top)
sort_idx = np.argsort(port_sharpes)
port_returns = port_returns[sort_idx]
port_vols = port_vols[sort_idx]
port_sharpes = port_sharpes[sort_idx]
random_weights = random_weights[sort_idx]

# Find Tangency Portfolio (Max Sharpe)
max_sharpe_idx = np.argmax(port_sharpes)
opt_ret = port_returns[max_sharpe_idx]
opt_vol = port_vols[max_sharpe_idx]
opt_weights = random_weights[max_sharpe_idx]

# Calculate the True Efficient Frontier Line (Upper boundary)
target_returns = np.linspace(port_returns.min(), port_returns.max() * 1.05, 60)
ef_vols = []
for tr in target_returns:
    res = sco.minimize(lambda w: np.sqrt(w.T @ cov_matrix @ w), 
                       np.ones(n_assets)/n_assets, method='SLSQP',
                       bounds=[(0, 1) for _ in range(n_assets)],
                       constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                                    {'type': 'eq', 'fun': lambda w: w.T @ expected_returns - tr}])
    ef_vols.append(res.fun if res.success else np.nan)
ef_vols = np.array(ef_vols)

# ==========================================
# --- 3. OPTIMIZED ANIMATION FRAMES ---
# ==========================================
frames = []
n_frames = 100

for i in range(1, n_frames + 1):
    progress = i / n_frames
    
    num_ports_to_show = int(n_portfolios * progress)
    ef_points_to_show = int(len(target_returns) * progress)
    show_tangency = progress > 0.95
    
    frame_data = [
        # 1. Top Chart: Random Portfolios Scatter
        go.Scatter(
            x=port_vols[:num_ports_to_show], y=port_returns[:num_ports_to_show], mode='markers',
            marker=dict(color=port_sharpes[:num_ports_to_show], colorscale='Viridis', 
                        size=4, opacity=0.7, showscale=False), name='Portfolios'
        ),
        # 2. Top Chart: True Efficient Frontier Line
        go.Scatter(
            x=ef_vols[:ef_points_to_show], y=target_returns[:ef_points_to_show], mode='lines',
            line=dict(color='#00ffff', width=3, dash='dash'), name='Efficient Frontier'
        ),
        # 3. Top Chart: Tangency Point
        go.Scatter(
            x=[opt_vol] if show_tangency else [None], y=[opt_ret] if show_tangency else [None], 
            mode='markers', marker=dict(color='#39ff14', size=16, symbol='star', line=dict(color='white', width=1)), 
            name='Tangency Portfolio'
        ),
        # 4. Bottom Chart: Tangency Weights Bar Chart
        go.Bar(
            x=assets, y=opt_weights * (progress if not show_tangency else 1.0), 
            marker_color='#39ff14', name='Weights'
        )
    ]
    frames.append(go.Frame(data=frame_data, name=f"step{i}"))

# ==========================================
# --- 4. FIGURE INITIALIZATION ---
# ==========================================
fig = make_subplots(
    rows=2, cols=1, 
    subplot_titles=[
        "Markowitz Bullet: Risk vs. Expected Return",
        "Tangency Portfolio Capital Allocation"
    ],
    vertical_spacing=0.15,
    row_heights=[0.6, 0.4]
)

# Add Dummy Traces
for _ in range(3):
    fig.add_trace(go.Scatter(x=[None], y=[None]), row=1, col=1)
fig.add_trace(go.Bar(x=assets, y=[0]*n_assets), row=2, col=1)

fig.frames = frames

# ==========================================
# --- 5. LAYOUT & METRICS ANNOTATION ---
# ==========================================
# Sliders pushed down and prefix removed
sliders = [dict(
    active=0, currentvalue={"prefix": "", "font": {"color": "white"}}, pad={"t": 10},
    x=0.15, len=0.85, y=-0.15, 
    steps=[dict(
        method="animate",
        args=[[frames[idx].name], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
        label=f"{int(progress * 100)}%"
    ) for idx, progress in enumerate(np.linspace(0, 1, n_frames))]
)]

fig.update_layout(
    height=800, width=1000,
    title_text="Empirical Portfolio Theory: Efficient Frontier from Historical Returns",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
    showlegend=False, sliders=sliders, 
    margin=dict(t=80, b=150, r=50, l=50), # Expanded bottom margin (b=150) to fit controls
    updatemenus=[{
        'type': 'buttons', 'x': 0.0, 'y': -0.18, # Buttons pushed further down
        'xanchor': 'left', 'yanchor': 'top', 'direction': 'left', 'showactive': False,
        'buttons': [{'label': '▶ Play Simulation', 'method': 'animate', 'args': [None, {'frame': {'duration': 40, 'redraw': True}, 'transition': {'duration': 0}, 'fromcurrent': True, 'mode': 'immediate'}]}]
    }]
)

# Enforce static axes limits
fig.update_xaxes(title_text='Volatility (Risk, σ)', range=[0, port_vols.max()*1.1], row=1, col=1, gridcolor='rgba(128,128,128,0.2)')

ef_y_min = min(0, port_returns.min() * 1.1)
ef_y_max = max(0, target_returns.max() * 1.05)
fig.update_yaxes(title_text='Expected Return (μ)', range=[ef_y_min, ef_y_max], row=1, col=1, gridcolor='rgba(128,128,128,0.2)')

fig.update_xaxes(title_text='Asset Class', row=2, col=1, gridcolor='rgba(128,128,128,0.2)', tickangle=45)
fig.update_yaxes(title_text='Weight (%)', range=[0, opt_weights.max()*1.2], row=2, col=1, gridcolor='rgba(128,128,128,0.2)')

# --- ADD ANNOTATION OVERLAYS ---
max_sharpe_val = port_sharpes.max()
fig.add_annotation(
    text=f"<b>Max Achievable Sharpe: {max_sharpe_val:.2f}</b>",
    xref="x domain", yref="y domain", x=0.02, y=0.98,
    showarrow=False, align="left", font=dict(color="#39ff14", size=14),
    bgcolor="rgba(0,0,0,0.6)", bordercolor="#39ff14", borderwidth=1, borderpad=6,
    row=1, col=1
)

fig.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.optimize as sco

# ==========================================
# --- 1. DATA IMPORT & PROCESSING ---
# ==========================================
np.random.seed(42)

# Load empirical data
df = pd.read_csv('efficient_frontier_daily_returns.csv')
df['date'] = pd.to_datetime(df['date'])
df_ret = df.pivot(index='date', columns='symbol', values='daily_return').dropna()

assets = df_ret.columns.tolist()
n_assets = len(assets)

# Define the subset of assets for the "Horror" demonstration
# Note: Using GOOGL as per your provided CSV sample
subset_symbols = ['AVGO', 'GOOGL', 'WELL']
subset_indices = [assets.index(s) for s in subset_symbols if s in assets]

# --- Original Universe ---
expected_returns = df_ret.mean().values * 252
cov_matrix_orig = df_ret.cov().values * 252

# --- Perturbed Universe (+0.2% Daily Noise) ---
df_centered = df_ret - df_ret.mean()
noise = np.random.normal(0, 0.002, df_ret.shape) 
df_pert = df_centered + noise
cov_matrix_pert = df_pert.cov().values * 252

# ==========================================
# --- 2. THE EFFICIENT FRONTIER MATH ---
# ==========================================
risk_free_rate = 0.02
n_portfolios = 5000 

random_weights_orig = np.random.dirichlet(np.ones(n_assets) * 0.15, n_portfolios)
random_weights_pert = np.random.dirichlet(np.ones(n_assets) * 0.15, n_portfolios)

def compute_frontier_data(weights, expected_ret, cov_mat):
    p_ret = weights @ expected_ret
    p_vol = np.sqrt(np.einsum('ij,jk,ik->i', weights, cov_mat, weights))
    p_sharpe = (p_ret - risk_free_rate) / p_vol
    sort_idx = np.argsort(p_sharpe)
    return weights[sort_idx], p_ret[sort_idx], p_vol[sort_idx], p_sharpe[sort_idx]

rw_orig, pr_orig, pv_orig, ps_orig = compute_frontier_data(random_weights_orig, expected_returns, cov_matrix_orig)
rw_pert, pr_pert, pv_pert, ps_pert = compute_frontier_data(random_weights_pert, expected_returns, cov_matrix_pert)

# Find Tangency Portfolios
opt_idx_orig = np.argmax(ps_orig)
opt_w_orig = rw_orig[opt_idx_orig]
opt_v_orig, opt_r_orig = pv_orig[opt_idx_orig], pr_orig[opt_idx_orig]

opt_idx_pert = np.argmax(ps_pert)
opt_w_pert = rw_pert[opt_idx_pert]
opt_v_pert, opt_r_pert = pv_pert[opt_idx_pert], pr_pert[opt_idx_pert]

# Efficient Frontier Lines
target_returns = np.linspace(pr_orig.min(), pr_orig.max() * 1.05, 50)
def optimize_ef_line(cov_mat):
    vols = []
    for tr in target_returns:
        res = sco.minimize(lambda w: np.sqrt(w.T @ cov_mat @ w), 
                           np.ones(n_assets)/n_assets, method='SLSQP',
                           bounds=[(0, 1) for _ in range(n_assets)],
                           constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                                        {'type': 'eq', 'fun': lambda w: w.T @ expected_returns - tr}])
        vols.append(res.fun if res.success else np.nan)
    return np.array(vols)

ef_vols_orig = optimize_ef_line(cov_matrix_orig)
ef_vols_pert = optimize_ef_line(cov_matrix_pert)

# ==========================================
# --- 3. OPTIMIZED ANIMATION FRAMES ---
# ==========================================
frames = []
n_frames = 100

for i in range(1, n_frames + 1):
    progress = i / n_frames
    num_ports = int(n_portfolios * progress)
    ef_points = int(len(target_returns) * progress)
    show_tangency = progress > 0.95
    
    frame_data = [
        # --- ORIGINAL (LEFT) ---
        go.Scatter(x=pv_orig[:num_ports], y=pr_orig[:num_ports], mode='markers', marker=dict(color=ps_orig[:num_ports], colorscale='Viridis', size=4, opacity=0.7, showscale=False)),
        go.Scatter(x=ef_vols_orig[:ef_points], y=target_returns[:ef_points], mode='lines', line=dict(color='#00E5FF', width=3, dash='dash')),
        go.Scatter(x=[opt_v_orig] if show_tangency else [None], y=[opt_r_orig] if show_tangency else [None], mode='markers', marker=dict(color='#00E5FF', size=16, symbol='star', line=dict(color='white', width=1))),
        
        # --- PERTURBED (RIGHT) ---
        go.Scatter(x=pv_pert[:num_ports], y=pr_pert[:num_ports], mode='markers', marker=dict(color=ps_pert[:num_ports], colorscale='Viridis', size=4, opacity=0.7, showscale=False)),
        go.Scatter(x=ef_vols_pert[:ef_points], y=target_returns[:ef_points], mode='lines', line=dict(color='#39ff14', width=3, dash='dash')),
        go.Scatter(x=[opt_v_pert] if show_tangency else [None], y=[opt_r_pert] if show_tangency else [None], mode='markers', marker=dict(color='#39ff14', size=16, symbol='star', line=dict(color='white', width=1))),
        
        # --- BOTTOM BARS (SUBSET ONLY) ---
        go.Bar(x=subset_symbols, y=opt_w_orig[subset_indices] * (progress if not show_tangency else 1.0), marker_color='#00E5FF'),
        go.Bar(x=subset_symbols, y=opt_w_pert[subset_indices] * (progress if not show_tangency else 1.0), marker_color='#39ff14')
    ]
    frames.append(go.Frame(data=frame_data, name=f"step{i}"))

# ==========================================
# --- 4. FIGURE INITIALIZATION ---
# ==========================================
fig = make_subplots(
    rows=2, cols=2, 
    subplot_titles=[
        "Original Data: Markowitz Bullet",
        "Perturbed Data (+0.2% Noise): Markowitz Bullet",
        "Original Conviction: AVGO / GOOGL / WELL",
        "Perturbed Conviction: AVGO / GOOGL / WELL"
    ],
    vertical_spacing=0.15, horizontal_spacing=0.08, row_heights=[0.6, 0.4]
)

# Initialize traces
for _ in range(3): fig.add_trace(go.Scatter(x=[None], y=[None]), row=1, col=1) 
for _ in range(3): fig.add_trace(go.Scatter(x=[None], y=[None]), row=1, col=2)
fig.add_trace(go.Bar(x=subset_symbols, y=[0]*3), row=2, col=1)
fig.add_trace(go.Bar(x=subset_symbols, y=[0]*3), row=2, col=2)

fig.frames = frames

# ==========================================
# --- 5. LAYOUT & AESTHETICS ---
# ==========================================
sliders = [dict(
    active=0, currentvalue={"prefix": "", "font": {"color": "white"}}, pad={"t": 10},
    x=0.15, len=0.85, y=-0.15, 
    steps=[dict(method="animate", args=[[f.name], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))], label=f"{int(idx)}%") for idx, f in enumerate(frames)]
)]

fig.update_layout(
    height=800, width=1100,
    title_text="Sensitivity Horror: How Tiny Covariance Shifts Flip Portfolio Conviction",
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
    showlegend=False, sliders=sliders, margin=dict(t=80, b=150, r=50, l=50),
    updatemenus=[{
        'type': 'buttons', 'x': 0.0, 'y': -0.18, 'xanchor': 'left', 'yanchor': 'top', 'direction': 'left', 'showactive': False,
        'buttons': [{'label': '▶ Play Simulation', 'method': 'animate', 'args': [None, {'frame': {'duration': 40, 'redraw': True}, 'transition': {'duration': 0}, 'fromcurrent': True, 'mode': 'immediate'}]}]
    }]
)

# Axis Limits
global_max_vol = max(pv_orig.max(), pv_pert.max()) * 1.1
global_max_ret = target_returns.max() * 1.05
global_max_weight = max(opt_w_orig[subset_indices].max(), opt_w_pert[subset_indices].max()) * 1.5

fig.update_xaxes(title_text='Volatility', range=[0, global_max_vol], row=1, col=1, gridcolor='rgba(128,128,128,0.2)')
fig.update_yaxes(title_text='Expected Return', range=[0, global_max_ret], row=1, col=1, gridcolor='rgba(128,128,128,0.2)')
fig.update_xaxes(title_text='Volatility', range=[0, global_max_vol], row=1, col=2, gridcolor='rgba(128,128,128,0.2)')
fig.update_yaxes(title_text='Expected Return', range=[0, global_max_ret], row=1, col=2, gridcolor='rgba(128,128,128,0.2)')

# Bar Charts
fig.update_yaxes(title_text='Weight (%)', range=[0, global_max_weight], row=2, col=1, gridcolor='rgba(128,128,128,0.2)')
fig.update_yaxes(title_text='Weight (%)', range=[0, global_max_weight], row=2, col=2, gridcolor='rgba(128,128,128,0.2)')

# Annotations
fig.add_annotation(text=f"<b>Orig Sharpe: {ps_orig.max():.2f}</b>", xref="x domain", yref="y domain", x=0.02, y=0.98, showarrow=False, font=dict(color="#00E5FF"), bgcolor="rgba(0,0,0,0.6)", row=1, col=1)
fig.add_annotation(text=f"<b>Pert Sharpe: {ps_pert.max():.2f}</b>", xref="x domain", yref="y domain", x=0.02, y=0.98, showarrow=False, font=dict(color="#39ff14"), bgcolor="rgba(0,0,0,0.6)", row=1, col=2)

fig.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.optimize as sco
import statsmodels.api as sm

# ==========================================
# --- 1. DATA SLICE: THE LATEST YEAR ---
# ==========================================
np.random.seed(42)

df = pd.read_csv('efficient_frontier_daily_returns.csv')
df['date'] = pd.to_datetime(df['date'])
df_ret = df.pivot(index='date', columns='symbol', values='daily_return').dropna()

# Slice the latest 252 trading days (1 Year)
df_latest_year = df_ret.tail(252)

# 6-Month Split (126 days each)
split_6mo = 126
df_is = df_latest_year.iloc[:split_6mo]   # In-Sample
df_oos = df_latest_year.iloc[split_6mo:]  # Out-of-Sample

assets = df_ret.columns.tolist()
n_assets = len(assets)
risk_free_rate = 0.02

# ==========================================
# --- 2. THE OPTIMIZATION (6-MONTH OVERFIT) ---
# ==========================================
mu_is = df_is.mean().values * 252
cov_is = df_is.cov().values * 252

def get_neg_sharpe(weights, mu, cov):
    p_ret = weights @ mu
    p_vol = np.sqrt(weights.T @ cov @ weights)
    return -(p_ret - risk_free_rate) / p_vol

res = sco.minimize(
    get_neg_sharpe,
    np.ones(n_assets) / n_assets,
    args=(mu_is, cov_is),
    method='SLSQP',
    bounds=[(0, 1) for _ in range(n_assets)],
    constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
)

w_opt = res.x
is_vol = np.sqrt(w_opt.T @ cov_is @ w_opt)
is_ret = w_opt @ mu_is
is_sharpe = -res.fun

# --- Out-of-Sample Performance ---
oos_opt_daily = df_oos @ w_opt
oos_mkt_daily = df_oos.mean(axis=1)
oos_opt_cum = (1 + oos_opt_daily).cumprod()
oos_mkt_cum = (1 + oos_mkt_daily).cumprod()

# --- Out-of-Sample Stats ---
oos_opt_sharpe = (oos_opt_daily.mean() * 252 - risk_free_rate) / (oos_opt_daily.std() * np.sqrt(252))
oos_mkt_sharpe = (oos_mkt_daily.mean() * 252 - risk_free_rate) / (oos_mkt_daily.std() * np.sqrt(252))

X = sm.add_constant(oos_mkt_daily)
model = sm.OLS(oos_opt_daily, X).fit()
beta_val = model.params.iloc[1]
alpha_p_value = model.pvalues.iloc[0]

# --- Frontier Line ---
target_rets = np.linspace(df_is.mean().min() * 252, df_is.mean().max() * 252, 50)
ef_vols_is = [
    sco.minimize(
        lambda w: np.sqrt(w.T @ cov_is @ w),
        np.ones(n_assets) / n_assets,
        method='SLSQP',
        bounds=[(0, 1) for _ in range(n_assets)],
        constraints=[
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: w.T @ mu_is - tr}
        ]
    ).fun
    for tr in target_rets
]

# ==========================================
# --- 3. ANIMATION FRAMES ---
# ==========================================
frames = []
n_frames = 100
step_size = max(1, len(df_oos) // n_frames)
oos_indices = list(range(1, len(df_oos) + 1, step_size))

for idx in oos_indices:
    frame_data = [
        # Frame Data must match trace indices 0, 1, 2, 3 in the main fig
        go.Scatter(x=ef_vols_is, y=target_rets),                 # Trace 0
        go.Scatter(x=[is_vol], y=[is_ret]),                      # Trace 1
        go.Scatter(x=df_oos.index[:idx], y=oos_opt_cum[:idx]),  # Trace 2
        go.Scatter(x=df_oos.index[:idx], y=oos_mkt_cum[:idx])   # Trace 3
    ]
    frames.append(go.Frame(data=frame_data, name=f"step{idx}"))

# ==========================================
# --- 4. FIGURE & LAYOUT ---
# ==========================================
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=[
        "6-Month In-Sample Optimization: The Star",
        "6-Month Out-of-Sample: Optimized (Green) vs. Market (Cyan)"
    ],
    vertical_spacing=0.15
)

# Initialize 4 Explicit Traces
# Trace 0: IS Frontier Line
fig.add_trace(
    go.Scatter(
        x=ef_vols_is,
        y=target_rets,
        mode='lines',
        line=dict(color='rgba(255,255,255,0.2)', width=2, dash='dash'),
        showlegend=False
    ),
    row=1, col=1
)

# Trace 1: IS Tangency Star
fig.add_trace(
    go.Scatter(
        x=[is_vol],
        y=[is_ret],
        mode='markers',
        marker=dict(color='#39ff14', size=16, symbol='star'),
        showlegend=False
    ),
    row=1, col=1
)

# Trace 2: OOS Optimized Path
fig.add_trace(
    go.Scatter(
        x=[df_oos.index[0]],
        y=[1.0],
        mode='lines',
        line=dict(color='#39ff14', width=4),
        name='Optimized Portfolio'
    ),
    row=2, col=1
)

# Trace 3: OOS Market Path
fig.add_trace(
    go.Scatter(
        x=[df_oos.index[0]],
        y=[1.0],
        mode='lines',
        line=dict(color='#00E5FF', width=3, dash='dot'),
        name='Market Portfolio'
    ),
    row=2, col=1
)

fig.frames = frames

# Play Button and Slider Styling
fig.update_layout(
    height=700, width=1100,
    title_text="The Overfitting Trap: 6-Mo Optimization vs. 6-Mo Forward Reality",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=False,
    margin=dict(t=100, b=150, r=50, l=50),
    updatemenus=[{
        'type': 'buttons',
        'x': -0.05,
        'y': -0.18,
        'xanchor': 'left',
        'yanchor': 'top',
        'direction': 'left',
        'showactive': False,
        'buttons': [{
            'label': '▶ Play Simulation',
            'method': 'animate',
            'args': [None, {
                'frame': {'duration': 40, 'redraw': True},
                'transition': {'duration': 0},
                'fromcurrent': True,
                'mode': 'immediate'
            }]
        }],
        'font': {'color': 'white', 'size': 14},
        'bgcolor': 'rgba(0,0,0,0)',
        'bordercolor': 'white',
        'borderwidth': 1
    }],
    sliders=[dict(
        active=0,
        currentvalue={"prefix": ""},
        pad={"t": 20},
        x=0.15,
        len=0.85,
        y=-0.15,
        steps=[
            dict(
                method="animate",
                args=[[f.name], dict(mode="immediate", frame=dict(duration=0, redraw=True))],
                label=f"{int(i / len(frames) * 100)}%"
            )
            for i, f in enumerate(frames)
        ]
    )],
    # Add Static Breakeven Line
    shapes=[dict(
        type="line",
        xref="x2",
        yref="y2",
        x0='2025-11-01',
        x1='2026-04-30',
        y0=1.0,
        y1=1.0,
        line=dict(color="white", width=1, dash="dash"),
        opacity=0.3
    )]
)

# Fixed Axis Configuration
fig.update_xaxes(
    title_text="Volatility",
    range=[0, is_vol * 2.5],
    row=1, col=1,
    gridcolor='rgba(128,128,128,0.2)'
)
fig.update_yaxes(
    title_text="Return",
    row=1, col=1,
    gridcolor='rgba(128,128,128,0.2)'
)

fig.update_xaxes(
    title_text="Date (Out-of-Sample)",
    range=['2025-09-01', '2026-04-30'],
    row=2, col=1,
    gridcolor='rgba(128,128,128,0.2)'
)
fig.update_yaxes(
    title_text="Value ($1 Start)",
    range=[0.9, 1.3],
    row=2, col=1,
    gridcolor='rgba(128,128,128,0.2)'
)

# Annotations
fig.add_annotation(
    text=f"<b>IS Overfitted Sharpe: {is_sharpe:.2f}</b>",
    xref="x domain",
    yref="y domain",
    x=0.02,
    y=0.95,
    showarrow=False,
    font=dict(color="#39ff14", size=14),
    row=1, col=1
)

fig.add_annotation(
    text=(
        f"<b>OOS Optimized Sharpe: {oos_opt_sharpe:.2f}</b><br>"
        f"<b>OOS Market Sharpe: {oos_mkt_sharpe:.2f}</b><br><br>"
        f"Portfolio Beta: {beta_val:.2f}<br>"
        f"Alpha p-value: {alpha_p_value:.4f}"
    ),
    xref="x domain",
    yref="y domain",
    x=0.98,
    y=0.95,
    showarrow=False,
    align="right",
    font=dict(color="#ffffff", size=13),
    bgcolor="rgba(30,30,30,0.8)",
    bordercolor="white",
    borderwidth=1,
    borderpad=8,
    row=2, col=1
)

fig.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# --- 1. DATA IMPORT & REGIME SPLITS ---
# ==========================================
df = pd.read_csv('efficient_frontier_daily_returns.csv')
df['date'] = pd.to_datetime(df['date'])
df_ret = df.pivot(index='date', columns='symbol', values='daily_return').dropna()

# Use the Equal-Weight Market Portfolio
market_daily_ret = df_ret.mean(axis=1)
n_days = len(market_daily_ret)

# Divide into 3 disjoint "Regimes" (approx 4 months each)
split_points = [0, n_days // 3, (2 * n_days) // 3, n_days]

# Calculate the mean for each disjoint period (Annualized)
regime_means = []
for i in range(3):
    start, end = split_points[i], split_points[i + 1]
    regime_mu = market_daily_ret.iloc[start:end].mean() * 252
    regime_means.append((df_ret.index[start], df_ret.index[end - 1], regime_mu))

# --- LLN Convergence Math ---
running_mean = (market_daily_ret.expanding().mean() * 252).values
market_cum_growth = (1 + market_daily_ret).cumprod().values

# ==========================================
# --- 2. ANIMATION FRAMES ---
# ==========================================
frames = []
n_frames = 100
indices = np.linspace(1, n_days, n_frames, dtype=int)

for k in indices:
    current_dates = market_daily_ret.index[:k]

    # Logic to show the "Local" regime mean only when we are in that period
    curr_regime_line = np.full(k, np.nan)
    for start_dt, end_dt, mu in regime_means:
        mask = (current_dates >= start_dt) & (current_dates <= end_dt)
        curr_regime_line[mask] = mu

    frame_data = [
        go.Scatter(x=current_dates, y=curr_regime_line),   # Trace 0
        go.Scatter(x=current_dates, y=running_mean[:k]),   # Trace 1
        go.Scatter(x=current_dates, y=market_cum_growth[:k])  # Trace 2
    ]
    frames.append(go.Frame(data=frame_data, name=f"step{k}"))

# ==========================================
# --- 3. FIGURE & LAYOUT ---
# ==========================================
fig = make_subplots(
    rows=2,
    cols=1,
    subplot_titles=[
        "LLN Convergence: Disjoint Regime Means vs. Running Average",
        "Market Portfolio: Realized Cumulative Growth"
    ],
    vertical_spacing=0.2,
    row_heights=[0.48, 0.52]
)

# Trace 0: Disjoint Local Mean
fig.add_trace(
    go.Scatter(
        x=[market_daily_ret.index[0]],
        y=[regime_means[0][2]],
        mode='lines',
        line=dict(color='rgba(0,255,255,0.4)', width=4, dash='dot'),
        name='Local Regime Mean'
    ),
    row=1, col=1
)

# Trace 1: Running Average
fig.add_trace(
    go.Scatter(
        x=[market_daily_ret.index[0]],
        y=[running_mean[0]],
        mode='lines',
        line=dict(color='#ffffff', width=3),
        name='Running Cumulative Mean'
    ),
    row=1, col=1
)

# Trace 2: Realized Equity Curve
fig.add_trace(
    go.Scatter(
        x=[market_daily_ret.index[0]],
        y=[1.0],
        mode='lines',
        line=dict(color='#39ff14', width=4),
        name='Cumulative Growth'
    ),
    row=2, col=1
)

fig.frames = frames

# Play Button Styling
play_button = {
    'type': 'buttons',
    'x': 0.0,
    'y': -0.21,
    'xanchor': 'left',
    'yanchor': 'top',
    'direction': 'left',
    'showactive': False,
    'bgcolor': 'rgba(0,0,0,0)',
    'bordercolor': 'white',
    'borderwidth': 1,
    'font': {'color': 'white'},
    'buttons': [
        {
            'label': '▶ Play Simulation',
            'method': 'animate',
            'args': [
                None,
                {
                    'frame': {'duration': 30, 'redraw': True},
                    'fromcurrent': True
                }
            ]
        }
    ]
}

# Fix: Reference frame index and label correctly
slider_steps = []
for i, k in enumerate(indices):
    step = {
        "args": [[f"step{k}"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
        "label": str(market_daily_ret.index[k - 1].date()),
        "method": "animate"
    }
    slider_steps.append(step)

fig.update_layout(
    height=700,
    width=1100,
    title_text="Regime Shifts & The Law of Large Numbers (Market Portfolio)",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.48,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(0,0,0,0.5)"
    ),
    margin=dict(t=100, b=160, r=50, l=50),
    updatemenus=[play_button],
    sliders=[
        dict(
            active=0,
            currentvalue={"prefix": ""},
            x=0.15,
            len=0.85,
            y=-0.18,
            steps=slider_steps
        )
    ]
)

# Axis Configuration with "modern" grid like in snippet: thin, translucent, mild gray, always shown
fig.update_xaxes(
    title_text="Date",
    row=1,
    col=1,
    gridcolor='rgba(128,128,128,0.2)',
    showgrid=True,
    griddash='solid',
    gridwidth=1,
    zeroline=False
)
fig.update_yaxes(
    title_text="Annualized Return",
    range=[running_mean.min() - 0.1, running_mean.max() + 0.1],
    row=1,
    col=1,
    gridcolor='rgba(128,128,128,0.2)',
    showgrid=True,
    griddash='solid',
    gridwidth=1,
    zeroline=False
)

fig.update_xaxes(
    title_text="Date",
    row=2,
    col=1,
    gridcolor='rgba(128,128,128,0.2)',
    showgrid=True,
    griddash='solid',
    gridwidth=1,
    zeroline=False
)
fig.update_yaxes(
    title_text="Value ($1 Start)",
    range=[0.9, 1.7],
    row=2,
    col=1,
    gridcolor='rgba(128,128,128,0.2)',
    showgrid=True,
    griddash='solid',
    gridwidth=1,
    zeroline=False
)

# Static Breakeven Line
fig.add_shape(
    type="line",
    xref="x2",
    yref="y2",
    x0=market_daily_ret.index[0],
    x1=market_daily_ret.index[-1],
    y0=1.0,
    y1=1.0,
    line=dict(color="white", width=1, dash="dash"),
    opacity=0.3
)

fig.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.optimize as sco

# ==========================================
# --- 1. DATA IMPORT & REVERSE OPTIMIZATION ---
# ==========================================
np.random.seed(42)

df = pd.read_csv('efficient_frontier_daily_returns.csv')
df['date'] = pd.to_datetime(df['date'])
df_ret = df.pivot(index='date', columns='symbol', values='daily_return').dropna()

assets = df_ret.columns.tolist()
n_assets = len(assets)

# --- BLACK-LITTERMAN EQUILIBRIUM MATH ---
risk_free_rate = 0.02
delta = 2.5  # Risk aversion coefficient
cov_matrix = df_ret.cov().values * 252

# Anchor: Assume Equal-Weight is the Market Equilibrium
w_mkt = np.ones(n_assets) / n_assets

# Calculate Implied Equilibrium Returns (Pi)
# Pi = delta * Sigma * w_mkt
pi = delta * cov_matrix @ w_mkt

# ==========================================
# --- 2. THE EFFICIENT FRONTIER MATH ---
# ==========================================
n_portfolios = 6000 
random_weights = np.random.dirichlet(np.ones(n_assets) * 0.15, n_portfolios)

# Use Equilibrium Returns (Pi)
port_returns = random_weights @ pi
port_vols = np.sqrt(np.einsum('ij,jk,ik->i', random_weights, cov_matrix, random_weights))
port_sharpes = (port_returns - risk_free_rate) / port_vols

sort_idx = np.argsort(port_sharpes)
port_returns, port_vols, port_sharpes = port_returns[sort_idx], port_vols[sort_idx], port_sharpes[sort_idx]

# Tangency Point (Market Equilibrium)
max_sharpe_idx = np.argmax(port_sharpes)
opt_ret, opt_vol = port_returns[max_sharpe_idx], port_vols[max_sharpe_idx]

# EF Line Calculation
target_returns = np.linspace(port_returns.min(), port_returns.max() * 1.05, 60)
ef_vols = []
for tr in target_returns:
    res = sco.minimize(lambda w: np.sqrt(w.T @ cov_matrix @ w), 
                       np.ones(n_assets)/n_assets, method='SLSQP',
                       bounds=[(0, 1) for _ in range(n_assets)],
                       constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                                    {'type': 'eq', 'fun': lambda w: w.T @ pi - tr}])
    ef_vols.append(res.fun if res.success else np.nan)
ef_vols = np.array(ef_vols)

# ==========================================
# --- 3. ANIMATION FRAMES ---
# ==========================================
frames = []
n_frames = 100
for i in range(1, n_frames + 1):
    progress = i / n_frames
    num_ports = int(n_portfolios * progress)
    ef_points = int(len(target_returns) * progress)
    show_tangency = progress > 0.95
    
    frame_data = [
        # 1. Equilibrium Cloud
        go.Scatter(x=port_vols[:num_ports], y=port_returns[:num_ports], mode='markers',
                   marker=dict(color=port_sharpes[:num_ports], colorscale='Viridis', size=4, opacity=0.7, showscale=False)),
        # 2. Equilibrium EF Line
        go.Scatter(x=ef_vols[:ef_points], y=target_returns[:ef_points], mode='lines',
                   line=dict(color='#00ffff', width=3, dash='dash')),
        # 3. Tangency Star
        go.Scatter(x=[opt_vol] if show_tangency else [None], y=[opt_ret] if show_tangency else [None], 
                   mode='markers', marker=dict(color='#39ff14', size=16, symbol='star', line=dict(color='white', width=1)))
    ]
    frames.append(go.Frame(data=frame_data, name=f"step{i}"))

# ==========================================
# --- 4. FIGURE INITIALIZATION ---
# ==========================================

# Play Button Styling (to match previous code cell, see @file_context_0)
play_button = {
    'type': 'buttons',
    'x': 0.0,
    'y': -0.21,
    'xanchor': 'left',
    'yanchor': 'top',
    'direction': 'left',
    'showactive': False,
    'bgcolor': 'rgba(0,0,0,0)',
    'bordercolor': 'white',
    'borderwidth': 1,
    'font': {'color': 'white'},
    'buttons': [
        {
            'label': '▶ Play Simulation',
            'method': 'animate',
            'args': [
                None,
                {
                    'frame': {'duration': 40, 'redraw': True},
                    'fromcurrent': True
                }
            ]
        }
    ]
}

slider_steps = [
    dict(
        method="animate",
        args=[[f"step{i+1}"], dict(mode="immediate", frame=dict(duration=0, redraw=True))],
        label=f"{i}%"
    ) for i in range(n_frames)
]

fig = go.Figure(
    data=[
        go.Scatter(x=[None], y=[None], mode='markers'),
        go.Scatter(x=[None], y=[None], mode='lines'),
        go.Scatter(x=[None], y=[None], mode='markers')
    ]
)

fig.update_layout(
    title="Black-Litterman Stage 1: The Implied Equilibrium Frontier",
    xaxis=dict(title="Volatility (Risk, σ)", range=[0, port_vols.max()*1.1], gridcolor='rgba(128,128,128,0.2)'),
    yaxis=dict(title="Implied Return (Π)", range=[0, target_returns.max()*1.05], gridcolor='rgba(128,128,128,0.2)'),
    plot_bgcolor='rgba(0,0,0,0)', 
    paper_bgcolor='rgba(0,0,0,0)', 
    font=dict(color='white'),
    height=700,
    showlegend=False,
    margin=dict(t=80, b=150, r=50, l=50),
    updatemenus=[play_button],
    sliders=[dict(
        active=0,
        currentvalue={"prefix": ""},
        x=0.15,
        len=0.85,
        y=-0.18,
        steps=slider_steps
    )]
)

fig.frames = frames

# --- ANNOTATION OVERLAY ---
fig.add_annotation(
    text=f"<b>Equilibrium Sharpe: {port_sharpes.max():.2f}</b>",
    xref="paper", yref="paper", x=0.02, y=0.98,
    showarrow=False, align="left", font=dict(color="#39ff14", size=14),
    bgcolor="rgba(0,0,0,0.6)", bordercolor="#39ff14", borderwidth=1, borderpad=6
)

fig.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.optimize as sco

# ==========================================
# --- 1. DATA PREP & SHARED PERTURBATION ---
# ==========================================
np.random.seed(42)
df = pd.read_csv('efficient_frontier_daily_returns.csv')
df['date'] = pd.to_datetime(df['date'])
df_ret = df.pivot(index='date', columns='symbol', values='daily_return').dropna()

assets = df_ret.columns.tolist()
n_assets = len(assets)
subset_symbols = ['AVGO', 'GOOGL', 'WELL']
subset_indices = [assets.index(s) for s in subset_symbols if s in assets]

# --- THE SHARED PERTURBATION (0.5% Daily Noise) ---
noise_level = 0.005
shared_noise = np.random.normal(0, noise_level, df_ret.shape) 
df_noisy = df_ret + shared_noise

# ==========================================
# --- 2. MODEL CALCULATIONS ---
# ==========================================
risk_free_rate = 0.02
delta = 2.5
w_mkt = np.ones(n_assets) / n_assets

def solve_weights(mu, cov):
    res = sco.minimize(lambda w: -(w @ mu - risk_free_rate) / np.sqrt(w.T @ cov @ w),
                       np.ones(n_assets)/n_assets, method='SLSQP',
                       bounds=[(0, 1) for _ in range(n_assets)],
                       constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}])
    return res.x if res.success else np.zeros(n_assets)

# --- MVO (Historical) ---
mu_mvo_o, cov_mvo_o = df_ret.mean().values * 252, df_ret.cov().values * 252
mu_mvo_n, cov_mvo_n = df_noisy.mean().values * 252, df_noisy.cov().values * 252
w_mvo_o = solve_weights(mu_mvo_o, cov_mvo_o)
w_mvo_n = solve_weights(mu_mvo_n, cov_mvo_n)

# --- BL (Equilibrium Anchor) ---
pi_o = delta * cov_mvo_o @ w_mkt
pi_n = delta * cov_mvo_n @ w_mkt
w_bl_o = solve_weights(pi_o, cov_mvo_o)
w_bl_n = solve_weights(pi_n, cov_mvo_n)

# ==========================================
# --- 3. PLOTTING WITH SHARED NOISE TITLE ---
# ==========================================
fig = make_subplots(
    rows=2, cols=2, vertical_spacing=0.15,
    subplot_titles=[
        "MVO: Original Weights", f"MVO: Shared Noise (+{noise_level*100}%)",
        "BL: Original Weights", f"BL: Shared Noise (+{noise_level*100}%)"
    ]
)

# Colors
c_orig, c_mvo_n, c_bl = '#00E5FF', '#FF3D00', '#39ff14'

fig.add_trace(go.Bar(x=subset_symbols, y=w_mvo_o[subset_indices], marker_color=c_orig), row=1, col=1)
fig.add_trace(go.Bar(x=subset_symbols, y=w_mvo_n[subset_indices], marker_color=c_mvo_n), row=1, col=2)
fig.add_trace(go.Bar(x=subset_symbols, y=w_bl_o[subset_indices], marker_color=c_bl), row=2, col=1)
fig.add_trace(go.Bar(x=subset_symbols, y=w_bl_n[subset_indices], marker_color=c_bl), row=2, col=2)

# Delta Annotations
mvo_delta = np.sum(np.abs(w_mvo_n[subset_indices] - w_mvo_o[subset_indices])) * 100
bl_delta = np.sum(np.abs(w_bl_n[subset_indices] - w_bl_o[subset_indices])) * 100

fig.add_annotation(text=f"<b>Total Subset Shift: {mvo_delta:.1f}%</b>", 
                   xref="x2 domain", yref="y2 domain", x=0.95, y=0.9, showarrow=False, 
                   font=dict(color=c_mvo_n, size=14), bgcolor="rgba(0,0,0,0.7)", row=1, col=2)

fig.add_annotation(text=f"<b>Total Subset Shift: {bl_delta:.1f}%</b>", 
                   xref="x4 domain", yref="y4 domain", x=0.95, y=0.9, showarrow=False, 
                   font=dict(color=c_bl, size=14), bgcolor="rgba(0,0,0,0.7)", row=2, col=2)

fig.update_layout(
    height=700, width=1100, 
    title_text=f"Shared Noise Impact ({noise_level*100}% Perturbation): MVO Overfitting vs. Black-Litterman Stability",
    template="plotly_dark", showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
)

fig.update_yaxes(title_text="Weight (%)", range=[0, 0.7], gridcolor='rgba(255,255,255,0.1)')
fig.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.optimize as sco

# ==========================================
# --- 1. DATA PREP ---
# ==========================================
np.random.seed(42)
df = pd.read_csv('efficient_frontier_daily_returns.csv')
df['date'] = pd.to_datetime(df['date'])
df_ret = df.pivot(index='date', columns='symbol', values='daily_return').dropna()

assets = df_ret.columns.tolist()
n_assets = len(assets)
subset_symbols = ['AVGO', 'GOOGL', 'WELL']
subset_indices = [assets.index(s) for s in subset_symbols if s in assets]

cov_matrix = df_ret.cov().values * 252

# ==========================================
# --- 2. BLACK-LITTERMAN MATH (THE TILT) ---
# ==========================================
risk_free_rate = 0.02
delta = 2.5
tau = 0.05 # Scaling factor for the uncertainty of the prior
w_mkt = np.ones(n_assets) / n_assets

# --- STAGE 1: The Prior (Equilibrium Returns) ---
pi = delta * cov_matrix @ w_mkt

# --- STAGE 2: The Views (Subjective Expertise) ---
# We express absolute views: AVGO will return 25%, GOOGL will return 20%
K = 2 # Number of views
P = np.zeros((K, n_assets)) # Pick Matrix
P[0, assets.index('AVGO')] = 1.0
P[1, assets.index('GOOGL')] = 1.0

Q = np.array([0.25, 0.20]) # View Expected Returns

# Omega (Uncertainty Matrix). Using the standard proportional formulation.
omega = np.diag(np.diag(P @ (tau * cov_matrix) @ P.T))

# --- STAGE 3: The Posterior (Combined Expected Returns) ---
# The BL Master Formula: Combining the Prior (Pi) with our Views (Q)
tau_cov_inv = np.linalg.inv(tau * cov_matrix)
omega_inv = np.linalg.inv(omega)

M1 = np.linalg.inv(tau_cov_inv + P.T @ omega_inv @ P)
M2 = tau_cov_inv @ pi + P.T @ omega_inv @ Q
mu_bl = M1 @ M2

# ==========================================
# --- 3. OPTIMIZATION ---
# ==========================================
def solve_weights(mu, cov):
    res = sco.minimize(lambda w: -(w @ mu - risk_free_rate) / np.sqrt(w.T @ cov @ w),
                       np.ones(n_assets)/n_assets, method='SLSQP',
                       bounds=[(0, 1) for _ in range(n_assets)],
                       constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}])
    return res.x if res.success else np.zeros(n_assets)

# Optimize for both states
w_prior = solve_weights(pi, cov_matrix)
w_posterior = solve_weights(mu_bl, cov_matrix)

# ==========================================
# --- 4. PLOTTING ---
# ==========================================
fig = make_subplots(
    rows=1, cols=2, 
    subplot_titles=[
        "BL Stage 1: Equilibrium Prior (Neutral)", 
        "BL Stage 2: Posterior (Hard Tech Tilt)"
    ]
)

c_prior = '#39ff14' # Neon Green
c_post = '#00E5FF'  # Cyan

# Bar Charts
fig.add_trace(go.Bar(x=subset_symbols, y=w_prior[subset_indices], marker_color=c_prior), row=1, col=1)
fig.add_trace(go.Bar(x=subset_symbols, y=w_posterior[subset_indices], marker_color=c_post), row=1, col=2)

# Calculate Deltas to show the surgical shift
avgo_shift = (w_posterior[assets.index('AVGO')] - w_prior[assets.index('AVGO')]) * 100
googl_shift = (w_posterior[assets.index('GOOGL')] - w_prior[assets.index('GOOGL')]) * 100
well_shift = (w_posterior[assets.index('WELL')] - w_prior[assets.index('WELL')]) * 100

fig.add_annotation(
    text=f"<b>AVGO Shift: +{avgo_shift:.1f}%</b><br><b>GOOGL Shift: +{googl_shift:.1f}%</b><br><span style='color:#AAAAAA'>WELL Shift: {well_shift:.1f}%</span>", 
    xref="x2 domain", yref="y2 domain", x=0.95, y=0.95, showarrow=False, align="right",
    font=dict(color=c_post, size=14), bgcolor="rgba(0,0,0,0.7)", bordercolor=c_post, borderwidth=1, borderpad=6, row=1, col=2
)

fig.update_layout(
    height=600, width=1100, 
    title_text="Subjective Expertise: Filtering Views Through the Black-Litterman Model",
    template="plotly_dark", showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
)

max_weight = max(w_prior.max(), w_posterior.max()) * 1.2
fig.update_yaxes(title_text="Weight (%)", range=[0, max_weight], gridcolor='rgba(255,255,255,0.1)')

fig.show()
```


---

## 19. How a Quant Manages a Portfolio (Lecture 88)

`2026 Video Lectures/88. How a Quant Manages a Portfolio/ff.ipynb` — Factor-model portfolio management.

```python
import numpy as np
import pandas as pd
import yfinance as yf  # replaces the Interactive Brokers (ibapi) data scripts


def _yf_close(ticker, start, end=None):
    """Daily adjusted close for one ticker from Yahoo Finance, as a Series indexed by date."""
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    s = raw["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = s.dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.name = ticker
    return s


# Writes one <TICKER>.csv per symbol (columns: date, close) from 2023 onward
for _t in ["AAPL", "MSFT", "AVGO", "UNH", "JNJ", "AMGN", "WMT", "COST", "PG", "SPY"]:
    _yf_close(_t, "2023-01-01").rename("close").rename_axis("date").reset_index().to_csv(f"{_t}.csv", index=False)

```

```python
%%html
<style>
/* Overwrite the hard-coded white background for ipywidgets */
.cell-output-ipywidget-background {
    background-color: transparent !important;
}
/* Set widget foreground text and color to match the VS Code dark theme */
:root {
    --jp-widgets-color: var(--vscode-editor-foreground);
    --jp-widgets-font-size: var(--vscode-editor-font-size);
}
</style>
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. Configuration ---
sector_map = {
    'Tech': ['AAPL', 'MSFT', 'AVGO'],
    'Healthcare': ['UNH', 'JNJ', 'AMGN'],
    'Consumer Staples': ['WMT', 'COST', 'PG']
}

# Colors
sector_colors = {
    'Tech': '#1f77b4',       # Blue
    'Healthcare': '#2ca02c', # Green
    'Consumer Staples': '#d62728' # Red
}

# Line styles (cycling)
line_styles = ['solid', 'dash', 'dot']

all_tickers = [t for tickers in sector_map.values() for t in tickers]

# --- 2. Data Loading ---
df_all = pd.DataFrame()

print("Reading CSV files...")
for ticker in all_tickers:
    filename = f"{ticker}.csv"
    try:
        # Load and standardize
        df = pd.read_csv(filename)
        df.columns = [c.lower() for c in df.columns]
        
        # Parse date
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Grab close
        series = df['close'].rename(ticker).copy()
        
        if df_all.empty:
            df_all = pd.DataFrame(series)
        else:
            df_all = df_all.join(series, how='outer')
            
    except Exception as e:
        print(f"Skipping {ticker}: {e}")

# Filter 2024 - Present
df_all = df_all[df_all.index >= '2024-01-01'].copy()
df_all.ffill(inplace=True)
df_all.dropna(inplace=True)

if df_all.empty:
    raise ValueError("No data found. Check dates/files.")

dates = df_all.index

# Normalize to 100
df_norm = (df_all / df_all.iloc[0]) * 100

# --- 3. Plotly Construction ---
fig = make_subplots(
    rows=1, cols=3, 
    subplot_titles=list(sector_map.keys()),
    horizontal_spacing=0.05
)

trace_configs = []

# --- 4. Add Initial Traces ---
for col_idx, (sector, tickers) in enumerate(sector_map.items(), start=1):
    color = sector_colors[sector]
    
    for i, ticker in enumerate(tickers):
        style = line_styles[i % len(line_styles)]
        
        fig.add_trace(go.Scatter(
            x=[dates[0]], 
            y=[df_norm[ticker].iloc[0]],
            mode='lines',
            name=ticker,
            line=dict(color=color, width=2.5, dash=style),
            legendgroup=sector,
            showlegend=True
        ), row=1, col=col_idx)
        
        trace_configs.append({'ticker': ticker, 'col': col_idx})

# --- 5. Animation Frames ---
num_frames = 60
step = max(1, len(dates) // num_frames)
indices = list(range(1, len(dates), step))
if indices[-1] != len(dates)-1:
    indices.append(len(dates)-1)

frames = []
for idx in indices:
    frame_data = []
    for conf in trace_configs:
        ticker = conf['ticker']
        frame_data.append(go.Scatter(
            x=dates[:idx], 
            y=df_norm[ticker][:idx],
            mode='lines'
        ))
    frames.append(go.Frame(data=frame_data, name=str(idx)))

fig.frames = frames

# --- 6. Layout & Styling ---
off_white = "#e0e0e0"
off_black = "#222222"

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=700, # Increased height to accommodate the larger bottom margin
    width=1000,
    
    # Large bottom margin to fit Legend AND Button without overlap
    margin=dict(t=100, b=250), 
    
    # --- Legend: Positioned just below the X-axis ---
    showlegend=True,
    legend=dict(
        orientation='h',         # Horizontal
        x=0.5, xanchor='center', # Centered
        y=-0.15, yanchor='top',  # Just below the plots
        font=dict(color=off_white, size=12),
        bgcolor='rgba(0,0,0,0)',
        bordercolor='rgba(0,0,0,0)'
    ),
    
    # --- Play Button: Pushed significantly further down ---
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.35, 'xanchor': 'center', # Moves button way down
        'font': {'color': off_black},
        'bgcolor': off_white,
        'buttons': [{'label': '▶ Play 2024-Present', 'method': 'animate',
                     'args': [None, {'frame': {'duration': 20, 'redraw': False}, 'fromcurrent': True}]}]
    }]
)

axis_style = dict(
    showgrid=True,
    gridcolor='rgba(255,255,255,0.1)',
    tickfont=dict(color=off_white),
    linecolor=off_white,
    zeroline=False,
    title_font=dict(color=off_white)
)

fig.update_xaxes(axis_style)
fig.update_yaxes(axis_style)

# --- 7. Y-Axis Limits (As Requested) ---

# Tech (Auto/Data Driven)
tech_min = df_norm[sector_map['Tech']].min().min() * 0.95
tech_max = df_norm[sector_map['Tech']].max().max() * 1.05
fig.update_xaxes(range=[dates[0], dates[-1]], row=1, col=1)
fig.update_yaxes(range=[tech_min, tech_max], title_text="Wealth Index", row=1, col=1)

# Healthcare (40 to 160)
fig.update_xaxes(range=[dates[0], dates[-1]], row=1, col=2)
fig.update_yaxes(range=[40, 160], row=1, col=2)

# Staples (90 to 250)
fig.update_xaxes(range=[dates[0], dates[-1]], row=1, col=3)
fig.update_yaxes(range=[90, 250], row=1, col=3)

# Titles
for annotation in fig['layout']['annotations']:
    annotation['font'] = dict(color=off_white, size=16)

fig.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- 1. Configuration ---
tickers = ['UNH', 'JNJ', 'AMGN']
sector_color = '#2ca02c' # Green
line_styles = ['solid', 'dash', 'dot']

# --- 2. Data Loading ---
df_all = pd.DataFrame()

print("Reading CSV files...")
for ticker in tickers:
    filename = f"{ticker}.csv"
    try:
        df = pd.read_csv(filename)
        df.columns = [c.lower() for c in df.columns]
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        series = df['close'].rename(ticker).copy()
        
        if df_all.empty:
            df_all = pd.DataFrame(series)
        else:
            df_all = df_all.join(series, how='outer')
            
    except Exception as e:
        print(f"Skipping {ticker}: {e}")

# Filter 2024 - Present
df_all = df_all[df_all.index >= '2024-01-01'].copy()
df_all.ffill(inplace=True)
df_all.dropna(inplace=True)

if df_all.empty:
    raise ValueError("No data found.")

dates = df_all.index

# Normalize to 100
df_norm = (df_all / df_all.iloc[0]) * 100
# Calculate Returns
df_pct = ((df_all / df_all.iloc[0]) - 1) * 100

# --- 3. Plotly Construction ---
fig = go.Figure()

# --- 4. Add Initial Traces ---
for i, ticker in enumerate(tickers):
    style = line_styles[i % len(line_styles)]
    
    # Line Trace
    fig.add_trace(go.Scatter(
        x=[dates[0]], 
        y=[df_norm[ticker].iloc[0]],
        mode='lines',
        name=ticker,
        line=dict(color=sector_color, width=3, dash=style),
        legendgroup=ticker
    ))
    
    # Label Trace
    fig.add_trace(go.Scatter(
        x=[dates[0]], 
        y=[df_norm[ticker].iloc[0]],
        mode='markers+text',
        name=ticker,
        marker=dict(color=sector_color, size=10),
        text=[f"0.0%"],
        textposition="middle right",
        textfont=dict(color="white", size=14, family="monospace", weight="bold"),
        showlegend=False,
        legendgroup=ticker
    ))

# --- 5. Animation Frames ---
num_frames = 90
step = max(1, len(dates) // num_frames)
indices = list(range(1, len(dates), step))
if indices[-1] != len(dates)-1:
    indices.append(len(dates)-1)

frames = []
for idx in indices:
    frame_data = []
    current_date = dates[idx]
    
    for ticker in tickers:
        # Update Line
        frame_data.append(go.Scatter(
            x=dates[:idx], 
            y=df_norm[ticker][:idx],
            mode='lines'
        ))
        
        # Update Label
        current_val = df_norm[ticker].iloc[idx]
        current_ret = df_pct[ticker].iloc[idx]
        label_text = f" {ticker}: {current_ret:+.1f}%"
        
        frame_data.append(go.Scatter(
            x=[current_date],
            y=[current_val],
            mode='markers+text',
            text=[label_text]
        ))
        
    frames.append(go.Frame(data=frame_data, name=str(idx)))

fig.frames = frames

# --- 6. Layout & Styling ---
off_white = "#e0e0e0"
off_black = "#222222"

fig.update_layout(
    title=dict(
        text="<b>Healthcare Sector Idiosyncratic Risk Analysis</b><br><sup>Divergence of Returns (2024-Present)</sup>",
        font=dict(color=off_white, size=22),
        x=0.05
    ),
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=700,
    width=1200,
    margin=dict(t=100, b=150, r=50),
    
    showlegend=True,
    legend=dict(
        orientation='h',
        x=0.5, xanchor='center',
        y=-0.1, yanchor='top',
        font=dict(color=off_white, size=14)
    ),
    
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.25, 'xanchor': 'center',
        'font': {'color': off_black},
        'bgcolor': off_white,
        'buttons': [{'label': '▶ Play Risk Simulation', 'method': 'animate',
                     'args': [None, {'frame': {'duration': 30, 'redraw': False}, 'fromcurrent': True}]}]
    }]
)

axis_style = dict(
    showgrid=True,
    gridcolor='rgba(255,255,255,0.1)',
    tickfont=dict(color=off_white),
    linecolor=off_white,
    zeroline=False,
    title_font=dict(color=off_white)
)

fig.update_xaxes(axis_style)
fig.update_yaxes(axis_style)

# --- KEY FIX: Extend X-Axis ---
# We calculate a buffer date 180 days (approx 6 months) past the last data point
end_date_buffer = dates[-1] + pd.Timedelta(days=180)

fig.update_xaxes(
    range=[dates[0], end_date_buffer], # Force the range to include empty space
    title_text="Date"
)

fig.update_yaxes(
    range=[40, 160], 
    title_text="Wealth Index (Start=100)"
)

fig.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. Configuration ---
sector_map = {
    'Tech': ['AAPL', 'MSFT', 'AVGO'],
    'Healthcare': ['UNH', 'JNJ', 'AMGN'],
    'Consumer Staples': ['WMT', 'COST', 'PG']
}

sector_colors = {
    'Tech': '#1f77b4',       # Blue
    'Healthcare': '#2ca02c', # Green
    'Consumer Staples': '#d62728' # Red
}

line_styles = ['solid', 'dash', 'dot']
all_tickers = [t for tickers in sector_map.values() for t in tickers]

# --- 2. Data Loading ---
df_all = pd.DataFrame()

print("Reading CSV files...")
for ticker in all_tickers:
    filename = f"{ticker}.csv"
    try:
        df = pd.read_csv(filename)
        df.columns = [c.lower() for c in df.columns]
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        series = df['close'].rename(ticker).copy()
        
        if df_all.empty:
            df_all = pd.DataFrame(series)
        else:
            df_all = df_all.join(series, how='outer')
            
    except Exception as e:
        print(f"Skipping {ticker}: {e}")

# Filter 2024 - Present
df_all = df_all[df_all.index >= '2024-01-01'].copy()
df_all.ffill(inplace=True)
df_all.dropna(inplace=True)

if df_all.empty:
    raise ValueError("No data found.")

dates = df_all.index

# Normalize to 100 (Wealth Index)
df_norm = (df_all / df_all.iloc[0]) * 100

# Calculate Sector Composites (Equally Weighted)
for sector, tickers in sector_map.items():
    df_norm[f"{sector}_Composite"] = df_norm[tickers].mean(axis=1)

# --- 3. Generate Titles with Performance ---
titles = []
for sector in sector_map.keys():
    # Calculate Total Return: (End Value - 100)
    final_val = df_norm[f"{sector}_Composite"].iloc[-1]
    total_ret = final_val - 100
    titles.append(f"{sector} | {total_ret:+.1f}%")

# --- 4. Plotly Construction ---
fig = make_subplots(
    rows=1, cols=3, 
    subplot_titles=titles,
    horizontal_spacing=0.05
)

trace_configs = []

# --- 5. Add Initial Traces ---
for col_idx, (sector, tickers) in enumerate(sector_map.items(), start=1):
    base_color = sector_colors[sector]
    
    # A. Individual Stocks (Faded, Thin)
    for i, ticker in enumerate(tickers):
        style = line_styles[i % len(line_styles)]
        
        fig.add_trace(go.Scatter(
            x=[dates[0]], 
            y=[df_norm[ticker].iloc[0]],
            mode='lines',
            name=ticker,
            opacity=0.4,  # Faded
            line=dict(color=base_color, width=1, dash=style), # Thin
            showlegend=False
        ), row=1, col=col_idx)
        
        trace_configs.append({'name': ticker, 'col': col_idx, 'type': 'stock'})

    # B. Composite Index (Solid, Medium Thickness)
    fig.add_trace(go.Scatter(
        x=[dates[0]], 
        y=[df_norm[f"{sector}_Composite"].iloc[0]],
        mode='lines',
        name=f"{sector} Index",
        opacity=1.0,
        line=dict(color=base_color, width=3), # Thinner than before (was 5)
        showlegend=False
    ), row=1, col=col_idx)
    
    trace_configs.append({'name': f"{sector}_Composite", 'col': col_idx, 'type': 'composite'})

# --- 6. Animation Frames ---
num_frames = 60
step = max(1, len(dates) // num_frames)
indices = list(range(1, len(dates), step))
if indices[-1] != len(dates)-1:
    indices.append(len(dates)-1)

frames = []
for idx in indices:
    frame_data = []
    
    for conf in trace_configs:
        name = conf['name']
        frame_data.append(go.Scatter(
            x=dates[:idx], 
            y=df_norm[name][:idx],
            mode='lines'
        ))
            
    frames.append(go.Frame(data=frame_data, name=str(idx)))

fig.frames = frames

# --- 7. Layout & Styling ---
off_white = "#e0e0e0"
off_black = "#222222"

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=600,
    width=1000,
    margin=dict(t=100, b=100), # Reduced bottom margin since legend is gone
    
    showlegend=False, # Global disable
    
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.15, 'xanchor': 'center',
        'font': {'color': off_black},
        'bgcolor': off_white,
        'buttons': [{'label': '▶ Play Sector Correlation', 'method': 'animate',
                     'args': [None, {'frame': {'duration': 20, 'redraw': False}, 'fromcurrent': True}]}]
    }]
)

axis_style = dict(
    showgrid=True,
    gridcolor='rgba(255,255,255,0.1)',
    tickfont=dict(color=off_white),
    linecolor=off_white,
    zeroline=False,
    title_font=dict(color=off_white)
)

fig.update_xaxes(axis_style)
fig.update_yaxes(axis_style)

# --- 8. Y-Axis Limits ---
# Tech
tech_min = df_norm[sector_map['Tech']].min().min() * 0.95
tech_max = df_norm[sector_map['Tech']].max().max() * 1.05
fig.update_xaxes(range=[dates[0], dates[-1]], row=1, col=1)
fig.update_yaxes(range=[tech_min, tech_max], title_text="Wealth Index", row=1, col=1)

# Healthcare
fig.update_xaxes(range=[dates[0], dates[-1]], row=1, col=2)
fig.update_yaxes(range=[40, 160], row=1, col=2)

# Staples
fig.update_xaxes(range=[dates[0], dates[-1]], row=1, col=3)
fig.update_yaxes(range=[90, 250], row=1, col=3)

# Style Titles
for annotation in fig['layout']['annotations']:
    annotation['font'] = dict(color=off_white, size=16)

fig.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- 1. Configuration ---
sector_map = {
    'Tech': ['AAPL', 'MSFT', 'AVGO'],
    'Healthcare': ['UNH', 'JNJ', 'AMGN'],
    'Consumer Staples': ['WMT', 'COST', 'PG']
}

sector_colors = {
    'Tech': '#1f77b4',       # Blue
    'Healthcare': '#2ca02c', # Green
    'Consumer Staples': '#d62728', # Red
    'Composite Portfolio': '#ffffff' # White (Replaces SPY)
}

# Load only the sector stocks (Removed SPY)
all_tickers = [t for tickers in sector_map.values() for t in tickers]

# --- 2. Data Loading ---
df_all = pd.DataFrame()

print("Reading CSV files...")
for ticker in all_tickers:
    filename = f"{ticker}.csv"
    try:
        df = pd.read_csv(filename)
        df.columns = [c.lower() for c in df.columns]
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        series = df['close'].rename(ticker).copy()
        
        if df_all.empty:
            df_all = pd.DataFrame(series)
        else:
            df_all = df_all.join(series, how='outer')
            
    except Exception as e:
        print(f"Skipping {ticker}: {e}")

# Filter 2024 - Present
df_all = df_all[df_all.index >= '2024-01-01'].copy()
df_all.ffill(inplace=True)
df_all.dropna(inplace=True)

if df_all.empty:
    raise ValueError("No data found.")

dates = df_all.index

# --- 3. Create Portfolios ---
df_norm = (df_all / df_all.iloc[0]) * 100
df_plot = pd.DataFrame(index=dates)

# A. Calculate Sector Portfolios
for sector, tickers in sector_map.items():
    df_plot[sector] = df_norm[tickers].mean(axis=1)

# B. Calculate Composite Portfolio (Average of the 3 Sector Indices)
# This creates an equally weighted benchmark of your sectors
df_plot['Composite Portfolio'] = df_plot[list(sector_map.keys())].mean(axis=1)

# Calculate Returns for Labels
df_pct = ((df_plot / 100) - 1) * 100

# --- 4. Plotly Construction ---
fig = go.Figure()

# Helper for style
def get_style(name):
    color = sector_colors.get(name, '#ffffff')
    # Make Composite line slightly thicker for emphasis
    width = 4 if name == 'Composite Portfolio' else 2.5
    return dict(color=color, width=width, dash='solid')

# Add Initial Traces (Static)
for name in df_plot.columns:
    # 1. Line Trace
    fig.add_trace(go.Scatter(
        x=[dates[0]], 
        y=[df_plot[name].iloc[0]],
        mode='lines',
        name=name,
        line=get_style(name),
        legendgroup=name
    ))
    
    # 2. Label Trace
    fig.add_trace(go.Scatter(
        x=[dates[0]], 
        y=[df_plot[name].iloc[0]],
        mode='markers+text',
        name=name,
        marker=dict(color=get_style(name)['color'], size=8),
        text=[f"0.0%"],
        textposition="middle right",
        textfont=dict(color="white", size=14, family="monospace", weight="bold"),
        showlegend=False,
        legendgroup=name
    ))

# --- 5. Animation Frames ---
num_frames = 90
step = max(1, len(dates) // num_frames)
indices = list(range(1, len(dates), step))
if indices[-1] != len(dates)-1:
    indices.append(len(dates)-1)

frames = []
for idx in indices:
    frame_data = []
    current_date = dates[idx]
    
    for name in df_plot.columns:
        # Update Line
        frame_data.append(go.Scatter(
            x=dates[:idx], 
            y=df_plot[name][:idx],
            mode='lines'
        ))
        
        # Update Label
        current_val = df_plot[name].iloc[idx]
        current_ret = df_pct[name].iloc[idx]
        label_text = f" {name}: {current_ret:+.1f}%"
        
        frame_data.append(go.Scatter(
            x=[current_date],
            y=[current_val],
            mode='markers+text',
            text=[label_text]
        ))
        
    frames.append(go.Frame(data=frame_data, name=str(idx)))

fig.frames = frames

# --- 6. Layout & Annotations ---
off_white = "#e0e0e0"
off_black = "#222222"

# Define the two vertical red lines (Dates from your snippet)
drawdown_lines = [
    dict(
        type="line",
        x0="2025-03-01", y0=0, x1="2025-03-01", y1=1,
        xref="x", yref="paper",
        line=dict(color="red", width=1.5, dash="dash")
    ),
    dict(
        type="line",
        x0="2025-04-20", y0=0, x1="2025-04-20", y1=1,
        xref="x", yref="paper",
        line=dict(color="red", width=1.5, dash="dash")
    )
]

fig.update_layout(
    title=dict(
        text="<b>Systematic Risk Analysis</b><br><sup>Sector Performance vs. Composite Portfolio (Equal Weighted)</sup>",
        font=dict(color=off_white, size=22),
        x=0.05
    ),
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=700,
    width=1200,
    margin=dict(t=100, b=150, r=50),
    
    # Add the red lines here
    shapes=drawdown_lines,
    
    showlegend=True,
    legend=dict(
        orientation='h',
        x=0.5, xanchor='center',
        y=-0.1, yanchor='top',
        font=dict(color=off_white, size=14)
    ),
    
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.25, 'xanchor': 'center',
        'font': {'color': off_black},
        'bgcolor': off_white,
        'buttons': [{'label': '▶ Play Risk Analysis', 'method': 'animate',
                     'args': [None, {'frame': {'duration': 30, 'redraw': False}, 'fromcurrent': True}]}]
    }]
)

axis_style = dict(
    showgrid=True,
    gridcolor='rgba(255,255,255,0.1)',
    tickfont=dict(color=off_white),
    linecolor=off_white,
    zeroline=False,
    title_font=dict(color=off_white)
)

fig.update_xaxes(axis_style)
fig.update_yaxes(axis_style)

# Extend X-axis
end_date_buffer = dates[-1] + pd.Timedelta(days=180)
fig.update_xaxes(range=[dates[0], end_date_buffer], title_text="Date")

# Y-axis
y_min = df_plot.min().min() * 0.95
y_max = df_plot.max().max() * 1.05
fig.update_yaxes(range=[y_min, y_max], title_text="Wealth Index (Start=100)")

fig.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. Configuration ---
sector_map = {
    'Tech': ['AAPL', 'MSFT', 'AVGO'],
    'Healthcare': ['UNH', 'JNJ', 'AMGN'],
    'Consumer Staples': ['WMT', 'COST', 'PG']
}

# Map sectors to Plotly built-in sequential colorscales
# Tech -> Blue, Healthcare -> Green, Staples -> Red
color_scales = {
    'Tech': 'Blues',
    'Healthcare': 'Greens',
    'Consumer Staples': 'Reds'
}

all_tickers = [t for tickers in sector_map.values() for t in tickers]

# --- 2. Data Loading ---
df_all = pd.DataFrame()

print("Reading CSV files...")
for ticker in all_tickers:
    filename = f"{ticker}.csv"
    try:
        df = pd.read_csv(filename)
        df.columns = [c.lower() for c in df.columns]
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        series = df['close'].rename(ticker).copy()
        
        if df_all.empty:
            df_all = pd.DataFrame(series)
        else:
            df_all = df_all.join(series, how='outer')
            
    except Exception as e:
        print(f"Skipping {ticker}: {e}")

# Filter 2024 - Present
df_all = df_all[df_all.index >= '2024-01-01'].copy()
df_all.ffill(inplace=True)
df_all.dropna(inplace=True)

if df_all.empty:
    raise ValueError("No data found.")

# Calculate Daily Returns for Correlation
df_returns = df_all.pct_change().dropna()

# --- 3. Plotly Construction ---
fig = make_subplots(
    rows=1, cols=3, 
    subplot_titles=list(sector_map.keys()),
    horizontal_spacing=0.10
)

for col_idx, (sector, tickers) in enumerate(sector_map.items(), start=1):
    # 1. Slice returns for this sector
    valid_tickers = [t for t in tickers if t in df_returns.columns]
    sector_rets = df_returns[valid_tickers]
    
    # 2. Calculate Correlation Matrix
    corr_matrix = sector_rets.corr()
    
    # 3. Add Heatmap
    fig.add_trace(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale=color_scales[sector],
        zmin=0, zmax=1,
        showscale=False, # Hide individual colorbars
        text=corr_matrix.values,
        texttemplate="%{z:.2f}", # Show values with 2 decimals
        # Removed the problematic 'textfont' argument to use default black/readable text
    ), row=1, col=col_idx)

# --- 4. Layout & Styling ---
off_white = "#e0e0e0"

fig.update_layout(
    title=dict(
        text="<b>Intra-Sector Correlation Matrix</b><br><sup>(2024-Present)</sup>",
        font=dict(color=off_white, size=22),
        x=0.05
    ),
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=500,
    width=1100,
    margin=dict(t=120, b=50),
)

# Force square aspect ratio
fig.update_yaxes(scaleanchor="x", scaleratio=1)

fig.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- 1. Configuration ---
tickers = ['AVGO', 'AMGN', 'WMT']

# --- 2. Data Loading ---
df_all = pd.DataFrame()

print("Reading CSV files...")
for ticker in tickers:
    filename = f"{ticker}.csv"
    try:
        df = pd.read_csv(filename)
        df.columns = [c.lower() for c in df.columns]
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        series = df['close'].rename(ticker).copy()
        
        if df_all.empty:
            df_all = pd.DataFrame(series)
        else:
            df_all = df_all.join(series, how='outer')
            
    except Exception as e:
        print(f"Skipping {ticker}: {e}")

# Filter 2024 - Present
df_all = df_all[df_all.index >= '2024-01-01'].copy()
df_all.ffill(inplace=True)
df_all.dropna(inplace=True)

if df_all.empty:
    raise ValueError("No data found.")

# Calculate Daily Returns
df_returns = df_all.pct_change().dropna()

# --- 3. Calculate Correlation ---
corr_matrix = df_returns.corr()

# --- 4. Plotly Construction ---
fig = go.Figure()

fig.add_trace(go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.index,
    colorscale='Viridis', # A good scale for mixed correlations
    zmin=-1, zmax=1,      # Range from -1 to 1 for full correlation spectrum
    text=corr_matrix.values,
    texttemplate="%{z:.2f}",
    showscale=True,
    colorbar=dict(title="Correlation")
))

# --- 5. Layout & Styling ---
off_white = "#e0e0e0"

fig.update_layout(
    title=dict(
        text="<b>Cross-Sector Correlation Matrix</b><br><sup>AVGO (Tech) vs. AMGN (Healthcare) vs. WMT (Staples)</sup>",
        font=dict(color=off_white, size=22),
        x=0.5, xanchor='center'
    ),
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=600,
    width=700,
    margin=dict(t=120, b=50),
)

# Force square aspect ratio
fig.update_yaxes(scaleanchor="x", scaleratio=1)

fig.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- 1. Configuration ---
sector_map = {
    'Tech': ['AAPL', 'MSFT', 'AVGO'],
    'Healthcare': ['UNH', 'JNJ', 'AMGN'],
    'Consumer Staples': ['WMT', 'COST', 'PG']
}

sector_colors = {
    'Tech': '#1f77b4',       # Blue
    'Healthcare': '#2ca02c', # Green
    'Consumer Staples': '#d62728' # Red
}

# 2 Weeks = approx 10 trading days
window_size = 30 
all_tickers = [t for tickers in sector_map.values() for t in tickers]

# --- 2. Data Loading ---
df_all = pd.DataFrame()

print("Reading CSV files...")
for ticker in all_tickers:
    filename = f"{ticker}.csv"
    try:
        df = pd.read_csv(filename)
        df.columns = [c.lower() for c in df.columns]
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        series = df['close'].rename(ticker).copy()
        
        if df_all.empty:
            df_all = pd.DataFrame(series)
        else:
            df_all = df_all.join(series, how='outer')
            
    except Exception as e:
        print(f"Skipping {ticker}: {e}")

# Filter 2024 - Present
df_all = df_all[df_all.index >= '2025-01-01'].copy()
df_all.ffill(inplace=True)
df_all.dropna(inplace=True)

if df_all.empty:
    raise ValueError("No data found.")

# Calculate Daily Returns
df_rets = df_all.pct_change()

# --- 3. Build Indices ---
sector_rets = pd.DataFrame(index=df_rets.index)

# A. Create Sector Indices (Equal Weight of their stocks)
for sector, tickers in sector_map.items():
    sector_rets[sector] = df_rets[tickers].mean(axis=1)

# B. Create Grand Market Index (Equal Weight of the 3 Sectors)
# This serves as the "All-Sector" benchmark to test correlation against
sector_rets['Grand_Index'] = sector_rets[list(sector_map.keys())].mean(axis=1)

# --- 4. Calculate Rolling Correlation ---
df_corr = pd.DataFrame(index=sector_rets.index)

for sector in sector_map.keys():
    # Rolling correlation of Sector vs. Grand Index
    df_corr[sector] = sector_rets[sector].rolling(window=window_size).corr(sector_rets['Grand_Index'])

# Drop NaNs from the start
df_corr.dropna(inplace=True)
dates = df_corr.index

# --- 5. Plotly Construction ---
fig = go.Figure()

# Add Initial Traces (Static)
for sector in sector_map.keys():
    color = sector_colors[sector]
    
    # Line (Thinner width=1.5)
    fig.add_trace(go.Scatter(
        x=[dates[0]], 
        y=[df_corr[sector].iloc[0]],
        mode='lines',
        name=sector,
        line=dict(color=color, width=1.5),
        legendgroup=sector
    ))
    
    # Floating Label
    fig.add_trace(go.Scatter(
        x=[dates[0]], 
        y=[df_corr[sector].iloc[0]],
        mode='markers+text',
        name=sector,
        marker=dict(color=color, size=6),
        text=[f"{df_corr[sector].iloc[0]:.2f}"],
        textposition="middle right",
        textfont=dict(color="white", size=13, family="monospace"),
        showlegend=False,
        legendgroup=sector
    ))

# --- 6. Animation Frames ---
num_frames = 90
step = max(1, len(dates) // num_frames)
indices = list(range(1, len(dates), step))
if indices[-1] != len(dates)-1:
    indices.append(len(dates)-1)

frames = []
for idx in indices:
    frame_data = []
    current_date = dates[idx]
    
    for sector in sector_map.keys():
        # Update Line
        frame_data.append(go.Scatter(
            x=dates[:idx], 
            y=df_corr[sector][:idx],
            mode='lines'
        ))
        
        # Update Label
        current_val = df_corr[sector].iloc[idx]
        frame_data.append(go.Scatter(
            x=[current_date],
            y=[current_val],
            mode='markers+text',
            text=[f" {sector}: {current_val:.2f}"]
        ))
        
    frames.append(go.Frame(data=frame_data, name=str(idx)))

fig.frames = frames

# --- 7. Layout & Styling ---
off_white = "#e0e0e0"
off_black = "#222222"

# Define the Vertical Red Lines (Drawdown Markers)
drawdown_lines = [
    dict(
        type="line",
        x0="2025-04-20", y0=0, x1="2025-04-20", y1=1,
        xref="x", yref="paper",
        line=dict(color="red", width=1.5, dash="dash")
    ),
    dict(
        type="line",
        x0="2025-04-01", y0=0, x1="2025-04-01", y1=1,
        xref="x", yref="paper",
        line=dict(color="red", width=1.5, dash="dash")
    )
]

fig.update_layout(
    title=dict(
        text="<b>Sector Correlation Dynamics</b><br><sup>Rolling 2-Week Correlation: Sector vs. Aggregate Portfolio</sup>",
        font=dict(color=off_white, size=22),
        x=0.05
    ),
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=600,
    width=1200,
    margin=dict(t=100, b=150, r=80),
    
    # Add vertical lines
    shapes=drawdown_lines,
    
    showlegend=True,
    legend=dict(
        orientation='h',
        x=0.5, xanchor='center',
        y=-0.15, yanchor='top',
        font=dict(color=off_white, size=14)
    ),
    
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.25, 'xanchor': 'center',
        'font': {'color': off_black},
        'bgcolor': off_white,
        'buttons': [{'label': '▶ Play Sector Correlations', 'method': 'animate',
                     'args': [None, {'frame': {'duration': 30, 'redraw': False}, 'fromcurrent': True}]}]
    }]
)

axis_style = dict(
    showgrid=True,
    gridcolor='rgba(255,255,255,0.1)',
    tickfont=dict(color=off_white),
    linecolor=off_white,
    zeroline=False,
    title_font=dict(color=off_white)
)

fig.update_xaxes(axis_style)
fig.update_yaxes(axis_style)

# Extend X-axis to see the future lines and label text
end_date_buffer = dates[-1] + pd.Timedelta(days=90)
fig.update_xaxes(range=[dates[0], end_date_buffer], title_text="Date")

# Y-Axis (-1 to 1, though usually positive for sectors)
fig.update_yaxes(range=[-0.5, 1.1], title_text="Correlation to Aggregate Portfolio")

fig.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. Configuration ---
sector_map = {
    'Tech': ['AAPL', 'MSFT', 'AVGO'],
    'Healthcare': ['UNH', 'JNJ', 'AMGN'],
    'Consumer Staples': ['WMT', 'COST', 'PG']
}
all_tickers = [t for tickers in sector_map.values() for t in tickers] + ['SPY']

# Window for rolling correlation (approx 3 months)
window_size = 63 

# --- 2. Data Loading ---
df_all = pd.DataFrame()

print("Reading CSV files...")
for ticker in all_tickers:
    filename = f"{ticker}.csv"
    try:
        df = pd.read_csv(filename)
        df.columns = [c.lower() for c in df.columns]
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        series = df['close'].rename(ticker).copy()
        
        if df_all.empty:
            df_all = pd.DataFrame(series)
        else:
            df_all = df_all.join(series, how='outer')
            
    except Exception as e:
        print(f"Skipping {ticker}: {e}")

# Filter 2024 - Present
df_all = df_all[df_all.index >= '2024-01-01'].copy()
df_all.ffill(inplace=True)
df_all.dropna(inplace=True)

if df_all.empty:
    raise ValueError("No data found.")

dates = df_all.index

# --- 3. Calculations ---
# A. Daily Returns
df_rets = df_all.pct_change()

# B. Construct Portfolio Return (Equal Weight of 9 stocks)
stock_cols = [c for c in df_rets.columns if c != 'SPY']
df_rets['Portfolio'] = df_rets[stock_cols].mean(axis=1)

# C. Rolling Correlation (Portfolio vs SPY)
rolling_corr = df_rets['Portfolio'].rolling(window=window_size).corr(df_rets['SPY'])
rolling_corr.fillna(0, inplace=True)

# D. Wealth Indices (Start=100)
df_norm = pd.DataFrame(index=dates)
df_norm['Portfolio'] = (1 + df_rets['Portfolio']).cumprod() * 100
df_norm['SPY'] = (1 + df_rets['SPY']).cumprod() * 100

# Align all data
df_plot = df_norm.join(rolling_corr.rename('Corr'))
df_plot = df_plot.iloc[window_size:] 
dates_plot = df_plot.index

# --- Calculate Final Performance for Legend ---
spy_ret = (df_plot['SPY'].iloc[-1] / df_plot['SPY'].iloc[0] - 1) * 100
port_ret = (df_plot['Portfolio'].iloc[-1] / df_plot['Portfolio'].iloc[0] - 1) * 100

spy_name = f"SPY Benchmark ({spy_ret:+.1f}%)"
port_name = f"Eq. Wgt Portfolio ({port_ret:+.1f}%)"

# --- 4. Plotly Construction ---
fig = make_subplots(
    rows=2, cols=1,
    row_heights=[0.7, 0.3], 
    vertical_spacing=0.1,
    subplot_titles=("<b>Performance: Portfolio vs. SPY</b>", "<b>Rolling Correlation (60-Day)</b>")
)

# --- Initial Traces (Static) ---
# Top Chart: SPY (Gray)
fig.add_trace(go.Scatter(
    x=[dates_plot[0]], y=[df_plot['SPY'].iloc[0]],
    mode='lines', name=spy_name,
    line=dict(color='#888888', width=2),
    legendgroup='perf'
), row=1, col=1)

# Top Chart: Portfolio (White)
fig.add_trace(go.Scatter(
    x=[dates_plot[0]], y=[df_plot['Portfolio'].iloc[0]],
    mode='lines', name=port_name,
    line=dict(color='white', width=3),
    legendgroup='perf'
), row=1, col=1)

# Bottom Chart: Rolling Correlation (Yellow)
fig.add_trace(go.Scatter(
    x=[dates_plot[0]], y=[df_plot['Corr'].iloc[0]],
    mode='lines', name='Correlation',
    line=dict(color='#FFD700', width=2), # Gold
    fill='tozeroy', 
    fillcolor='rgba(255, 215, 0, 0.1)',
    showlegend=False
), row=2, col=1)

# --- 5. Animation Frames ---
num_frames = 90
step = max(1, len(dates_plot) // num_frames)
indices = list(range(1, len(dates_plot), step))
if indices[-1] != len(dates_plot)-1:
    indices.append(len(dates_plot)-1)

frames = []
for idx in indices:
    frame_data = [
        # 1. SPY Line
        go.Scatter(x=dates_plot[:idx], y=df_plot['SPY'][:idx]),
        # 2. Portfolio Line
        go.Scatter(x=dates_plot[:idx], y=df_plot['Portfolio'][:idx]),
        # 3. Correlation Line
        go.Scatter(x=dates_plot[:idx], y=df_plot['Corr'][:idx]),
    ]
    frames.append(go.Frame(data=frame_data, name=str(idx)))

fig.frames = frames

# --- 6. Layout ---
off_white = "#e0e0e0"
off_black = "#222222"

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=700,
    width=1100, 
    
    # Legend Position: Outside Right
    showlegend=True,
    legend=dict(
        orientation='v', 
        y=0.5, yanchor='middle',
        x=1.02, xanchor='left',
        font=dict(color=off_white, size=12)
    ),
    
    margin=dict(t=80, b=100, r=150),
    
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.15, 'xanchor': 'center',
        'font': {'color': off_black},
        'bgcolor': off_white,
        'buttons': [{'label': '▶ Play Analysis', 'method': 'animate',
                     'args': [None, {'frame': {'duration': 20, 'redraw': False}, 'fromcurrent': True}]}]
    }]
)

# Styling Axes
axis_style = dict(
    showgrid=True, gridcolor='rgba(255,255,255,0.1)',
    tickfont=dict(color=off_white), title_font=dict(color=off_white)
)
fig.update_xaxes(axis_style)
fig.update_yaxes(axis_style)

# Set Ranges
# Top Chart
y_min = df_plot[['Portfolio', 'SPY']].min().min() * 0.95
y_max = df_plot[['Portfolio', 'SPY']].max().max() * 1.05
fig.update_yaxes(range=[y_min, y_max], title="Wealth Index", row=1, col=1)
fig.update_xaxes(range=[dates_plot[0], dates_plot[-1]], row=1, col=1)

# Bottom Chart
fig.update_yaxes(range=[-0.5, 1.1], title="Correlation", row=2, col=1)
fig.update_xaxes(range=[dates_plot[0], dates_plot[-1]], title="Date", row=2, col=1)

# Style Titles
for annotation in fig['layout']['annotations']:
    annotation['font'] = dict(color=off_white, size=14)

fig.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- 1. Configuration ---
sector_map = {
    'Tech': ['AAPL', 'MSFT', 'AVGO'],
    'Healthcare': ['UNH', 'JNJ', 'AMGN'],
    'Consumer Staples': ['WMT', 'COST', 'PG']
}
all_tickers = [t for tickers in sector_map.values() for t in tickers]

# --- 2. Data Loading ---
df_all = pd.DataFrame()

print("Reading CSV files...")
for ticker in all_tickers:
    filename = f"{ticker}.csv"
    try:
        df = pd.read_csv(filename)
        df.columns = [c.lower() for c in df.columns]
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        series = df['close'].rename(ticker).copy()
        
        if df_all.empty:
            df_all = pd.DataFrame(series)
        else:
            df_all = df_all.join(series, how='outer')
            
    except Exception as e:
        print(f"Skipping {ticker}: {e}")

# Filter 2024 - Present
df_all = df_all[df_all.index >= '2024-01-01'].copy()
df_all.ffill(inplace=True)
df_all.dropna(inplace=True)

if df_all.empty:
    raise ValueError("No data found.")

# --- 3. Spectral Decomposition (PCA) ---
# Calculate Daily Log Returns (Standard for PCA)
df_rets = np.log(df_all / df_all.shift(1)).dropna()

# Compute Correlation Matrix
corr_matrix = df_rets.corr()

# Eigendecomposition
# eigh is for symmetric matrices (like correlation matrices), returns sorted eigenvalues
eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)

# Sort descending (eigh returns ascending)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Calculate Explained Variance
total_var = np.sum(eigenvalues)
explained_variance_ratio = eigenvalues / total_var
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# --- 4. Plotly Construction ---
fig = go.Figure()

# A. Scree Plot (Bar Chart of Eigenvalues)
fig.add_trace(go.Bar(
    x=[f"PC{i+1}" for i in range(len(eigenvalues))],
    y=eigenvalues,
    name='Eigenvalue',
    marker_color='#00d1ff', # Cyan
    opacity=0.7
))

# B. Cumulative Variance (Line Chart)
# Map to secondary Y-axis? Or just show % on hover.
# Standard scree plot plots Eigenvalues. 
# Sometimes people plot Explained Variance % instead.
# Let's plot Explained Variance % (Scree) and Cumulative % (Line).
# The user asked for "Spectral Decay", which usually refers to the Eigenvalues themselves.
# I will stick to Eigenvalues on the left, but maybe add text for %.

# Let's do a Dual Axis: Left=Eigenvalue, Right=Cumulative %
fig.add_trace(go.Scatter(
    x=[f"PC{i+1}" for i in range(len(eigenvalues))],
    y=cumulative_variance_ratio,
    name='Cumulative Variance',
    mode='lines+markers',
    marker=dict(color='white', size=8),
    line=dict(color='white', width=2),
    yaxis='y2'
))

# --- 5. Layout ---
off_white = "#e0e0e0"

fig.update_layout(
    title=dict(
        text="<b>Spectral Decomposition of Portfolio Returns</b><br><sup>Eigenvalue Decay (Scree Plot)</sup>",
        font=dict(color=off_white, size=22),
        x=0.5, xanchor='center'
    ),
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=600,
    width=1000,
    margin=dict(t=100, b=50, r=50),
    showlegend=True,
    legend=dict(
        orientation='h',
        y=1.02, x=0.5, xanchor='center'
    ),
    
    # Axes
    xaxis=dict(
        title="Principal Component (Mode)",
        showgrid=False,
        tickfont=dict(color=off_white),
        title_font=dict(color=off_white)
    ),
    yaxis=dict(
        title="Eigenvalue Magnitude",
        showgrid=True,
        gridcolor='rgba(255,255,255,0.1)',
        tickfont=dict(color=off_white),
        title_font=dict(color=off_white)
    ),
    yaxis2=dict(
        title="Cumulative Explained Variance",
        overlaying='y',
        side='right',
        range=[0, 1.1],
        tickformat='.0%',
        showgrid=False,
        tickfont=dict(color='white'),
        title_font=dict(color='white')
    )
)

# Add Marchenko-Pastur Threshold (Optional, but cool for "Spectral Decay")
# Only if requested, but standard spectral decay is just the scree plot.
# I'll keep it simple.

fig.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- 1. Configuration ---
sector_map = {
    'Tech': ['AAPL', 'MSFT', 'AVGO'],
    'Healthcare': ['UNH', 'JNJ', 'AMGN'],
    'Consumer Staples': ['WMT', 'COST', 'PG']
}

# Color Mapping for Bars
color_map = {
    'AAPL': '#1f77b4', 'MSFT': '#1f77b4', 'AVGO': '#1f77b4', # Blue
    'UNH': '#2ca02c', 'JNJ': '#2ca02c', 'AMGN': '#2ca02c',   # Green
    'WMT': '#d62728', 'COST': '#d62728', 'PG': '#d62728'     # Red
}

all_tickers = [t for tickers in sector_map.values() for t in tickers]

# --- 2. Data Loading ---
df_all = pd.DataFrame()

print("Reading CSV files...")
for ticker in all_tickers:
    filename = f"{ticker}.csv"
    try:
        df = pd.read_csv(filename)
        df.columns = [c.lower() for c in df.columns]
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        series = df['close'].rename(ticker).copy()
        
        if df_all.empty:
            df_all = pd.DataFrame(series)
        else:
            df_all = df_all.join(series, how='outer')
            
    except Exception as e:
        print(f"Skipping {ticker}: {e}")

# Filter 2024 - Present
df_all = df_all[df_all.index >= '2024-01-01'].copy()
df_all.ffill(inplace=True)
df_all.dropna(inplace=True)

if df_all.empty:
    raise ValueError("No data found.")

# Calculate Daily Returns
df_rets = df_all.pct_change().dropna()

# --- 3. Run PCA ---
# Standardize returns (Mean=0, Std=1)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_rets)

# Fit PCA
pca = PCA(n_components=3)
pca.fit(scaled_data)

# Extract Loadings (Eigenvectors)
loadings = pd.DataFrame(
    pca.components_.T, 
    index=df_rets.columns, 
    columns=['PC1', 'PC2', 'PC3']
)

# Explained Variance
exp_var = pca.explained_variance_ratio_ * 100

# --- 4. Plotly Construction ---
fig = make_subplots(
    rows=1, cols=3, 
    subplot_titles=(
        f"<b>PC1 (Market)</b><br>Var Explained: {exp_var[0]:.1f}%", 
        f"<b>PC2 (Sector Factor A)</b><br>Var Explained: {exp_var[1]:.1f}%", 
        f"<b>PC3 (Sector Factor B)</b><br>Var Explained: {exp_var[2]:.1f}%"
    ),
    horizontal_spacing=0.08
)

# Helper to add bars
def add_pc_bars(col_name, col_idx):
    # Sort values for cleaner visualization in the specific PC
    # (Optional: remove sort_values if you want fixed order)
    # subset = loadings[col_name].sort_values(ascending=False)
    subset = loadings[col_name] # Keep fixed order to compare across plots easily
    
    colors = [color_map[t] for t in subset.index]
    
    fig.add_trace(go.Bar(
        x=subset.index,
        y=subset.values,
        marker_color=colors,
        text=subset.values,
        texttemplate="%{text:.2f}",
        textposition="auto",
        showlegend=False
    ), row=1, col=col_idx)

# Add traces for PC1, PC2, PC3
add_pc_bars('PC1', 1)
add_pc_bars('PC2', 2)
add_pc_bars('PC3', 3)

# --- 5. Layout ---
off_white = "#e0e0e0"

fig.update_layout(
    title=dict(
        text="<b>Principal Component Loadings</b><br><sup>Decomposing Risk Factors: Market vs. Sector Exposure</sup>",
        font=dict(color=off_white, size=22),
        x=0.5, xanchor='center'
    ),
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=500,
    width=1000,
    margin=dict(t=120, b=50),
)

# Y-Axis Labels
fig.update_yaxes(title="Loading Value", row=1, col=1)

# Style Annotations
for annotation in fig['layout']['annotations']:
    annotation['font'] = dict(color=off_white, size=14)

# Add a dummy legend for sectors manually (since we disabled auto-legend)
# This is just a visual trick to show the color key
fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='#1f77b4', size=10), name='Tech'))
fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='#2ca02c', size=10), name='Healthcare'))
fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='#d62728', size=10), name='Staples'))

fig.update_layout(
    showlegend=True,
    legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center")
)

fig.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- 1. Configuration ---
sector_map = {
    'Tech': ['AAPL', 'MSFT', 'AVGO'],
    'Healthcare': ['UNH', 'JNJ', 'AMGN'],
    'Consumer Staples': ['WMT', 'COST', 'PG']
}

# Color Mapping
sector_colors = {
    'Tech': '#1f77b4',       # Blue
    'Healthcare': '#2ca02c', # Green
    'Consumer Staples': '#d62728' # Red
}

all_tickers = [t for tickers in sector_map.values() for t in tickers]

# --- 2. Data Loading ---
df_all = pd.DataFrame()

print("Reading CSV files...")
for ticker in all_tickers:
    filename = f"{ticker}.csv"
    try:
        df = pd.read_csv(filename)
        df.columns = [c.lower() for c in df.columns]
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        series = df['close'].rename(ticker).copy()
        
        if df_all.empty:
            df_all = pd.DataFrame(series)
        else:
            df_all = df_all.join(series, how='outer')
            
    except Exception as e:
        print(f"Skipping {ticker}: {e}")

# Filter 2024 - Present
df_all = df_all[df_all.index >= '2024-01-01'].copy()
df_all.ffill(inplace=True)
df_all.dropna(inplace=True)

if df_all.empty:
    raise ValueError("No data found.")

# Calculate Daily Returns
df_rets = df_all.pct_change().dropna()

# --- 3. Calculate Idiosyncratic Risk via PCA ---
# Standardize returns so variance = 1.0 for all stocks
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_rets)

# Fit PCA with 3 Components (Market + Sector Factors)
# We assume the first 3 factors capture the "Systematic" risk
pca = PCA(n_components=3)
components = pca.fit_transform(scaled_data)

# Reconstruct the data using ONLY the systematic factors
reconstructed_data = pca.inverse_transform(components)

# The "Error" between original and reconstructed is the Idiosyncratic part
residuals = scaled_data - reconstructed_data

# Calculate Variance of the residuals
# Since original variance was standardized to 1.0, this result is directly the % Idio
idio_variance = np.var(residuals, axis=0)

# Create DataFrame for Plotting
df_idio = pd.DataFrame({
    'Ticker': df_rets.columns,
    'Idio_Pct': idio_variance * 100,
    'Sector': [next(k for k, v in sector_map.items() if t in v) for t in df_rets.columns]
})

# Sort for better visuals (optional)
df_idio = df_idio.sort_values('Idio_Pct', ascending=False)

# --- 4. Plotly Construction ---
fig = go.Figure()

for sector, color in sector_colors.items():
    subset = df_idio[df_idio['Sector'] == sector]
    
    fig.add_trace(go.Bar(
        x=subset['Ticker'],
        y=subset['Idio_Pct'],
        name=sector,
        marker_color=color,
        text=subset['Idio_Pct'].apply(lambda x: f"{x:.1f}%"),
        textposition='auto',
        textfont=dict(color='white', size=14, weight='bold')
    ))

# --- 5. Layout ---
off_white = "#e0e0e0"

fig.update_layout(
    title=dict(
        text="<b>Idiosyncratic Risk Analysis</b><br><sup>% of Stock Variance Unexplained by Market & Sector Factors</sup>",
        font=dict(color=off_white, size=22),
        x=0.5, xanchor='center'
    ),
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=600,
    width=1000,
    margin=dict(t=120, b=50),
    
    xaxis=dict(
        title="Stock Ticker",
        title_font=dict(size=16),
        tickfont=dict(size=14)
    ),
    
    yaxis=dict(
        title="Idiosyncratic Variance (%)",
        title_font=dict(size=16),
        tickfont=dict(size=14),
        range=[0, 100] # Full percentage scale
    ),
    
    legend=dict(
        orientation="h", 
        y=-0.2, x=0.5, xanchor="center",
        font=dict(size=14)
    ),
    
    # Add an annotation explaining the metric
    annotations=[dict(
        x=0.5, y=1.08, xref="paper", yref="paper",
        text="Higher % = More 'Unique' Movement (Earnings, News, Product Launches)",
        showarrow=False,
        font=dict(color="gray", size=12)
    )]
)

fig.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# --- 1. Data Setup ---
sector_map = {
    'Tech': ['AAPL', 'MSFT', 'AVGO'],
    'Healthcare': ['UNH', 'JNJ', 'AMGN'],
    'Consumer Staples': ['WMT', 'COST', 'PG']
}

sector_colors = {
    'Tech': '#1f77b4',       # Blue
    'Healthcare': '#2ca02c', # Green
    'Consumer Staples': '#d62728' # Red
}

# Combine lists including SPY
all_tickers = [t for tickers in sector_map.values() for t in tickers] + ['SPY']

# --- 2. Data Loading ---
df_all = pd.DataFrame()

print("Reading CSV files...")
for ticker in all_tickers:
    filename = f"{ticker}.csv"
    try:
        df = pd.read_csv(filename)
        df.columns = [c.lower() for c in df.columns]
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        series = df['close'].rename(ticker).copy()
        
        if df_all.empty:
            df_all = pd.DataFrame(series)
        else:
            df_all = df_all.join(series, how='outer')
            
    except Exception as e:
        print(f"Skipping {ticker}: {e}")

# Filter 2024 - Present
df_all = df_all[df_all.index >= '2024-01-01'].copy()
df_all.ffill(inplace=True)
df_all.dropna(inplace=True)

if df_all.empty:
    raise ValueError("No data found.")

# Calculate Daily Returns
df_rets = df_all.pct_change().dropna()

# --- 3. Plotly Construction ---
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("Tech", "Healthcare", "Consumer Staples"),
    horizontal_spacing=0.08,
    x_title="Market Return (SPY)",
    y_title="Sector Return"
)

# Iterate through sectors to Calculate & Plot CAPM
for col_idx, (sector, tickers) in enumerate(sector_map.items(), start=1):
    color = sector_colors[sector]
    
    # Construct Sector Portfolio Returns (Equal Weight)
    valid_tickers = [t for t in tickers if t in df_rets.columns]
    if not valid_tickers: continue
        
    sector_series = df_rets[valid_tickers].mean(axis=1)
    market_series = df_rets['SPY']
    
    # Linear Regression (scipy.stats)
    slope, intercept, r_value, p_value, std_err = stats.linregress(market_series, sector_series)
    beta = slope
    alpha = intercept
    
    # Generate Regression Line Points (for plotting the line)
    x_range = np.linspace(market_series.min(), market_series.max(), 100)
    y_pred = alpha + beta * x_range
    
    # 1. Scatter Plot (Actual Data)
    fig.add_trace(go.Scatter(
        x=market_series,
        y=sector_series,
        mode='markers',
        name=f"{sector} Data",
        marker=dict(color=color, opacity=0.4, size=5),
        showlegend=False
    ), row=1, col=col_idx)
    
    # 2. Regression Line
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_pred,
        mode='lines',
        name=f"{sector} Fit",
        line=dict(color='white', width=2, dash='solid'),
        showlegend=False
    ), row=1, col=col_idx)
    
    # 3. Annotation (The Equation)
    # Determine correct axis reference names for annotation positioning
    # (Plotly names them x, x2, x3... y, y2, y3...)
    axis_suffix = "" if col_idx == 1 else str(col_idx)
    xref = f"x{axis_suffix} domain"
    yref = f"y{axis_suffix} domain"
    
    equation_text = (
        f"<b>{sector}</b><br>"
        f"y = {alpha:.5f} + {beta:.2f}x<br>"
        f"----------------<br>"
        f"<b>Beta: {beta:.2f}</b><br>"
        f"Alpha: {alpha:.5f}"
    )
    
    # Place annotation inside the plot (top-left corner)
    fig.add_annotation(
        x=0.05, y=0.95,
        xref=xref, yref=yref,
        text=equation_text,
        showarrow=False,
        align="left",
        font=dict(color="white", size=12, family="monospace"),
        bgcolor="rgba(0,0,0,0.6)", # Semi-transparent background
        bordercolor=color,
        borderwidth=1
    )

# --- 4. Layout & Styling ---
off_white = "#e0e0e0"

fig.update_layout(
    title=dict(
        text="<b>CAPM Regression Analysis</b><br><sup>Sector Returns vs. SPY (2024-Present) | Isolating Alpha & Beta</sup>",
        font=dict(color=off_white, size=22),
        x=0.5, xanchor='center'
    ),
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=500,
    width=1200,
    margin=dict(t=100, b=50, l=60),
    showlegend=False
)

# Update axis styling (Zero lines help visualize Alpha/Beta)
fig.update_xaxes(
    showgrid=True, gridcolor='rgba(255,255,255,0.1)',
    zeroline=True, zerolinecolor='rgba(255,255,255,0.3)'
)
fig.update_yaxes(
    showgrid=True, gridcolor='rgba(255,255,255,0.1)',
    zeroline=True, zerolinecolor='rgba(255,255,255,0.3)'
)

fig.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- 1. Configuration ---
sector_map = {
    'Tech': ['AAPL', 'MSFT', 'AVGO'],
    'Healthcare': ['UNH', 'JNJ', 'AMGN'],
    'Consumer Staples': ['WMT', 'COST', 'PG']
}

sector_colors = {
    'Tech': '#1f77b4',       # Blue
    'Healthcare': '#2ca02c', # Green
    'Consumer Staples': '#d62728', # Red
    'Market (SPY)': '#ffffff' # White
}

# Add SPY to the loading list
all_tickers = [t for tickers in sector_map.values() for t in tickers] + ['SPY']

# --- 2. Data Loading ---
df_all = pd.DataFrame()

print("Reading CSV files...")
for ticker in all_tickers:
    filename = f"{ticker}.csv"
    try:
        df = pd.read_csv(filename)
        df.columns = [c.lower() for c in df.columns]
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        series = df['close'].rename(ticker).copy()
        
        if df_all.empty:
            df_all = pd.DataFrame(series)
        else:
            df_all = df_all.join(series, how='outer')
            
    except Exception as e:
        print(f"Skipping {ticker}: {e}")

# Filter 2024 - Present
df_all = df_all[df_all.index >= '2024-01-01'].copy()
df_all.ffill(inplace=True)
df_all.dropna(inplace=True)

if df_all.empty:
    raise ValueError("No data found.")

dates = df_all.index

# --- 3. Create Portfolios ---
df_norm = (df_all / df_all.iloc[0]) * 100
df_plot = pd.DataFrame(index=dates)

# Sector Portfolios
for sector, tickers in sector_map.items():
    df_plot[sector] = df_norm[tickers].mean(axis=1)

# Market Benchmark
if 'SPY' in df_norm.columns:
    df_plot['Market (SPY)'] = df_norm['SPY']

# Calculate Returns for Labels
df_pct = ((df_plot / 100) - 1) * 100

# --- 4. Plotly Construction ---
fig = go.Figure()

# Helper for style
def get_style(name):
    color = sector_colors.get(name, '#ffffff')
    # Reduced thickness to 2.5, all solid lines
    return dict(color=color, width=2.5, dash='solid')

# Add Initial Traces (Static)
for name in df_plot.columns:
    # 1. Line Trace
    fig.add_trace(go.Scatter(
        x=[dates[0]], 
        y=[df_plot[name].iloc[0]],
        mode='lines',
        name=name,
        line=get_style(name),
        legendgroup=name
    ))
    
    # 2. Label Trace
    fig.add_trace(go.Scatter(
        x=[dates[0]], 
        y=[df_plot[name].iloc[0]],
        mode='markers+text',
        name=name,
        marker=dict(color=get_style(name)['color'], size=8),
        text=[f"0.0%"],
        textposition="middle right",
        textfont=dict(color="white", size=14, family="monospace", weight="bold"),
        showlegend=False,
        legendgroup=name
    ))

# --- 5. Animation Frames ---
num_frames = 90
step = max(1, len(dates) // num_frames)
indices = list(range(1, len(dates), step))
if indices[-1] != len(dates)-1:
    indices.append(len(dates)-1)

frames = []
for idx in indices:
    frame_data = []
    current_date = dates[idx]
    
    for name in df_plot.columns:
        # Update Line
        frame_data.append(go.Scatter(
            x=dates[:idx], 
            y=df_plot[name][:idx],
            mode='lines'
        ))
        
        # Update Label
        current_val = df_plot[name].iloc[idx]
        current_ret = df_pct[name].iloc[idx]
        label_text = f" {name}: {current_ret:+.1f}%"
        
        frame_data.append(go.Scatter(
            x=[current_date],
            y=[current_val],
            mode='markers+text',
            text=[label_text]
        ))
        
    frames.append(go.Frame(data=frame_data, name=str(idx)))

fig.frames = frames

# --- 6. Layout & Annotations ---
off_white = "#e0e0e0"
off_black = "#222222"

fig.update_layout(
    title=dict(
        text="<b>Systematic Risk Analysis</b><br><sup>Sector Performance vs. Broad Market (SPY)</sup>",
        font=dict(color=off_white, size=22),
        x=0.05
    ),
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=700,
    width=1200,
    margin=dict(t=100, b=150, r=50),
    
    showlegend=True,
    legend=dict(
        orientation='h',
        x=0.5, xanchor='center',
        y=-0.1, yanchor='top',
        font=dict(color=off_white, size=14)
    ),
    
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.25, 'xanchor': 'center',
        'font': {'color': off_black},
        'bgcolor': off_white,
        'buttons': [{'label': '▶ Play Market Risk Analysis', 'method': 'animate',
                     'args': [None, {'frame': {'duration': 30, 'redraw': False}, 'fromcurrent': True}]}]
    }]
)

axis_style = dict(
    showgrid=True,
    gridcolor='rgba(255,255,255,0.1)',
    tickfont=dict(color=off_white),
    linecolor=off_white,
    zeroline=False,
    title_font=dict(color=off_white)
)

fig.update_xaxes(axis_style)
fig.update_yaxes(axis_style)

# Extend X-axis
end_date_buffer = dates[-1] + pd.Timedelta(days=180)
fig.update_xaxes(range=[dates[0], end_date_buffer], title_text="Date")

# Y-axis
y_min = df_plot.min().min() * 0.95
y_max = df_plot.max().max() * 1.05
fig.update_yaxes(range=[y_min, y_max], title_text="Wealth Index (Start=100)")

fig.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. Data Generation (2025 Bear Market Simulation) ---
np.random.seed(42)
dates = pd.date_range(start='2025-01-01', end='2025-12-31', freq='B')
n = len(dates)

# Market: Target ~-30% annual return (Bear Market / Recession)
target_annual_return = -0.30
trading_days = 252
# Calculate daily geometric return for -30% annualized
m_mu = (1 + target_annual_return)**(1/trading_days) - 1
m_vol = 0.012   # keep same vol as before for one year

market_rets = np.random.normal(m_mu, m_vol, n)

# Target Sharpe (Annual 2.5 -> Daily 2.5 / sqrt(252))
target_sharpe = 2.5
daily_target = target_sharpe / np.sqrt(252)

# --- LEFT PORTFOLIO: High Beta (1.5), *NO Alpha* ---
# This portfolio should lose more than the market in a bear market.
beta_left_target = 1.5

# Set alpha = 0, so expected mean is just beta * market_mu
# Mean_p = beta * market_mu (which is deeply negative)
# Std_p = sqrt((beta * market_vol)^2 + sigma_noise^2)
noise_vol = 0.002
rets_left = (beta_left_target * market_rets) + np.random.normal(0, noise_vol, n)

# --- RIGHT PORTFOLIO: Beta 0.0, Sharpe 2.5 (Market Neutral Alpha) ---
raw_noise = np.random.normal(0, 1, n)
p2_std = 0.01 
p2_mean = daily_target * p2_std
rets_right = p2_mean + (raw_noise - np.mean(raw_noise)) / np.std(raw_noise) * p2_std

# Helper to calculate realized metrics
def get_realized_sharpe(rets):
    return (np.mean(rets) / np.std(rets)) * np.sqrt(252)

def capm_beta_alpha(port_rets, market_rets):
    # Regression: port_rets = alpha + beta * market_rets + residuals
    X = market_rets
    y = port_rets
    X = np.vstack([np.ones_like(X), X]).T  # add intercept for alpha
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    alpha = beta_hat[0]
    beta = beta_hat[1]
    return beta, alpha

s_left = get_realized_sharpe(rets_left)
s_right = get_realized_sharpe(rets_right)
beta_left, alpha_left = capm_beta_alpha(rets_left, market_rets)
beta_right, alpha_right = capm_beta_alpha(rets_right, market_rets)

# Convert alpha (daily) to annualized percent return for reporting
alpha_left_ann_pct = ( (1 + alpha_left) ** 252 - 1 ) * 100
alpha_right_ann_pct = ( (1 + alpha_right) ** 252 - 1 ) * 100

# Cumulative Wealth (Starting at 100)
df = pd.DataFrame({
    'Date': dates,
    'Market': np.cumprod(1 + market_rets) * 100,
    'Port_Left': np.cumprod(1 + rets_left) * 100,
    'Port_Right': np.cumprod(1 + rets_right) * 100
})

# --- 2. Plotly Construction ---
title_l = (
    f"Bear Market vs. High Beta Portfolio | Previous Sharpe: 2.65"
)
title_r = (
    f"Bear Market vs. Market Neutral Alpha | Previous Sharpe: 2.5"
)

fig = make_subplots(
    rows=1, cols=2, 
    subplot_titles=(title_l, title_r),
    horizontal_spacing=0.1
)

# Colors and Styling
off_white = "#e0e0e0"
market_color = "#888888"
port_color = "#00d1ff"
play_blue = "#00bfff"
play_darkblue = "#143c99"  # darker blue for play button text

# Plot initialization
for col_idx, (market_col, port_col) in enumerate([('Market', 'Port_Left'), ('Market', 'Port_Right')], start=1):
    show_leg = (col_idx == 2)  # legend on the 2nd subplot
    fig.add_trace(go.Scatter(
        x=[df['Date'].iloc[0]], y=[df[market_col].iloc[0]],
        line=dict(color=market_color, width=1.5),
        name="Market",
        showlegend=show_leg,
        legendgroup='grp1',
        mode='lines'
    ), row=1, col=col_idx)
    fig.add_trace(go.Scatter(
        x=[df['Date'].iloc[0]], y=[df[port_col].iloc[0]],
        line=dict(color=port_color, width=2.5),
        name="Portfolio",
        showlegend=show_leg,
        legendgroup='grp1',
        mode='lines'
    ), row=1, col=col_idx)

# --- 3. Animation Frames ---
num_frames = 60
indices = np.linspace(1, n-1, num_frames, dtype=int)
frames = []
for idx in indices:
    frames.append(go.Frame(
        data=[
            go.Scatter(x=df['Date'][:idx], y=df['Market'][:idx], mode='lines'),
            go.Scatter(x=df['Date'][:idx], y=df['Port_Left'][:idx], mode='lines'),
            go.Scatter(x=df['Date'][:idx], y=df['Market'][:idx], mode='lines'),
            go.Scatter(x=df['Date'][:idx], y=df['Port_Right'][:idx], mode='lines')
        ],
        name=str(idx)
    ))
fig.frames = frames

# --- 4. Layout & Transparency ---
axis_style = dict(
    showgrid=True,
    gridcolor='rgba(255,255,255,0.1)',
    tickfont=dict(color=off_white),
    linecolor=off_white,
    zeroline=False,
    title_font=dict(color=off_white)
)

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=600,
    margin=dict(t=120, b=120),
    showlegend=True,
    legend=dict(
        orientation='v',
        # Place the legend at the bottom right of the 2nd subplot
        # The 2nd subplot's x-domain is the right half, usually 0.5 to 1.0, so x=1.0 is bottom right
        x=0.995, xanchor='right', y=0.08, yanchor='bottom',
        font=dict(color=off_white, size=15),
        bgcolor='rgba(0,0,0,0.55)',
        bordercolor='rgba(0,0,0,0)'
    ),
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.18, 'xanchor': 'center',
        'font': {'color': play_darkblue},
        'buttons': [{'label': '▶ Play 2025 Bear Market Simulation', 'method': 'animate',
                     'args': [None, {'frame': {'duration': 30, 'redraw': False}, 'fromcurrent': True}]}]
    }]
)
# Set all axes to off-white
fig.update_xaxes(axis_style)
fig.update_yaxes(axis_style)

# Suitable fixed y-axes for drawdown (bear market)
fig.update_xaxes(
    range=[pd.Timestamp('2025-01-01'), pd.Timestamp('2025-12-31')], 
    row=1, col=1
)
fig.update_yaxes(
    range=[50, 120],
    title_text="Wealth Index (Initial=100)", 
    row=1, col=1
)

fig.update_xaxes(
    range=[pd.Timestamp('2025-01-01'), pd.Timestamp('2025-12-31')], 
    row=1, col=2
)
fig.update_yaxes(
    range=[65, 155],
    row=1, col=2
)

# Style titles
for annotation in fig['layout']['annotations']:
    annotation['font'] = dict(color=off_white, size=13)

fig.show()
```

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- 1. Data Setup ---
sector_map = {
    'Tech': ['AAPL', 'MSFT', 'AVGO'],
    'Healthcare': ['UNH', 'JNJ', 'AMGN'],
    'Consumer Staples': ['WMT', 'COST', 'PG']
}

sector_colors = {
    'Tech': '#1f77b4',       # Blue
    'Healthcare': '#2ca02c', # Green
    'Consumer Staples': '#d62728' # Red
}

# Combine lists including SPY
all_tickers = [t for tickers in sector_map.values() for t in tickers] + ['SPY']

# --- 2. Data Loading ---
df_all = pd.DataFrame()

print("Reading CSV files...")
for ticker in all_tickers:
    filename = f"{ticker}.csv"
    try:
        df = pd.read_csv(filename)
        df.columns = [c.lower() for c in df.columns]
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        series = df['close'].rename(ticker).copy()
        
        if df_all.empty:
            df_all = pd.DataFrame(series)
        else:
            df_all = df_all.join(series, how='outer')
            
    except Exception as e:
        print(f"Skipping {ticker}: {e}")

# Filter 2024 - Present
df_all = df_all[df_all.index >= '2024-01-01'].copy()
df_all.ffill(inplace=True)
df_all.dropna(inplace=True)

if df_all.empty:
    raise ValueError("No data found.")

# Calculate Daily Returns
df_rets = df_all.pct_change()

# --- 3. Build Indices & Calculate Rolling Beta ---
sector_rets = pd.DataFrame(index=df_rets.index)
window_size = 30 # 30 Days

# A. Create Sector Indices (Equal Weight)
for sector, ticks in sector_map.items():
    valid_ticks = [t for t in ticks if t in df_rets.columns]
    sector_rets[sector] = df_rets[valid_ticks].mean(axis=1)

# B. Market Index
market_rets = df_rets['SPY']

# C. Calculate Rolling Beta
# Beta = Cov(Sector, Market) / Var(Market)
df_beta = pd.DataFrame(index=sector_rets.index)
rolling_market_var = market_rets.rolling(window=window_size).var()

for sector in sector_map.keys():
    rolling_cov = sector_rets[sector].rolling(window=window_size).cov(market_rets)
    df_beta[sector] = rolling_cov / rolling_market_var

# Drop initial NaNs from window
df_beta.dropna(inplace=True)
dates_plot = df_beta.index

# --- 4. Plotly Construction ---
fig = go.Figure()

# Add Initial Traces (Static)
for sector in sector_map.keys():
    color = sector_colors[sector]
    
    # Line
    fig.add_trace(go.Scatter(
        x=[dates_plot[0]], 
        y=[df_beta[sector].iloc[0]],
        mode='lines',
        name=sector,
        line=dict(color=color, width=2),
        legendgroup=sector
    ))
    
    # Floating Label
    fig.add_trace(go.Scatter(
        x=[dates_plot[0]], 
        y=[df_beta[sector].iloc[0]],
        mode='markers+text',
        name=sector,
        marker=dict(color=color, size=6),
        text=[f"{df_beta[sector].iloc[0]:.2f}"],
        textposition="middle right",
        textfont=dict(color="white", size=13, family="monospace"),
        showlegend=False,
        legendgroup=sector
    ))

# --- 5. Animation Frames ---
num_frames = 90
step = max(1, len(dates_plot) // num_frames)
indices = list(range(1, len(dates_plot), step))
if indices[-1] != len(dates_plot)-1:
    indices.append(len(dates_plot)-1)

frames = []
for idx in indices:
    frame_data = []
    current_date = dates_plot[idx]
    
    for sector in sector_map.keys():
        # Update Line
        frame_data.append(go.Scatter(
            x=dates_plot[:idx], 
            y=df_beta[sector][:idx],
            mode='lines'
        ))
        
        # Update Label
        current_val = df_beta[sector].iloc[idx]
        frame_data.append(go.Scatter(
            x=[current_date],
            y=[current_val],
            mode='markers+text',
            text=[f" {sector}: {current_val:.2f}"]
        ))
        
    frames.append(go.Frame(data=frame_data, name=str(idx)))

fig.frames = frames

# --- 6. Layout & Styling ---
off_white = "#e0e0e0"
off_black = "#222222"

# Vertical Red Lines (Drawdown Markers)
drawdown_lines = [
    dict(
        type="line",
        x0="2025-03-20", y0=0, x1="2025-03-20", y1=1,
        xref="x", yref="paper",
        line=dict(color="red", width=1.5, dash="dash")
    ),
    dict(
        type="line",
        x0="2025-04-01", y0=0, x1="2025-04-01", y1=1,
        xref="x", yref="paper",
        line=dict(color="red", width=1.5, dash="dash")
    )
]

fig.update_layout(
    title=dict(
        text="<b>Rolling Sector Beta (30-Day)</b><br><sup>Sensitivity to SPY Benchmark Over Time</sup>",
        font=dict(color=off_white, size=22),
        x=0.05
    ),
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=600,
    width=1200,
    margin=dict(t=100, b=150, r=80),
    shapes=drawdown_lines,
    showlegend=True,
    legend=dict(
        orientation='h',
        x=0.5, xanchor='center',
        y=-0.15, yanchor='top',
        font=dict(color=off_white, size=14)
    ),
    updatemenus=[{
        'type': 'buttons',
        'x': 0.5, 'y': -0.25, 'xanchor': 'center',
        'font': {'color': off_black},
        'bgcolor': off_white,
        'buttons': [{'label': '▶ Play Beta Dynamics', 'method': 'animate',
                     'args': [None, {'frame': {'duration': 30, 'redraw': False}, 'fromcurrent': True}]}]
    }]
)

# Axis Styling
axis_style = dict(
    showgrid=True,
    gridcolor='rgba(255,255,255,0.1)',
    tickfont=dict(color=off_white),
    linecolor=off_white,
    zeroline=False,
    title_font=dict(color=off_white)
)
fig.update_xaxes(axis_style)
fig.update_yaxes(axis_style)

# --- KEY UPDATE: Extend X-axis to Aug 2026 ---
target_end_date = pd.Timestamp('2026-08-31')
fig.update_xaxes(range=[dates_plot[0], target_end_date], title_text="Date")

# Y-Axis
fig.update_yaxes(title_text="Beta (Sensitivity to SPY)")

fig.show()
```
