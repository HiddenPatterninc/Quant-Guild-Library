# Quant Guild Library: Jupyter Notebook Index

106 notebooks in total: 56 in `2025 Video Lectures/` and 50 in `2026 Video Lectures/`. Each lesson folder is numbered after its video. Summaries come from each notebook's titles, section headings and imports, so they describe the topic and the method rather than every line of code.

---

## 2025 Video Lectures

| # | Notebook | What it does |
|---|----------|--------------|
| 1 | Inverse Transform Method.ipynb | Generates random variables from uniform draws using the inverse CDF. Worked examples are a die roll (discrete) and an exponential distribution (continuous). |
| 2 | Control Variates for Variance Reduction.ipynb | Cuts Monte Carlo variance with control variates. Prices a European option under arithmetic Brownian motion, then discusses rough models and neural-net approximations. |
| 3 | Problems with Statistics in Trading.ipynb | Shows how parameter estimation error, non-stationarity and "edge" can mislead a trader. Includes simulated equity curves and interactive widgets. |
| 4 | When to Stop Trading a Profitable Strategy.ipynb | Simulates a trading edge that decays over time and asks when to shut a strategy off. Introduces non-stationary dynamics. |
| 6 | Black-ScholesTrading.ipynb | Uses Black-Scholes prices to trade options. Computes P/L edge versus the true dynamics, simulates equity curves and covers the pitfalls of trading model prices (uses `qfin`). |
| 7 | Martingale Volatility Trading.ipynb | Martingale betting in roulette and why losing streaks are devastating. Then volatility mean reversion with an Ornstein-Uhlenbeck process on real data (`yfinance`). |
| 9 | Delta Hedging Trading Strategy.ipynb | Delta-hedges a short call under geometric Brownian motion. Monte Carlo shows P/L converging to the theoretical price, plus an example of trading a view on volatility. |
| 11 | Managing Option Portfolios.ipynb | Linear approximation and Taylor ideas applied to the option Greeks. Has interactive Greek explorers and hedging strategies for option portfolios. |
| 12 | Equity Trading and Tariffs.ipynb | Models the effect of tariffs on a stock with a DCF. Covers a baseline DCF, tariffs with re-optimization, price repricing and recovery. |
| 13 | Can AI Learn Black-Scholes.ipynb | Trains a PyTorch neural network to learn Black-Scholes option prices. Compares it with the exact formula and a Monte Carlo approximation. |
| 14 | Quant Investing Strategies for Beginners.ipynb | Intro to risk types and how correlation drives diversification. |
| 15a | alpaca.ipynb | AI trading bot: connects to the Alpaca brokerage API (`alpaca_trade_api`). |
| 15b | openai.ipynb | AI trading bot: combines the OpenAI API with Alpaca to generate trading decisions. |
| 15c | checklist.ipynb | Short TODO/checklist of the steps for building the AI trading bot. |
| 16 | neural_network_basics.ipynb | Shows how hard stock returns are to predict. Compares linear regression with a PyTorch neural network on a chronological train/test split. |
| 17 | pca_stock_returns_analysis.ipynb | Principal Component Analysis of stock returns. Covers correlation structure and PCA by both SVD and covariance eigendecomposition. |
| 18 | pricing.ipynb | Why pricing matters. Coin-flip and dice games played at fair, below-fair and above-fair prices, with simulated outcomes. |
| 19 | montecarlo_blackshcoles.ipynb | Moves from analytical expectation (discrete and continuous) to Monte Carlo approximation, then applies it to Black-Scholes pricing. Interactive. |
| 20 | portfolio_optimization.ipynb | Sharpe-ratio portfolio optimization on a portfolio of correlated games. Argues that optimization overfits, with a balanced view. |
| 21 | expected_returns_dont_exist.ipynb | Questions "expected returns" in Markowitz, CAPM and VaR. Covers higher moments, simulated expectations and Kalman filtering (`pykalman`). |
| 22 | how_to_trade.ipynb | Game-theory simulation of naive vs quant traders in static and dynamic games, with same or different models. |
| 23 | How to Trade Option Implied Volatility.ipynb | Volatility filtering, smoothing and forecasting. Compares popular volatility models and realized-vol measures. |
| 24 | Trading_with_Violated_Model_Assumptions.ipynb | What happens when model assumptions fail: fat tails (excess kurtosis), non-IID correlation and non-stationarity. |
| 25 | fbm.ipynb | Simulates fractional Brownian motion with the Davies-Harte method (time grid, covariance, FFT) and discusses financial uses. |
| 26 | is_quant_trading_gambling.ipynb | Compares edge in roulette (chance), poker and trading (incomplete information). Includes edge-accumulation sample paths. |
| 28 | gamblers_ruin_in_quant_trading.ipynb | Gambler's ruin analytically and by simulation, applied to a trader's expected value and a trading system. |
| 29 | itos_lemma.ipynb | Visual walk-through of Itô's lemma against ordinary derivatives and Taylor series, with simulation to test the approximation. |
| 30 | the_implied_volatility_surface.ipynb | Black-Scholes pricing, solving for implied vol, and building and exploring the implied volatility surface (Plotly 3D). |
| 31 | ito_integration.ipynb | Riemann integration versus Itô integration. Covers random variables, Brownian motion and interactive visuals. |
| 32 | how_trading_desks_price_exotics.ipynb | General recipe for exotic payoffs. Covers why Black-Scholes is inconsistent and moves to richer models. |
| 33 | why_monte_carlo_simulation_works.ipynb | Shows the Law of Large Numbers and related ideas that justify Monte Carlo simulation. |
| 34 | how_to_trade_with_an_edge.ipynb | Defines "edge" with roulette and other examples, then shows how edge plays out in trading. |
| 35 | what_does_ai_actually_learn.ipynb | Frames outcomes as random variables and predictions as expectations. Shows with PyTorch what models actually learn. |
| 36 | how_to_trade_with_the_kelly_criterion.ipynb | Bet sizing with the Kelly criterion, from the expected value of a trading system through to simulated growth. |
| 37 | solving_stochastic_differential_equations.ipynb | Differential equations, then SDEs for finance. Includes analytic solutions and numerical simulation. |
| 38 | finite_differences_option_pricing.ipynb | Prices options by solving the Black-Scholes PDE with finite-difference schemes. |
| 39 | heston_fft.ipynb | Heston stochastic-volatility model priced with Fast Fourier Transforms. |
| 40 | quant_explains_retail_institutional_trading.ipynb | Simulation-based comparison of retail and institutional trading. |
| 42 | quant_on_trading_investing_market.ipynb | Risk and return, and the differences between trading and investing, with simulations. |
| 43 | how_to_trade_volatility_crush_option_strategy.ipynb | Models and trades the post-event implied-volatility crush. |
| 44 | time_series_analysis_for_quant_finance.ipynb | Time-series fundamentals for finance (stationarity, autocorrelation, models) on simulated data. |
| 46 | is_trading_luck_or_skill.ipynb | Separates luck from skill in trading using gambling, poker and edge simulations. |
| 47 | arch_garch.ipynb | Volatility clustering with ARCH and GARCH models: fitting, selecting parameters and forecasting (`statsmodels`). |
| 48 | trading_performance.ipynb | Shows why common trading metrics mislead unless the underlying assumptions hold. |
| 49 | markov_chains.ipynb | Markov chains for finance: transition matrices, stationary distributions and simulation. |
| 50 | poker_trading.ipynb | Why poker skills transfer to trading. Simulates poker and trading edge. |
| 51 | hidden_markov_models.ipynb | Hidden Markov Models for market regimes: filtering, decoding and estimation. |
| 52 | proof.ipynb | Poker simulation (`deuces`) to argue trading skill cannot be taught but can be learned. |
| 54 | quant_disc.ipynb | Quant vs discretionary trading, with extensive simulations and machine-learning comparisons. |
| 56 | myth_buster.ipynb | Tests three common trading myths with simulation and math. |
| 57 | banks_are_casinos.ipynb | Simulates why banks resemble casinos (house edge and the law of large numbers). |
| 58 | why_models_break.ipynb | Why quant models fail, with simulated breakdowns of assumptions. |
| 59 | brownian_motion.ipynb | Brownian motion in depth: properties, simulation and visualization. |
| 60 | is_trading_gambling.ipynb | Math-based argument that trading is not gambling, using edge and expectation simulations. |
| 61 | clt.ipynb | Central Limit Theorem for finance: simulations, convergence and where it breaks down. |
| 63 | nn_quant.ipynb | Neural networks for quant finance (TensorFlow and scikit-learn). |
| 65 | nlp.ipynb | NLP for quant trading: text processing (NLTK, spaCy), word clouds and sentiment-style features. |
| 67 | fk_bs.ipynb | Derives Black-Scholes via the Feynman-Kac link between PDEs and expectations. |
| 69 | htmam.ipynb | Algorithmic market-making: quoting, inventory and simulation. |
| 71 | mp.ipynb | The Markov property in quant trading, with simulation. |
| 73 | htpowmc.ipynb | Prices options with Monte Carlo simulation. |
| 75 | bilp.ipynb | Proper backtesting taught through poker simulation (`deuces`). |
| 77 | pt_wmsfl.ipynb | Profitable versus tradable: why backtested strategies fail live (costs, slippage, assumptions). |

---

## 2026 Video Lectures

| # | Notebook | What it does |
|---|----------|--------------|
| 78 | capm.ipynb | Explains alpha with CAPM (skill versus market luck) in about three minutes. |
| 80 | t5_qfp.ipynb | Walks through the five papers that built modern quant finance, with code demos. |
| 81 | e_ft.ipynb | Ergodicity: why time averages differ from ensemble averages and why most traders lose. |
| 82 | ppfqf.ipynb | Poisson processes for jump and arrival modeling in finance. |
| 83 | rnop.ipynb | Risk-neutral option pricing explained and demonstrated. |
| 85 | volterra.ipynb | Discretizes and simulates Volterra processes (rough-volatility style models). |
| 87 | wva.ipynb | Quant trading in three minutes (edge, variance and sizing). |
| 88 | ff.ipynb | Portfolio management with factor models (Fama-French style). |
| 89 | bsiv.ipynb | Black-Scholes implied volatility in three minutes, solving for IV numerically. |
| 92 | kffqf.ipynb | The Kalman filter for quant finance (state-space modeling, filtering). |
| 93 | wmtf.ipynb | Non-stationarity and why market timing fails. |
| 94 | hpqf.ipynb | Hawkes (self-exciting) processes for clustered events. |
| 95 | ktsmr.ipynb | Mean-reversion trading built on Kalman filters. |
| 96 | wnpt.ipynb | Tests whether alpha exists using regression on factor data (`pandas_datareader`, `statsmodels`). |
| 97 | qebp.ipynb | Three backtesting pitfalls that ruin strategies. |
| 100 | blmpfm.ipynb | Black-Litterman versus mean-variance portfolio optimization. |
| 101 | qas3m.ipynb | The limits of the Sharpe ratio and better alternatives. |
| 103 | qfi3m.ipynb | Quantitative finance overview in three minutes. |
| 106 | httcc.ipynb | How to trade the covered call, with payoff and risk analysis. |
| 107 | htobs.ipynb | Trading options with the Black-Scholes model. |
| 111 | retail_problem.ipynb | The mathematical problem with retail trading. |
| 112 | wyts.ipynb | Why your trading strategy sucks. |
| 113 | httcc.ipynb | How to trade the cash-secured put. Connects to Interactive Brokers (`ibapi`) for live data. |
| 115 | haqep.ipynb | How a quant engineers a portfolio and avoids being a "closet beta" manager. |
| 116 | haqep.ipynb | Why quants dislike U.S. equities, using a diversified equity portfolio as the example. |
| 118 | no_you_dont_need_to_backtest.ipynb | Argues you don't need to backtest every strategy (Interactive Brokers and `statsmodels`). |
| 119 | riveot.ipynb | Volatility risk premium: realized versus implied volatility for the S&P. |
| 120 | dvh.ipynb | The mathematics of diversification and hedging. |
| 121 | sp.ipynb | Shows stock picking is worse than gambling, with simulation and ML. |
| 122a | sp.ipynb.ipynb | Debunks the "day trader" scam. The filename has a doubled `.ipynb` extension. |
| 122b | sp_clean_no_outputs.ipynb | Same day-trader notebook with outputs cleared. |
| 123 | mln.ipynb | How a quant would allocate $1,000,000. |
| 124 | wtsds.ipynb | Why trading strategies don't need to be secret. |
| 125 | htbats.ipynb | Calculates and uses CAGR as a performance test. |
| 126 | bse.ipynb | Tail-risk and black-swan modeling (`arch`, `scipy`). |
| 127 | tooi.ipynb | The ultimate guide to quant portfolio management. |
| 128 | alpha_beta.ipynb | Calculates portfolio alpha and beta (Python with Interactive Brokers). |
| 135 | abmgbm.ipynb | Arithmetic versus geometric Brownian motion masterclass. |
| 136 | vol_drag.ipynb | How volatility drag destroys and creates wealth. |
| 138 | crash.ipynb | Ways to protect a stock portfolio against a crash (hedging and options). |
| 140 | sulp.ipynb | A "Situational Unawareness LP" style fund analysis, built on dataclasses. |
| 141 | eft.ipynb | Why emotion cannot be removed from trading. |
| 142 | iat.ipynb | Argues everything is technical analysis. |

---

## Notes
- Several different lectures reuse the same filename (`httcc.ipynb` in 106 and 113, `haqep.ipynb` in 115 and 116, `sp.ipynb` in 121 and 122), so go by the folder name, not the file name.
- A few summaries for short, abbreviated 2026 notebooks (for example 140 and 141) are inferred mostly from titles, so open the notebook to confirm the details.
- Lecture 15 (AI trading bot) is the only one with a nested project folder (`AI_Trading_Bot-main/`). Lectures 16, 17, 21, 23 and 24 are also nested in `*-main/` folders.
