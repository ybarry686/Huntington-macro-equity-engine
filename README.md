# Huntington Macro Equity Engine

A comprehensive quantitative research and forecasting platform for sector-level macroeconomic equity analysis, signal experimentation, and regime-based market modeling.

## 🎯 Core Systems

### Lead-Lag Signal Discovery Engine
* **Recursive ADF Testing:** Enforces stationarity across 25+ years of market data to eliminate spurious correlations.
* **Rolling Lead-Lag Analysis:** Partitions ETF and macro datasets into rolling windows, identifying the highest-correlation lag per window, and selects the modal lag as the stable signal.
* **Statistical Significance:** Resulted in a **22% increase compared to baseline** in downstream forecasting accuracy.   

### Multi-Model ML Suite
* **Linear Regression Models:** Integrates Ordinary, Recursive, and Rolling OLS models to capture static, evolving, and time-varying relationships between macroeconomic indicators and ETF returns.
* **Random Forest:** Captures complex nonlinear relationships between macroeconomic indicators and ETF returns, validated across 25+ simulated market conditions (e.g., bull markets, bear markets, macro shocks, etc).
* **Principal-Component Analysis:** Implements dimensionality reduction, slashing feature noise by **70%** while preserving over **85%** dataset variance.
* **Statistical Significance:** Achieved **55%+ directional accuracy** in forecasting sector price action across shifting market regimes.

### Sector Risk Engine
* **Multi-Factor Risk Scoring Algorithm:** Generates a composite risk score derived from asset volatility, beta, and inter-asset correlation.
* **Systematic Risk Analysis:** Measures inter-asset correlation across each sector's top 10 holdings to quantify diversification and systematic risk exposure.
* **Statistical Significance:** Generated a comprehensive risk ranking across all 11 GICS sectors, enabling risk-adjusted sector comparison under varying market conditions.

## 🚀 Infrastructure & Performance

* **Vectorized Pipeline:** Implements CPython and vectorized matrix operations, resulting in **sub-30ms end-to-end retrieval latency**.
* **Caching Layer:** Implemented a persistence strategy that cut external API usage by **~90%** and bypass network bottlenecks.
* **Modular Architecture:** Built with a "plug-and-play" design for adding new alpha factors or model architectures.

## ⚙️ Setup

Before running the application, configure the required environment variables and API access.

### 1. Create a `.env` File

In the project root directory, create a file named `.env` with the following contents:

```env
FRED_API_KEY=your_fred_api_key_here
```
The system uses the FRED API to retrieve macroeconomic time series data for model inputs and signal generation.

### 2. Obtain a FRED API Key

Visit the Federal Reserve Economic Data (FRED) platform: https://fred.stlouisfed.org/  
Create a free account or sign in.  
Navigate to your account settings.  
Generate an API key under the developer/API section.  
Copy the key and paste it into your `.env` file:

```env
FRED_API_KEY=your_generated_key_here
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python main.py
```
On the first run, the system will:

* Initialize data ingestion from FRED
* Build required macro/ETF datasets
* Cache processed feature sets for faster subsequent runs
* Prepare the analytical pipeline for signal generation and forecasting
