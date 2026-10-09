# AI Stock Market Chatbot

This investment assistant is aimed at beginner investors. It:

1. Groups NASDAQ-100 companies by risk.
2. Predicts the six-month price movement of the safest companies.
3. Answers questions in plain language. A language model writes SQL queries against the project's data warehouse and turns the results into answers.

It was a group project for a team of four at Breda University of Applied Sciences (Applied Data Science & AI, Year 1), in May and June 2025. It's a case study for a simulated fintech client played by staff.

## The problem

Starting to invest is confusing: beginners don't know which companies are safe to buy or how prices are likely to move. The goal was a chatbot that answers questions such as:

- "Which stocks are safest to invest in?"
- "Will this company's price go up in the next six months?"

The answers come from data, not guesswork.

## How it works

```
PostgreSQL warehouse (NASDAQ-100 daily prices, NASDAQ-100 index, macro indicators)
  -> K-Means clusters companies by risk -> the "safest" cluster
  -> XGBoost predicts each safe company's 6-month price movement class
  -> predictions are written back to the warehouse
  -> chatbot (LangGraph): pick a table -> LLM writes SQL -> run it -> LLM writes the answer
  -> Gradio chat interface (Mistral Small 22B served with Ollama)
```

1. **Data warehouse.** The SQL scripts in `sql/` build two tables:
   - A `features` table that joins daily prices with macroeconomic indicators.
   - A `stock_model_features` table that uses window functions for daily and rolling returns, volatility, moving averages, volume statistics and calendar features.
2. **Risk clustering.**
   - Each company is summarised over its last six months: mean and standard deviation of daily returns, intraday volatility, volume mean and standard deviation, and six-month return.
   - Outliers are removed with the IQR rule, the features are standardised, and K-Means groups the companies (k = 3, chosen with the elbow method).
   - The clusters are labelled *safest* (low volatility, positive returns), *solid* and *avoid*. The company closest to the centre of the safest cluster serves as the representative for training the classifier.
3. **Price movement classification.** XGBoost predicts the six-month return class:
   - **Classes:** very high (≥ 30%), high (15–30%), no change (−5% to 15%), low (−20% to −5%), very low (< −20%).
   - **Features:** technical indicators (126-day volatility, momentum and rate of change, Bollinger bandwidth, RSI-14, a 20/50-day moving-average crossover, Sharpe ratio, volume change) plus macro indicators. Volatile indicators are capped with the IQR rule.
   - **Training:** minority classes are oversampled, and a grid search over 729 settings uses stratified cross-validation on macro F1.
   - **Evaluation:** a rolling, time-based backtest is compared with an "always no change" baseline.
4. **Chatbot.** A LangGraph pipeline does four steps:
   1. It routes the question to the clustering or the prediction table.
   2. The LLM writes a SQL query.
   3. It runs the query.
   4. The LLM answers using only the query result.

   Gradio provides the chat interface.
5. **Engineering.** The code is a Poetry package with command-line scripts, pytest unit tests, Sphinx documentation and a Dockerfile.

## Results

| Component | Result |
|---|---|
| K-Means risk clustering (k = 3) | Silhouette score 0.288; 41 of 77 companies in the safest cluster |
| XGBoost, rolling backtest (3 time-based splits) | **0.613** mean accuracy, against **0.707** for always predicting "no change"; latest split 0.667 accuracy and 0.27 macro F1 |
| XGBoost, random train/test split | 0.883 accuracy and 0.87 macro F1. This is optimistic: neighbouring days share most of their six-month windows. |

The classifier did not beat the majority-class baseline in the time-based backtest. The analysis in [notebooks/model_development.ipynb](notebooks/model_development.ipynb) points to three causes:

- Outlier filtering during clustering removed most large price moves.
- The remaining classes are heavily imbalanced.
- The technical indicators carry little signal for rare tail events.

## My role

In a team of four, my parts were:

- **Data warehouse:** I wrote the SQL scripts that create and populate the PostgreSQL feature tables (`sql/`).
- **Risk clustering:** I trained the K-Means model that groups the companies by risk.
- **Price movement classification:** I trained the XGBoost model that predicts the six-month movement.
- **Packaging:** together with a teammate, I restructured the code into a modular Python package and set up Poetry.

My teammates mainly built the chatbot (LangGraph pipeline and Gradio interface), the unit tests and documentation, the exploratory analyses, and the client proposal and presentation.

## Tech stack

Python 3.9, Poetry, pandas, scikit-learn, XGBoost, imbalanced-learn, SQLAlchemy, PostgreSQL, LangChain, LangGraph, Ollama (Mistral Small 22B), Gradio, Matplotlib, seaborn, pytest, Sphinx, Docker.

## Repository layout

```
src/stock_chatbot/
  config.py                     # features, database settings (from environment variables), logging
  data_loader.py                # reads the warehouse tables
  clustering_preprocessing.py   # per-company features, outlier removal, scaling
  clustering_model.py           # K-Means
  clustering_analysis.py        # cluster labels, safest companies, results table
  classification_preprocessing.py
  model.py                      # loads the XGBoost model and scores new data
  ui/                           # LangGraph SQL pipeline, Ollama model, Gradio interface
run_clustering_cmd.py           # clustering pipeline -> results table
run_classification_cmd.py       # clustering + classification -> predictions table
run_chatbot.py                  # launches the chat interface
sql/                            # warehouse creation scripts
notebooks/model_development.ipynb
tests/                          # pytest unit tests (mocked database and LLM)
docs/                           # Sphinx documentation source
```

## Data and models

The NASDAQ-100 and macroeconomic datasets were provided by the course, and the trained model files are not included. The SQL scripts expect a PostgreSQL database with the `nasdaq_100_daily`, `nasdaq_100_index` and `macro_economic_indicators` tables. `run_classification_cmd.py` expects `models/xgb_ctsh_model.pkl` and `models/label_encoder_ctsh.pkl`, which are written by the model development notebook.

## How to run

```bash
poetry install                    # Python 3.9
cp .env.example .env              # fill in the database and Ollama settings
set -a; source .env; set +a       # export the variables (bash)

python run_clustering_cmd.py
python run_classification_cmd.py
python run_chatbot.py             # Gradio interface on http://localhost:7860

poetry run pytest                 # unit tests
```

With Docker:

```bash
docker build -t stock-chatbot .
docker run --env-file .env -p 7860:7860 stock-chatbot
```
