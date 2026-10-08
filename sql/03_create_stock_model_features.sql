-- Drop old table if it exists
DROP TABLE IF EXISTS stock_model_features;

-- Recreate with all features (excluding crude oil, gold, and FX)
CREATE TABLE stock_model_features AS
WITH daily_feat AS (
  SELECT
    d.date_value,
    d.company_prefix AS ticker,
    d.open_value,
    d.high_value,
    d.low_value,
    d.close_value,
    d.volume,
    -- Daily return & vol
    (d.close_value - d.open_value) / NULLIF(d.open_value, 0) AS return_1d,
    (d.high_value - d.low_value)   / NULLIF(d.open_value, 0) AS volatility_1d,
    -- Moving averages
    AVG(d.close_value) OVER (
      PARTITION BY d.company_prefix
      ORDER BY d.date_value
      ROWS 13 PRECEDING
    ) AS ma_14,
    AVG(d.close_value) OVER (
      PARTITION BY d.company_prefix
      ORDER BY d.date_value
      ROWS 19 PRECEDING
    ) AS ma_20,
    -- Rolling returns
    (d.close_value - LAG(d.close_value,5 ) OVER w) / NULLIF(LAG(d.close_value,5 ) OVER w,0) AS return_5d,
    (d.close_value - LAG(d.close_value,10) OVER w) / NULLIF(LAG(d.close_value,10) OVER w,0) AS return_10d,
    -- Rolling stats (20d window)
    AVG((d.close_value - d.open_value)/NULLIF(d.open_value,0)) OVER w      AS avg_daily_return,
    STDDEV((d.close_value - d.open_value)/NULLIF(d.open_value,0)) OVER w   AS std_dev_daily_return,
    AVG((d.high_value - d.low_value)/NULLIF(d.open_value,0))   OVER w      AS avg_daily_volatility,
    AVG(d.volume)                                              OVER w      AS avg_volume,
    STDDEV(d.volume)                                           OVER w      AS std_dev_volume,
    -- Short‐term vol
    STDDEV((d.close_value - d.open_value)/NULLIF(d.open_value,0)) OVER (
      PARTITION BY d.company_prefix
      ORDER BY d.date_value
      ROWS 4 PRECEDING
    ) AS volatility_5d,
    STDDEV((d.close_value - d.open_value)/NULLIF(d.open_value,0)) OVER (
      PARTITION BY d.company_prefix
      ORDER BY d.date_value
      ROWS 9 PRECEDING
    ) AS volatility_10d,
    -- Calendar
    EXTRACT(DOW FROM d.date_value) AS day_of_week,
    (d.date_value = MAX(d.date_value) OVER (PARTITION BY date_trunc('month', d.date_value))) AS is_month_end
  FROM nasdaq_100_daily d
  WINDOW w AS (
    PARTITION BY d.company_prefix
    ORDER BY d.date_value
    ROWS 19 PRECEDING
  )
)
SELECT
  df.*,

  -- Macro indicators
  mei.interest_rate_fed_funds,
  mei."10_year_treasury_yield",
  mei.stock_market_volatility_vix_index,
  mei.unemployment_rate,
  mei.inflation_cpi,
  mei.retail_sales_data_excluding_food_services,
  mei.gdp_growth,

  -- NASDAQ index benchmark
  n100.close_value AS nasdaq_index_close,
  (n100.close_value - n100.open_value) / NULLIF(n100.open_value,0) AS nasdaq_index_return_1d

FROM daily_feat df
LEFT JOIN macro_economic_indicators mei
  ON df.date_value = mei.date_value
LEFT JOIN nasdaq_100_index n100
  ON df.date_value = n100.date_value
ORDER BY df.date_value, df.ticker
;

