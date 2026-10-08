CREATE TABLE IF NOT EXISTS public.features (
  feature_id SERIAL PRIMARY KEY,

  -- Columns copied from nasdaq_100_daily:
  date_value    DATE       NOT NULL,
  company_prefix TEXT      NOT NULL,
  open_value     NUMERIC,
  high_value     NUMERIC,
  low_value      NUMERIC,
  close_value    NUMERIC,
  volume         NUMERIC,
  transactions   NUMERIC,

  -- Columns copied from macro_economic_indicators:
  gdp_growth                                NUMERIC,
  unemployment_rate                         NUMERIC,
  inflation_cpi                             NUMERIC,
  interest_rate_fed_funds                   NUMERIC,
  "10_year_treasury_yield"                  NUMERIC,
  stock_market_volatility_vix_index         NUMERIC,
  retail_sales_data_excluding_food_services NUMERIC
);
