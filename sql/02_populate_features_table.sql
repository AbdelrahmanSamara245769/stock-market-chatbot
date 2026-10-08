INSERT INTO public.features (
  date_value,
  company_prefix,
  open_value,
  high_value,
  low_value,
  close_value,
  volume,
  transactions,
  gdp_growth,
  unemployment_rate,
  inflation_cpi,
  interest_rate_fed_funds,
  "10_year_treasury_yield",
  stock_market_volatility_vix_index,
  retail_sales_data_excluding_food_services
)
SELECT
  n.date_value,
  n.company_prefix,
  n.open_value,
  n.high_value,
  n.low_value,
  n.close_value,
  n.volume,
  n.transactions,
  m.gdp_growth,
  m.unemployment_rate,
  m.inflation_cpi,
  m.interest_rate_fed_funds,
  m."10_year_treasury_yield",
  m.stock_market_volatility_vix_index,
  m.retail_sales_data_excluding_food_services
FROM
  public.nasdaq_100_daily AS n
  LEFT JOIN public.macro_economic_indicators AS m
    ON n.date_value = m.date_value;