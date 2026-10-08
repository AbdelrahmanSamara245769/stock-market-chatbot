"""
Configuration and constants for the stock_chatbot package.

This module defines feature lists, database connection parameters, and logging setup
for the stock_chatbot project. Connection details are read from environment variables
(see .env.example).

Exports:
- CLASSIFICATION_FEATURES: List of features used for classification models.
- DB_PARAMS: Dictionary of database connection parameters.
- get_db_url: Builds the PostgreSQL connection URL from DB_PARAMS.
- setup_logger: Function to configure and return a logger.
"""

import logging
import os

CLASSIFICATION_FEATURES = [
    "volume", "transactions", "gdp_growth", "unemployment_rate",
    "inflation_cpi", "interest_rate_fed_funds", "10_year_treasury_yield",
    "stock_market_volatility_vix_index", "retail_sales_data_excluding_food_services",
    "volatility_126_capped", "return_1d", "momentum_126", "roc_126",
    "bollinger_bandwidth", "ma_crossover_20_50", "volume_change_1d",
    "rsi_14_capped", "sharpe_126_capped"
]

DB_PARAMS = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'stock_warehouse'),
    'user': os.getenv('DB_USER', ''),
    'password': os.getenv('DB_PASSWORD', ''),
}


def get_db_url(params=None):
    """
    Builds the PostgreSQL connection URL for the data warehouse.

    Args:
        params (dict, optional): Connection parameters; defaults to DB_PARAMS.

    Returns:
        str: SQLAlchemy connection URL.
    """
    p = params or DB_PARAMS
    return f"postgresql://{p['user']}:{p['password']}@{p['host']}:{p['port']}/{p['database']}"


def setup_logger():
    """
    Sets up and returns a logger for the stock_chatbot package.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    return logger