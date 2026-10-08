from unittest.mock import patch, MagicMock
from stock_chatbot.ui.db_connections import get_database_configs  # Adjust based on filename


@patch("stock_chatbot.ui.db_connections.SQLDatabase")
def test_get_database_configs(mock_sql_database):
    # Mock the return value of SQLDatabase.from_uri
    mock_db_instance = MagicMock()
    mock_sql_database.from_uri.return_value = mock_db_instance

    # Define test inputs
    uri = "sqlite:///test.db"
    tables = ["users", "orders"]
    descriptions = {
        "users": "Contains user information",
        "orders": "Contains order history"
    }

    # Run the function
    configs = get_database_configs(uri, tables, descriptions)

    # Assertions
    mock_sql_database.from_uri.assert_called_once_with(uri, include_tables=tables)

    assert isinstance(configs, dict)
    assert set(configs.keys()) == set(tables)
    for table in tables:
        assert configs[table]["db"] == mock_db_instance
        assert descriptions[table] in configs[table]["table_info"]
        assert f"Table: {table}" in configs[table]["table_info"]
