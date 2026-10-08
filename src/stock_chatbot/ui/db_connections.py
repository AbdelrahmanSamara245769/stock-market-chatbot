"""
Database connection utilities for the stock_chatbot project.

This module provides a function to create SQLDatabase objects for specified tables
and collect their metadata and descriptions for use in the UI or chatbot.

Functions:
- get_database_configs: Initializes SQLDatabase objects and gathers table info and descriptions.
"""

from langchain_community.utilities import SQLDatabase

def get_database_configs(uri, tables, descriptions):
    """
    Initializes SQLDatabase objects for each table and gathers their info and descriptions.

    Args:
        uri (str): Database URI.
        tables (list): List of table names to include.
        descriptions (dict): Dictionary mapping table names to descriptions.

    Returns:
        dict: Dictionary with table names as keys and database/config info as values.
    """
    db_dict = {}
    for table in tables:
        try:
            sql_db = SQLDatabase.from_uri(database_uri=uri, include_tables=[table])
            sql_db_info = sql_db.get_table_info()
        except Exception as e:
            sql_db = None
            sql_db_info = f'Error retrieving table info: {str(e)}'

        db_dict[table] = {
            'db': sql_db,
            'description': descriptions[table],
            'table_info': sql_db_info
        }
    return db_dict
