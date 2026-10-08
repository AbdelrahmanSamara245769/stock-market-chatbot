import pytest
from unittest.mock import patch, MagicMock

import stock_chatbot.ui.sql_query_graph # Replace with your actual module name where the code lives


@pytest.fixture
def mock_llm():
    mock = MagicMock()
    # For llm.invoke calls returning an object with .content attribute
    def invoke_side_effect(prompt):
        if "database router" in prompt:
            # simulate choosing 'features' db
            response = MagicMock()
            response.content = "features"
            return response
        elif "SQL query executed" in prompt:
            # simulate answer generation
            response = MagicMock()
            response.content = "This is a generated answer."
            return response
        else:
            # fallback mock response
            response = MagicMock()
            response.content = "mocked response"
            return response

    mock.invoke.side_effect = invoke_side_effect

    # For llm.with_structured_output().invoke returning a dict with 'query'
    def with_structured_output_side_effect(_):
        class Invoker:
            def invoke(self_inner, prompt):
                return {"query": "SELECT * FROM features LIMIT 5;"}
        return Invoker()

    mock.with_structured_output.side_effect = with_structured_output_side_effect
    return mock


@pytest.fixture
def mock_db_configs():
    return {
        'features': {
            'db': MagicMock(dialect='postgresql'),
            'table_info': 'Table: features\nDescription: Dataset about daily values of features of stock indices\n\n'
        },
        'employees_mentor_test': {
            'db': MagicMock(dialect='postgresql'),
            'table_info': 'Table: employees_mentor_test\nDescription: Dataset about employees\n\n'
        }
    }


@patch('stock_chatbot.ui.sql_query_graph.init_llm')
@patch('stock_chatbot.ui.sql_query_graph.get_database_configs')
@patch('stock_chatbot.ui.sql_query_graph.QuerySQLDatabaseTool')
def test_choose_db_write_execute_generate(
    mock_tool_class,
    mock_get_db_configs,
    mock_init_llm,
    mock_llm,
    mock_db_configs,
):
    mock_init_llm.return_value = mock_llm
    mock_llm.invoke.return_value = MagicMock(content="This is a generated answer.")
    mock_get_db_configs.return_value = mock_db_configs

    mock_tool_instance = MagicMock()
    mock_tool_instance.invoke.return_value = "query result data"
    mock_tool_class.return_value = mock_tool_instance


@patch('stock_chatbot.ui.sql_query_graph.answer_question')
def test_get_answer_ui(mock_answer_question):
    # Mock the generator to yield a final state with answer
    mock_gen = iter([{
        'generate_answer': {'answer': 'final answer here'}
    }])
    mock_answer_question.return_value = mock_gen

    result = stock_chatbot.ui.sql_query_graph.get_answer_ui("Some question", history=[])
    assert result == 'final answer here'


@patch('stock_chatbot.ui.sql_query_graph.answer_question')
def test_get_answer_ui_no_answer(mock_answer_question):
    # Mock the generator to yield a final state without answer
    mock_gen = iter([{
        'generate_answer': {}
    }])
    mock_answer_question.return_value = mock_gen

    result = stock_chatbot.ui.sql_query_graph.get_answer_ui("Some question", history=[])
    assert result == 'No answer found.'
