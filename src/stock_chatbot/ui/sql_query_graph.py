"""
SQL query graph utilities for the stock_chatbot project.

This module provides a LangGraph-based workflow for answering natural language questions
about financial data using SQL queries. It selects the appropriate database, generates
SQL queries, executes them, and produces a user-friendly answer.

Functions:
- answer_question: Runs the full graph pipeline to answer a user's question.
- get_answer_ui: Returns the final answer for UI integration.
"""

from langchain_community.utilities import SQLDatabase
from typing import TypedDict
from typing_extensions import Annotated
from langgraph.graph import START, StateGraph
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from stock_chatbot.ui.chatbot_model import init_llm
from langchain.prompts import PromptTemplate
from stock_chatbot.ui.db_connections import get_database_configs
from stock_chatbot.config import get_db_url
from collections import deque

uri = get_db_url()
tables = ['clustering_predictions_poetry', 'classification_predictions_poetry']
descriptions = {
    'clustering_predictions_poetry': '''Dataset about categories (safest, solid, avoid, outlier) of stocks obtained from clustering. Available columns: category, cluster, intraday_volatility_mean, return_1d_mean, return_1d_std, return_6m, ticker, volume_mean, volume_std)''',

    'classification_predictions_poetry': '''Dataset about predictions of stock prices (very_high, high, no_change, low, very_low) obtained from classification. Available columns: company_prefix, date_value, predicted_price_label''',
}

llm = init_llm()

prompt_template = PromptTemplate.from_template(
    """
    Given an input question, create a syntactically correct {dialect} query to run to help find the answer. 
    Unless the user specifies in his question a specific number of examples they wish to obtain, 
    always limit your query to at most {top_k} results. 
    You can order the results by a relevant column to return the most interesting examples in the database.

    Never query for all the columns from a specific table, only ask for a few relevant columns given the question.

    Pay attention to use only the column names that you can see in the schema description. 
    Be careful to not query for columns that do not exist. 
    Also, pay attention to which column is in which table.

    Only use the following table's description and info:
    
    {table_info},
    
    {table_description}.

    Question: {input}
    """
)

class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str
    db: SQLDatabase
    db_info: str
    db_description: str

class QueryOutput(TypedDict):
    query: Annotated[str, ..., "Syntactically valid SQL query."]

def choose_db(state: State) -> dict:
    """
    Choose the database to query.
    """
    db_configs = get_database_configs(uri, tables, descriptions)
    prompt_template_router = PromptTemplate.from_template(
        """
        You are a database router. Your task is to choose **only one** database from the list that is best suited to answer the user's question.

        Available Databases:
        {db_descriptions}

        Question: {question}

        Return only the exact database name (e.g., features or employees_mentor_test). Do not explain your reasoning or include any extra text. Only output the database name.
        """
    )

    prompt = prompt_template_router.invoke(
        {'db_descriptions': db_configs, 'question': state['question']}
    )

    answer = llm.invoke(prompt)
    chosen_db = db_configs[answer.content.strip()]
    return {'db': chosen_db['db'], 'db_info': chosen_db['table_info'], 'db_description': chosen_db['description']}

def write_query(state: State) -> dict:
    """
    Generate a SQL query for the selected database and question.
    """
    prompt = prompt_template.invoke(
        {
            "dialect": state["db"].dialect,
            "top_k": 5,
            "table_info": state["db_info"],
            "table_description": state["db_description"],
            "input": state["question"],
        }
    )
    result = llm.with_structured_output(QueryOutput).invoke(prompt)
    return {"query": result["query"]}

def execute_query(state: State) -> dict:
    """
    Execute the generated SQL query on the selected database.
    """
    execute_query_tool = QuerySQLDatabaseTool(db=state['db'])
    return {"result": execute_query_tool.invoke(state["query"])}

def generate_answer(state: State) -> dict:
    """
    Generate a user-friendly answer based on the SQL result.
    """
    prompt = (
        "You are a helpful investment assistant specialized in financial data and stock market analysis.\n\n"
        "Based on the user's question, the SQL query executed, and the resulting data from the database, "
        "provide a clear, concise, and informative answer that helps the user understand or make informed investment decisions.\n\n"
        "Never invent, guess, or add information that is not explicitly stated in the SQL result."
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}

def answer_question(question: str):
    """
    Run the full graph pipeline to answer a user's question using SQL and LLM.
    """
    graph_builder = StateGraph(State).add_sequence(
        [choose_db, write_query, execute_query, generate_answer]
    )
    graph_builder.add_edge(START, "choose_db")
    graph = graph_builder.compile()

    return graph.stream({"question": question})

from collections import deque

def get_answer_ui(question: str, history):
    """
    Returns the final answer for UI integration.
    """
    graph_gen = answer_question(question)
    final_state = deque(graph_gen, maxlen=1)[0]
    final_answer = final_state.get('generate_answer', {}).get('answer', 'No answer found.')
    return final_answer