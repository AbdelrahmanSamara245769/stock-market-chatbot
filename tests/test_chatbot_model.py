from unittest.mock import patch
from stock_chatbot.ui.chatbot_model import init_llm


@patch.dict("os.environ", {"OLLAMA_BASE_URL": "http://llm.example:11434", "OLLAMA_MODEL": "test-model"})
@patch("stock_chatbot.ui.chatbot_model.ChatOllama")
def test_init_llm(mock_chat_ollama):
    # Setup mock return
    mock_instance = mock_chat_ollama.return_value

    # Call the function
    llm = init_llm()

    # The server and model come from the environment
    mock_chat_ollama.assert_called_once_with(
        base_url="http://llm.example:11434",
        model="test-model",
    )
    assert llm == mock_instance
