import pytest
from unittest.mock import patch
from stock_chatbot.ui.ui import launch_ui  

@patch("gradio.Blocks.launch")
def test_launch_ui(mock_launch):

    launch_ui()

    # Assert launch() was called once
    mock_launch.assert_called_once()

