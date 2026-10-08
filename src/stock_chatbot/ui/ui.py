import gradio as gr
from .sql_query_graph import get_answer_ui

def launch_ui():
    with gr.Blocks(theme=gr.themes.Soft(primary_hue='emerald'),
                   css="""
            #chat-container {
                max-width: 60%;
                margin-left: auto;
                margin-right: auto;
            }
            """
                   ) as demo:
        with gr.Column(elem_id="chat-container"):
            gr.Markdown(
                """
                <div style="font-size: 30px;">
                <strong>Welcome to the Investment Assistant!</strong><br>
                Ask any question about the company database or stock features. 
                I’ll generate an answer based on the most recent data.
                </div>
                """,
                elem_id="description"
            )
            gr.ChatInterface(
                fn=get_answer_ui,
                type='messages',
                chatbot=gr.Chatbot(height=500, type='messages'),
                textbox=gr.Textbox(
                    placeholder='Ask anything',
                    container=True,
                    scale=1,
                    lines=1
                )
            )

    demo.launch()