import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# interface/app.py
import gradio as gr
from transformers_module.transformer_brain import generate_empathic_reply
from context_engine.memory import add_to_memory, get_relevant_context

def respond(user_message, chat_history):
    # retrieve context from memory
    context = get_relevant_context(user_message)
    reply = generate_empathic_reply(user_message, context)
    # save into memory
    add_to_memory({"role":"user","text":user_message})
    add_to_memory({"role":"assistant","text":reply})
    chat_history.append(("You", user_message))
    chat_history.append(("Assistant", reply))
    return chat_history, ""

with gr.Blocks() as demo:
    chat = gr.Chatbot()
    msg = gr.Textbox(show_label=False, placeholder="Share how you're feeling...")
    state = gr.State([])

    def submit(message, history):
        history, _ = respond(message, history)
        return history, ""
    msg.submit(submit, [msg, state], [chat, msg])
    demo.launch()
