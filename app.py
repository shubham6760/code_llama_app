import gradio as gr
import time
from ctransformers import AutoModelForCausalLM

def load_llm():
    llm = AutoModelForCausalLM.from_pretrained("codellama-13b-instruct.Q5_K_M.gguf",
    model_type='llama',
    max_new_tokens = 1096,
    repetition_penalty = 1.13,
    temperature = 0.1
    )
    return llm

def llm_function(message, chat_history):

    llm = load_llm()
    response = llm(
        message
    )
    output_texts = response
    return output_texts

title = "Shubham_CodeLlama 13B GGUF Demo"

examples = [
    'Write a python code to connect with two SQL database and list down all the tables.',
    'Write the python code to train a linear regression',
    'Explain the concepts of RFM analysis.',
    'write a code for RFM analysis. '
]

gr.ChatInterface(
    fn=llm_function,
    title=title,
    examples=examples
).launch(share=True)



