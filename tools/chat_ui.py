"""A chat page for the RAG API. Start the API first (poetry poe run-inference-ml-service).

Usage: poetry run python -m tools.chat_ui  → open http://127.0.0.1:7860
"""

import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/rag"

EXAMPLES = [
    "What is the FTI architecture?",
    "How does 4-bit quantization with GPTQ work?",
    "What are the advanced RAG techniques?",
    "What is the capital of France?",
]


def ask_rag(message: str, history: list) -> str:
    try:
        response = requests.post(API_URL, json={"query": message}, timeout=120)
        response.raise_for_status()
    except requests.RequestException as error:
        return f"⚠️ Could not reach the RAG API at {API_URL}. Is it running?\n\n`{error}`"

    return response.json()["answer"]


demo = gr.ChatInterface(
    fn=ask_rag,
    type="messages",
    title="🧠 LLM Twin: RAG Assistant",
    description="Ask about LLMs, RAG and MLOps. Answers come from articles by Paul Iusztin and Maxime Labonne.",
    examples=EXAMPLES,
    theme=gr.themes.Soft(primary_hue="indigo"),
)


if __name__ == "__main__":
    demo.launch()
