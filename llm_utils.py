# llm_utils.py
from langchain_groq import ChatGroq

def initialize_llm(api_key):
    """
    Initialize the ChatGroq language model with the provided API key.
    """
    return ChatGroq(
        model="Llama3-8b-8192",
        temperature=0,
        groq_api_key=api_key
    )
