# webpage_loader.py
from langchain_community.document_loaders import WebBaseLoader

def load_webpage_data(url):
    """
    Loads webpage content from the specified URL.
    """
    loader = WebBaseLoader(url)
    webpage_data = loader.load().pop().page_content
    return webpage_data
