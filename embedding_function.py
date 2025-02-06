from langchain_ollama import OllamaEmbeddings

#function that return an embedding function
def get_embedding_function():
    embeddings =OllamaEmbeddings(model='mxbai-embed-large')
    return embeddings