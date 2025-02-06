import argparse
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def query_rag(query : str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory = CHROMA_PATH, embedding_function = embedding_function)

    result = db.similarity_search_with_score(query, k=5)
    context_text = "\n---\n".join([doc.page_content for doc, score in result])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context = context_text, question = query)

    model = OllamaLLM(model="llama3.2")
    response = model.invoke(prompt)

    source = [doc.metadata.get("id", None) for doc, score in result]
    formatted_response = f"Response: {response} \n Source: {source}"
    print(formatted_response)
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="The query text.")
    args = parser.parse_args()
    query = args.query
    query_rag(query)
