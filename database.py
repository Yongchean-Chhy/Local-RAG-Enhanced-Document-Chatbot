import argparse
import os
import shutil
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
from embedding_function import get_embedding_function

DATA_PATH = "data"
CHROMA_PATH = "vector_data_base"

def doc_loader():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    document = loader.load()
    return document

def doc_splitter(doucments: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 800,
                                                   chunk_overlap = 80,
                                                   length_function = len,
                                                   is_separator_regex=False)
    return text_splitter.split_documents(doucments)


def get_chunk_ids(chunks):
    last_page_id = None
    cur_chunk_idx = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        cur_page_id = f"{source} : {page}"

        if cur_page_id == last_page_id:
            cur_chunk_idx += 1
        else:
            cur_chunk_idx = 0
        
        chunk_id = f"{cur_page_id} : {cur_chunk_idx}"
        last_page_id = cur_page_id

        chunk.metadata["id"] = chunk_id
    return chunks

def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory = CHROMA_PATH, embedding_function = get_embedding_function()
    )

    chunks_with_ids = get_chunk_ids(chunks)

    # existing_items = db.get(include = [])
    existing_items = db.get()
    existing_ids = set(existing_items["ids"])
    print(f"Existing document is Database: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
    
    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunks_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, id = new_chunks_ids)
    else:
        print("No new element added")

def clear():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("Clearing Database")
        clear()

    documents = doc_loader()
    chunks = doc_splitter(documents)
    add_to_chroma(chunks)
