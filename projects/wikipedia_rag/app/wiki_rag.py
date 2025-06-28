"""
Wikipedia RAG Streamlit Application.

This module builds a Retrieval-Augmented Generation (RAG) application
using Streamlit and LlamaIndex. It fetches data from a predefined list
of Wikipedia pages, creates a vector index, and allows users to query
the index to get answers based on the Wikipedia content.
"""

import os
import sys
from typing import List

import streamlit as st
from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.readers.wikipedia import WikipediaReader

# --- Configuration ---
INDEX_DIR = "wiki_rag"

# Local LLM and Embedding Model Configuration (via Ollama)
# Make sure you have Ollama running and have pulled the models:
# `ollama pull llama3.2`
# `ollama pull nomic-embed-text`
LLM_MODEL = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_TEMPERATURE = 0
SIMILARITY_TOP_K = 3
OLLAMA_REQUEST_TIMEOUT = 300.0

# List of Wikipedia pages to use as the knowledge base
PAGES: List[str] = [
    "Honda", "Toyota", "Mazda", "Mitsubishi", "Nissan", "Suzuki",
    # "Volkswagen", "Hyundai", "Kia", "Ford Motor Company", "Chevrolet",
    # "Tesla, Inc.", "BMW", "Audi", "Mercedes-Benz", "BYD Auto", "MG Motor",
    # "SAIC Motor", "Fiat", "Renault", "Peugeot", "Geely", "XPENG", "Volvo",
    # "GAC Aion", "Great Wall Motor", "Ferrari", "Lamborghini", "Porsche",
    # "Automobiles Alpine", "Aston Martin", "Bentley", "Rolls-Royce",
    # "Bugatti", "Alfa Romeo", "McLaren", "Maserati",
]


@st.cache_resource
def get_index() -> VectorStoreIndex:
    """
    Loads the vector index from storage if it exists, otherwise creates and
    persists a new one.

    The index is built from a predefined list of Wikipedia pages.
    The function is cached using Streamlit's cache_resource to avoid
    re-creating the index on every app rerun.

    Returns:
        The loaded or newly created VectorStoreIndex.
    """
    if os.path.isdir(INDEX_DIR):
        st.info(f"Loading index from storage directory: {INDEX_DIR}")
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        return load_index_from_storage(storage_context)

    with st.spinner(f"Fetching and indexing {len(PAGES)} Wikipedia pages. This may take a moment..."):
        docs = WikipediaReader().load_data(pages=PAGES, auto_suggest=False)
        try:
            embed_model = OllamaEmbedding(
                model_name=EMBEDDING_MODEL,
                request_timeout=OLLAMA_REQUEST_TIMEOUT,
            )
        except Exception as e:
            st.error(f"Failed to initialize embedding model. Is Ollama running and is '{EMBEDDING_MODEL}' pulled? Error: {e}")
            st.stop()
            sys.exit(1)

        index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
        index.storage_context.persist(persist_dir=INDEX_DIR)
        st.success("Index created and saved successfully!")
    return index


@st.cache_resource
def get_query_engine() -> BaseQueryEngine:
    """
    Creates and returns a query engine from the vector index.

    The query engine is configured with a specific local Llama model and
    similarity settings. The function is cached to avoid re-creating
    the engine on every interaction.

    Returns:
        The configured query engine.
    """
    index = get_index()
    try:
        llm = Ollama(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            request_timeout=OLLAMA_REQUEST_TIMEOUT,
        )
    except Exception as e:
        st.error(f"Failed to initialize LLM. Is Ollama running and is '{LLM_MODEL}' pulled? Error: {e}")
        st.stop()
        sys.exit(1)

    return index.as_query_engine(llm=llm, similarity_top_k=SIMILARITY_TOP_K)


def main() -> None:
    """
    Sets up and runs the Streamlit user interface for the RAG application.
    """
    st.title("Wikipedia RAG Application")
    st.write(
        "Ask a question about one of the indexed car manufacturers, "
        "and the RAG system will find relevant information from Wikipedia to answer it. "
        "This version uses a local Llama model via Ollama."
    )

    question = st.text_input("Enter your question:", placeholder="e.g., What is the history of Ferrari?")

    if st.button("Get Answer") and question:
        with st.spinner("Generating answer..."):
            query_engine = get_query_engine()
            response = query_engine.query(question)

        st.subheader("Answer")
        st.write(response.response)

        st.subheader("Sources")
        for source in response.source_nodes:
            with st.expander(f"Source: {source.metadata.get('page_title', 'Unknown')} (Score: {source.score:.2f})"):
                st.markdown(source.get_content())


if __name__ == "__main__":
    main()