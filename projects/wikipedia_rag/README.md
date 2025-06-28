# Wikipedia RAG

This project implements a Retrieval-Augmented Generation (RAG) application that answers questions based on content from a curated list of Wikipedia articles. It uses a local LLM served by [Ollama](https://ollama.com/) for both language generation and creating embeddings, ensuring privacy and no API costs.

## Features

- **Local LLM-Powered**: Leverages local models via Ollama for both the language model and embeddings.
- **Wikipedia as a Knowledge Base**: Uses a predefined list of Wikipedia pages as the source of information.
- **Persistent Indexing**: Creates and saves a vector index of the Wikipedia content on the first run for faster subsequent startups.
- **Informative Sources**: Displays the source passages from Wikipedia that were used to generate the answer, along with their relevance scores.
- **Interactive UI**: Built with Streamlit for a simple and user-friendly web interface.

## Prerequisites

Before running the application, you must have Ollama installed and running. You also need to pull the required models.

1.  **Install Ollama**: Follow the official instructions at [ollama.com](https://ollama.com/).

2.  **Pull the necessary models**: Open your terminal and run the following commands:
    ```bash
    ollama pull llama3.2
    ollama pull nomic-embed-text
    ```
    > **Note**: `llama3.2` is the model specified in the code. If you encounter issues, you might try a more standard model tag like `llama3` or `llama3:8b` and update the `LLM_MODEL` constant in `wiki_rag.py`.

## Installation

All project dependencies can be installed from the root of the repository using Poetry. If you haven't already, run:

```bash
poetry install
```

## Running the Application

To run the Wikipedia RAG application, navigate to the root of the `ai-projects-playground` repository and execute the following command:

```bash
poetry run streamlit run projects/wikipedia_rag/app/wiki_rag.py
```

The first time you run the application, it will take some time to download the Wikipedia pages and build the vector index. This index will be saved in the `projects/wikipedia_rag/app/wiki_rag/` directory, making subsequent launches much faster.

## How It Works

1.  **Indexing**: On the first launch, the application uses `WikipediaReader` to fetch the content of all pages listed in the `PAGES` constant in `wiki_rag.py`.
2.  **Embedding**: The content of these pages is then chunked and converted into vector embeddings using the local `nomic-embed-text` model.
3.  **Storage**: The resulting vector index is persisted to disk in the `wiki_rag` directory. On subsequent runs, the app loads the index directly from this directory instead of rebuilding it.
4.  **Querying**: When you ask a question, the application converts your query into an embedding, finds the most relevant text chunks from the Wikipedia index (retrieval), and then passes these chunks along with your original question to the local `llama3.2` model to generate a comprehensive answer (generation).

## Configuration

You can easily customize the application by modifying the constants at the top of the `projects/wikipedia_rag/app/wiki_rag.py` file:

-   `LLM_MODEL`: Change the primary language model used for answering questions.
-   `EMBEDDING_MODEL`: Change the model used for generating embeddings.
-   `PAGES`: Add or remove Wikipedia page titles to customize the knowledge base.
-   `OLLAMA_REQUEST_TIMEOUT`: Increase this value if you experience timeouts with larger models or on slower hardware.