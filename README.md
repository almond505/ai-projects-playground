# ai-projects-playground

This repository contains a collection of AI projects.

## Projects

- [Wikipedia RAG](./projects/wikipedia_rag/README.md)
- [Sentiment Analyzer](.projects/sentiment_analyzer/app/sentiment_analyzer.py)
- [YouTube Summarizer](./projects/youtube_summarizer/app/summarizer.py)



## Quick Start

### 1. Installation

First, ensure you have [Poetry](https://python-poetry.org/docs/#installation) installed. Then, from the root of the repository, install the project dependencies:

```bash
poetry install
```

### 2. Running an Application

You can run any of the project applications using `poetry run`.

*   **To run the YouTube Summarizer:**
    ```bash
    poetry run python3 projects/youtube_summarizer/app/summarizer.py
    ```
*   **To run the Sentiment Analyzer:**
    ```bash
    poetry run python3 projects/sentiment_analyzer/app/sentiment_analyzer.py
    ```
*   **To run the Wikipedia RAG:**
    ```bash
    poetry run streamlit run projects/wikipedia_rag/app/wiki_rag.py
    ```

## Usage Notes

### Wikipedia RAG

- This application requires a local Ollama instance to be running.
- For detailed setup instructions, including which models to pull, please see the Wikipedia RAG README.
- [Wikipedia RAG](./projects/wikipedia_rag/README.md)

### Sentiment Analyzer

- The application expects an Excel file (`.xlsx`).
- The Excel file **must** contain a column named `Reviews` for the analysis to work correctly.
