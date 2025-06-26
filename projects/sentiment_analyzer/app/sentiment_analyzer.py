"""
Sentiment Analyzer Application
"""

from typing import Tuple
import matplotlib.figure
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from transformers import pipeline


# --- Constants ---
MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
REVIEW_COLUMN = "Reviews"


# --- Model Initialization ---
sentiment_pipeline = pipeline("text-classification", model=MODEL_NAME)


def get_sentiment_label(review: str) -> str:
    """
    Analyzes a single review string and returns its sentiment label.

    Args:
        review: The text of the review.

    Returns:
        The sentiment label ('POSITIVE' or 'NEGATIVE').
    """
    result = sentiment_pipeline(review)
    return result[0]['label']


def create_sentiment_pie_chart(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """
    Creates a pie chart of sentiment counts from a DataFrame.

    Args:
        df: A DataFrame containing a 'Sentiment' column.

    Returns:
        A matplotlib Figure object containing the pie chart.
    """
    sentiment_counts = df['Sentiment'].value_counts()

    fig, ax = plt.subplots()
    sentiment_counts.plot(
        kind='pie',
        ax=ax,
        autopct='%1.1f%%',
        colors=['green', 'red']
    )
    ax.set_title('Review Sentiment Analysis')
    ax.set_ylabel('')  # Hide the y-axis label for pie charts
    return fig




def process_reviews_file(
    file: object
) -> Tuple[pd.DataFrame, matplotlib.figure.Figure]:
    """
    Processes an uploaded Excel file containing user reviews, performs sentiment analysis
    on each review, and generates a pie chart summarizing the results.

    Args:
        file (object): An uploaded Excel file object
                       The file must contain a column named 'Reviews'.

    Returns:
        Tuple[pd.DataFrame, matplotlib.figure.Figure]: A tuple containing:
            - A pandas DataFrame with the original reviews and a new 'Sentiment' column.
            - A matplotlib Figure object showing the sentiment distribution as a pie chart.

    Raises:
        ValueError: If no file is uploaded or if the file does not contain a 'Reviews' column.
    """
    if file is None:
        raise ValueError("No file was uploaded.")

    df = pd.read_excel(file)

    if REVIEW_COLUMN not in df.columns:
        raise ValueError(f"Excel file must contain a '{REVIEW_COLUMN}' column.")

    df['Sentiment'] = df[REVIEW_COLUMN].apply(get_sentiment_label)
    chart_figure = create_sentiment_pie_chart(df)

    return df, chart_figure




def main():
    """Launches the Gradio interface for the sentiment analyzer."""
    gr.close_all() # Closes any existing Gradio interfaces
    demo = gr.Interface(
        fn=process_reviews_file,
        inputs=[gr.File(file_types=[".xlsx"], label="Upload your review comment file")],
        outputs=[
            gr.DataFrame(label="Analyzed Sentiments", interactive=False),
            gr.Plot(label="Sentiment Analysis Chart")
        ],
        title="Sentiment Analyzer",
        description=(
            "Upload an Excel file to analyze the sentiment of the reviews. "
            f"The file must contain a column named '{REVIEW_COLUMN}'."
        )
    )

    demo.launch()


if __name__ == "__main__":
    main()