"""
YouTube Video Summarizer

This script takes a YouTube video URL, extracts its transcript,
and generates a summary using a pre-trained Hugging Face model.
It then launches a Gradio interface for user interaction.
"""
import logging
import re
import traceback
from typing import Optional, Dict, Any, List
import torch
import gradio as gr
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    InvalidVideoId)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from youtube_transcript_api.formatters import TextFormatter
from transformers import pipeline

# Initialize the summarization pipeline
summarization_pipeline = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    torch_dtype=torch.bfloat16
)

def generate_summary(text_input: str) -> str:
    """
    Generates a summary for the given text.

    Args:
        text_input: The text to be summarized.

    Returns:
        The summarized text.
    """
    logging.info(f"Input text length for summarization: {len(text_input)} characters")

    # Determine the model's maximum input length (in tokens)
    # For 'sshleifer/distilbart-cnn-12-6', this is typically 1024 tokens.
    max_model_input_tokens = summarization_pipeline.tokenizer.model_max_length
    if max_model_input_tokens > 1024: # Fallback for models with very large max_length
        max_model_input_tokens = 1024

    target_chunk_tokens = int(max_model_input_tokens * 0.8) # Use 80% of max tokens for chunk
    chunk_size_chars = target_chunk_tokens * 4 # Convert to characters
    overlap_chars = int(max_model_input_tokens * 0.1 * 4) # 10% overlap in characters
    
    if len(text_input) > chunk_size_chars:
        logging.info(f"Transcript is too long ({len(text_input)} chars). Splitting into chunks for summarization.")
        summaries = []
        start_idx = 0
        while start_idx < len(text_input):
            end_idx = min(start_idx + chunk_size_chars, len(text_input))
            chunk = text_input[start_idx:end_idx]

            # Summarize each chunk with a shorter desired output length
            chunk_summary_output: List[Dict[str, Any]] = summarization_pipeline(chunk, min_length=20, max_length=100)
            summaries.append(chunk_summary_output[0]['summary_text'])

            start_idx += chunk_size_chars - overlap_chars
            if start_idx < 0: # Prevent negative start_idx if overlap is large
                start_idx = 0

        final_summary = " ".join(summaries) # This line was missing in the original code
        logging.info(f"Combined summary from {len(summaries)} chunks. Length: {len(final_summary)} characters.")
        return final_summary # Return the combined summary

    else:
        output: List[Dict[str, Any]] = summarization_pipeline(text_input, min_length=50, max_length=200)
        logging.info(f"Generated summary length: {len(output[0]['summary_text'])} characters")
        return output[0]['summary_text']

def extract_video_id(url: str) -> Optional[str]:
    """
    Extracts the video ID from various YouTube URL formats.

    Args:
        url: The YouTube video URL.

    Returns:
        The extracted video ID, or None if not found.
    """
    regex = (
        r"(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)"
        r"([a-zA-Z0-9_-]{11})"
    )
    match = re.search(regex, url)
    if match:
        return match.group(1)
    return None

def get_youtube_summary_from_url(video_url: str) -> str:
    """
    Fetches the transcript of a YouTube video and returns its summary.

    Args:
        video_url: The URL of the YouTube video.

    Returns:
        The summary of the video transcript, or an error message.
    """
    video_id = extract_video_id(video_url)
    if not video_id:
        return "Video ID could not be extracted. Please provide a valid YouTube URL."

    logging.info(f"Attempting to summarize video with ID: {video_id}")
    try:
        fetched_transcript = YouTubeTranscriptApi().fetch(video_id)
        formatter = TextFormatter()
        text_transcript = formatter.format_transcript(fetched_transcript)

        if not text_transcript.strip():
            logging.warning("Formatted transcript is empty or contains only whitespace.")
            return "Transcript is empty or could not be formatted."

        logging.info(f"Successfully fetched transcript. Length: {len(text_transcript)} characters.")
        summary_text = generate_summary(text_transcript)
        return summary_text
    except NoTranscriptFound:
        logging.warning(f"No transcript found for video {video_id}.")
        return "No transcript found for this video (or auto-generated captions are not available/disabled)."
    except TranscriptsDisabled:
        logging.warning(f"Transcripts are disabled for video {video_id}.")
        return "Transcripts are disabled for this video."
    except VideoUnavailable:
        logging.warning(f"Video {video_id} is unavailable or private.")
        return "Video is unavailable or private."
    except InvalidVideoId:
        logging.error(f"Invalid video ID: {video_id}.")
        return "Invalid video ID. Please check the URL."
    except Exception as e:
        logging.exception(f"An unexpected error occurred while processing video {video_id}:")
        return f"An unexpected error occurred: {e}"

if __name__ == "__main__":
    gr.close_all() # Closes any existing Gradio interfaces

    iface = gr.Interface(
        fn=get_youtube_summary_from_url,
        inputs=[
            gr.Textbox(label="Input YouTube URL to summarize", lines=1)
        ],
        outputs=[
            gr.Textbox(label="Summarized text", lines=10)
        ],
        title="YouTube Video Summarizer",
        description="Enter a YouTube video URL to get its transcript summarized."
    )
    iface.launch()