"""
YouTube Video Summarizer

This script takes a YouTube video URL, extracts its transcript,
and generates a summary using a pre-trained Hugging Face model.
It then launches a Gradio interface for user interaction.
"""

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
    output: List[Dict[str, Any]] = summarization_pipeline(text_input)
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
        return "Video ID could not be extracted."

    print(f"video_id found: {video_id}")
    try:
        # The fetch() method returns a FetchedTranscript object, which is what the formatter expects.
        # It will raise NoTranscriptFound if no transcript is available in the default languages (e.g., 'en').
        fetched_transcript = YouTubeTranscriptApi().fetch(video_id)
        formatter = TextFormatter()
        text_transcript = formatter.format_transcript(fetched_transcript)

        if not text_transcript.strip(): # Check if the formatted transcript is empty or just whitespace
            return "Transcript is empty or could not be formatted."

        summary_text = generate_summary(text_transcript)
        return summary_text
    except NoTranscriptFound:
        return "No transcript found for this video (or auto-generated captions are not available/disabled)."
    except TranscriptsDisabled:
        return "Transcripts are disabled for this video."
    except VideoUnavailable:
        return "Video is unavailable or private."
    except InvalidVideoId:
        return "Invalid video id."
    except Exception as e:
        traceback.print_exc() # This will print the full stack trace to stderr
        print(e)
        return f"An error occurred: {e}"

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