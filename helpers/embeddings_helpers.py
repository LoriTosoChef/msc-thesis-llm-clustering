import logging
import re
import string

from nltk import word_tokenize

logger = logging.getLogger(__name__)


def clean_and_tokenize_text(text: str, tokenizer: word_tokenize, stopwords: set) -> list:
    """Helper function to fully clean text and tokenize it"""
    # Lowercase words
    text = str(text).lower()
    # Remove [+XYZ chars] in content
    text = re.sub(r"\[(.*?)\]", "", text)
    # Remove multiple spaces in content
    text = re.sub(r"\s+", " ", text)
    # Remove ellipsis (and last word)
    text = re.sub(r"\w+…|…", "", text)
    # Replace dash between words
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)
    # Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)

    # Get tokens from text
    tokens = tokenizer(text)
    # Remove stopwords
    tokens = [t for t in tokens if not t in stopwords]
    # Remove digits
    tokens = ["" if t.isdigit() else t for t in tokens]
    # Remove short tokens
    tokens = [t for t in tokens if len(t) > 1]
    
    return tokens


