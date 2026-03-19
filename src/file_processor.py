"""
File processor — extracts text and analyzes content from uploaded files.
Supports: PDF, Word (.docx), Images (JPG/PNG), plain text.
Uses DeepSeek vision for image analysis.
"""
import base64
import io
from pathlib import Path
from openai import OpenAI


def extract_text(filename: str, file_bytes: bytes, client: OpenAI = None) -> str:
    """
    Extract meaningful text from any supported file type.
    Returns a string description of the file's content.
    """
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(file_bytes)
    elif suffix in (".docx", ".doc"):
        return _extract_word(file_bytes)
    elif suffix in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
        return _analyze_image(file_bytes, suffix, client)
    elif suffix in (".txt", ".md", ".csv"):
        return file_bytes.decode("utf-8", errors="ignore")
    else:
        # Try as plain text
        try:
            return file_bytes.decode("utf-8", errors="ignore")
        except Exception:
            return f"[Could not extract text from {filename}]"


def _extract_pdf(file_bytes: bytes) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
        return "\n\n".join(pages) or "[PDF has no extractable text]"
    except ImportError:
        return "[pypdf not installed — cannot extract PDF text]"
    except Exception as e:
        return f"[PDF extraction error: {e}]"


def _extract_word(file_bytes: bytes) -> str:
    try:
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs) or "[Word doc has no text content]"
    except ImportError:
        return "[python-docx not installed — cannot extract Word text]"
    except Exception as e:
        return f"[Word extraction error: {e}]"


def _analyze_image(file_bytes: bytes, suffix: str, client: OpenAI) -> str:
    """
    Use DeepSeek vision to analyze an image and describe what it reveals
    about the person's personality, lifestyle, and social signals.
    """
    if client is None:
        return "[Image uploaded — no API client available for analysis]"

    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    media_type = media_type_map.get(suffix, "image/jpeg")
    b64 = base64.standard_b64encode(file_bytes).decode("utf-8")

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{b64}"}
                    },
                    {
                        "type": "text",
                        "text": """Analyze this image from a social/personality psychology perspective.
Describe:
1. What this image reveals about the person's lifestyle and personality
2. Social context (where they are, who they're with, social setting)
3. Fashion/aesthetic style and what it signals
4. Mood and emotional tone
5. Any notable details about their interests, values, or status signals

Be specific and insightful. Focus on personality inference, not just description."""
                    }
                ]
            }],
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # DeepSeek may not support vision in all tiers — fallback
        return f"[Image uploaded — vision analysis unavailable: {e}]"


def get_filetype_label(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    labels = {
        ".pdf": "PDF",
        ".docx": "Word",
        ".doc": "Word",
        ".jpg": "Image",
        ".jpeg": "Image",
        ".png": "Image",
        ".webp": "Image",
        ".gif": "Image",
        ".txt": "Text",
        ".md": "Markdown",
        ".csv": "CSV",
    }
    return labels.get(suffix, "File")
