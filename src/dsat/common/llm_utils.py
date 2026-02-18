"""
LLM Utilities for DSAT - Vertex AI Gemini Integration

Provides shared functions for calling Gemini models and parsing JSON responses.
Used across all pipeline nodes that require LLM capabilities.
"""

import json
import re
import logging
from typing import Any, Dict, Optional

import vertexai
from vertexai.generative_models import GenerativeModel

logger = logging.getLogger(__name__)

# Track whether Vertex AI has been initialized
_vertex_ai_initialized = False


def init_vertex_ai(project_id: str, location: str = "us-central1") -> None:
    """
    Initialize Vertex AI. Safe to call multiple times — only initializes once.

    Args:
        project_id: GCP project ID
        location: GCP region (default: us-central1)
    """
    global _vertex_ai_initialized
    if not _vertex_ai_initialized:
        vertexai.init(project=project_id, location=location)
        _vertex_ai_initialized = True
        logger.info(f"Vertex AI initialized for project={project_id}, location={location}")


def call_gemini(
    prompt: str,
    model_name: str = "gemini-2.0-flash",
    project_id: Optional[str] = None,
    location: str = "us-central1",
) -> str:
    """
    Call Vertex AI Gemini model and return the response text.

    Args:
        prompt: The prompt to send to the model
        model_name: Gemini model name (default: gemini-2.0-flash)
        project_id: GCP project ID (used for auto-init if needed)
        location: GCP region

    Returns:
        Response text from the model
    """
    # Auto-init if project_id provided and not yet initialized
    if project_id and not _vertex_ai_initialized:
        init_vertex_ai(project_id, location)

    model = GenerativeModel(model_name)
    response = model.generate_content(prompt)

    # Extract text from response
    response_text = ""
    if response.candidates:
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                response_text += part.text

    logger.info(f"Gemini response ({model_name}): {response_text[:200]}...")
    return response_text


def parse_llm_json(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Safely extract and parse JSON from LLM response text.

    Handles common LLM output formats:
    - ```json { ... } ```
    - Raw JSON object { ... }
    - JSON with surrounding text

    Args:
        response_text: Raw text from LLM

    Returns:
        Parsed JSON dict, or None if parsing fails
    """
    json_str = None

    # 1. Try to extract from ```json ... ``` code fence
    code_fence_match = re.search(
        r"```(?:json)?\s*([\s\S]*?)\s*```",
        response_text,
        re.IGNORECASE,
    )

    if code_fence_match:
        json_str = code_fence_match.group(1).strip()

    # 2. Fallback: extract first JSON object anywhere in text
    if not json_str:
        fallback_match = re.search(r"\{[\s\S]*\}", response_text)
        if fallback_match:
            json_str = fallback_match.group(0).strip()

    # 3. Parse
    if not json_str:
        logger.warning("No JSON found in LLM response")
        return None

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        logger.error(f"Attempted to parse: {json_str[:400]}...")
        return None


def parse_llm_json_list(response_text: str) -> Optional[list]:
    """
    Safely extract and parse a JSON list from LLM response text.

    Args:
        response_text: Raw text from LLM

    Returns:
        Parsed JSON list, or None if parsing fails
    """
    json_str = None

    # 1. Try code fence
    code_fence_match = re.search(
        r"```(?:json)?\s*(\[[\s\S]*?\])\s*```",
        response_text,
        re.IGNORECASE,
    )
    if code_fence_match:
        json_str = code_fence_match.group(1).strip()

    # 2. Fallback: find first JSON list
    if not json_str:
        fallback_match = re.search(r"\[[\s\S]*\]", response_text)
        if fallback_match:
            json_str = fallback_match.group(0).strip()

    if not json_str:
        logger.warning("No JSON list found in LLM response")
        return None

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"JSON list decode error: {e}")
        return None
