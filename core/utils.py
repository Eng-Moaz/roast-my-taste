"""
utils.py — Shared utility functions for The Critic.

Provides input validation, prompt formatting helpers, and the
web search wrapper used by the response pipeline in bot.py.
"""

import os
from langchain_community.utilities import GoogleSerperAPIWrapper

def format_items_for_prompt(items: list) -> str:
    """
    Format a list of previously roasted items into a bulleted string.

    Args:
        items: A list of confession strings accumulated during the session.

    Returns:
        A newline-separated bullet list, or a placeholder string if the list
        is empty.
    """
    if not items:
        return "(No past crimes recorded yet)"

    return "\n".join(f"- {item}" for item in items)


def validate_user_input(user_input: str) -> tuple[bool, str]:
    """
    Validate and sanitize raw user input before it reaches the LLM.

    Rejects inputs that are empty, too short to be meaningful, or
    excessively long. Returns an in-character fallback message when
    validation fails so the user still receives a natural response.

    Args:
        user_input: The raw string submitted by the user.

    Returns:
        A tuple of (is_valid, result) where `result` is either the
        sanitized input string on success or a fallback message on failure.
    """
    clean_input = user_input.strip()

    if not clean_input:
        return False, "*stares blankly* ...You didn't say anything! Are you just standing there waiting for me to judge your silence? Confess your terrible taste!"

    if len(clean_input) < 2:
        return False, "*sighs loudly* A single letter or symbol? Speak up! Tell me a real movie, song, or food you actually like."

    if len(clean_input) > 500:
        return False, "*faints* Whoa, are you writing a novel? I don't have the patience to read all that. Keep your confession short and sweet!"

    return True, clean_input


def execute_safe_search(search_query: str, max_retries: int = 2) -> str:
    """
    Execute a Google search via the Serper API and return a text summary.

    Retries the request up to `max_retries` times on failure. Returns a
    graceful fallback string if the API key is absent or all attempts fail,
    so the response pipeline can continue without raising an exception.

    Args:
        search_query: The query string to submit to the search API.
        max_retries: Number of attempts before giving up (default: 2).

    Returns:
        A string containing the search result summary, or a fallback message
        if the search could not be completed.
    """
    serper_key = os.getenv("SERPER_API_KEY", "")
    if not serper_key:
        return "No internet context (SERPER_API_KEY not set in .env)."

    for attempt in range(max_retries):
        try:
            search = GoogleSerperAPIWrapper(
                serper_api_key=serper_key,
                k=8,
            )
            result = search.run(search_query)
            if result and len(result.strip()) >= 30:
                return result
        except Exception as e:
            print(f"[Search attempt {attempt + 1} failed] {type(e).__name__}: {e}")

    return "No internet context found (search failed)."
