import os
from langchain_community.utilities import GoogleSerperAPIWrapper

def format_items_for_prompt(items):
    """
    Format a list of previously roasted items into a bulleted string for the prompt.
    If the list is empty, return a string indicating there are no items yet.
    """
    if not items:
        return "(No past crimes recorded yet)"

    return "\n".join(f"- {item}" for item in items)


def validate_user_input(user_input):
    """
    Validates and sanitizes the user's input to handle unexpected or empty inputs gracefully.
    Returns a tuple: (is_valid, sanitized_input_or_fallback_message).
    This ensures the chatbot does not break on bad inputs and provides a natural conversational fallback.
    """
    clean_input = user_input.strip()

    if not clean_input:
        return False, "*stares blankly* ...You didn't say anything! Are you just standing there waiting for me to judge your silence? Confess your terrible taste!"

    if len(clean_input) < 2:
        return False, "*sighs loudly* A single letter or symbol? Speak up! Tell me a real movie, song, or food you actually like."

    if len(clean_input) > 500:
        return False, "*faints* Whoa, are you writing a novel? I don't have the patience to read all that. Keep your confession short and sweet!"

    return True, clean_input


def execute_safe_search(search_query, max_retries=2):
    """
    Executes a Google search via the Serper API using LangChain's GoogleSerperAPIWrapper.
    Serper is far more reliable and accurate than DuckDuckGo — it returns real Google results
    with no event-loop conflicts, no rate-limiting surprises, and proper structured output.
    Requires SERPER_API_KEY in the .env file (free tier: 2,500 searches/month at serper.dev).
    Falls back gracefully if the search fails.
    """
    serper_key = os.getenv("SERPER_API_KEY", "")
    if not serper_key:
        return "No internet context (SERPER_API_KEY not set in .env)."

    for attempt in range(max_retries):
        try:
            search = GoogleSerperAPIWrapper(
                serper_api_key=serper_key,
                k=8,  # Number of results to return
            )
            result = search.run(search_query)
            if result and len(result.strip()) >= 30:
                return result
        except Exception as e:
            print(f"[Search Attempt {attempt + 1} Failed] {type(e).__name__}: {e}")

    return "No internet context found (search failed)."
