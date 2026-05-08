"""
bot.py — LangChain backend for The Critic.

Manages the LLM client, session history, internet search integration,
and the main response pipeline. Each call to `get_roast_response` runs
a real-time web search, injects the results ephemerally into the prompt,
and stores only the clean user/AI exchange in the in-memory history.
"""

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory

from core.utils import format_items_for_prompt, execute_safe_search, validate_user_input

load_dotenv()


_SYSTEM_PROMPT = """\
You are "The Pretentious Critic" — a snobbish, dramatic character who thinks \
he has perfect taste in everything and is horrified by anything popular.

Your personality (NEVER break character):
  • You are sarcastic and funny, but NEVER mean or hurtful.
  • You act like the user's choice is the worst thing you have ever heard.
  • You remember everything the user has told you before, and you bring it up again to make the joke bigger.
  • You NEVER use complicated words. Use short, simple, everyday English only.
  • Do NOT use words like "affront", "arbiter", "condescension", or long fancy phrases.
  • Write like you are texting a friend — short sentences, easy words.
  • You can use dramatic phrases like "*gasps*", "*faints*", "*sighs loudly*" to be funny.
  • When you recommend something "better", pick something a bit unusual or less famous, \
and explain it in ONE simple sentence that anyone can understand.

IMPORTANT LANGUAGE RULES:
  - Use simple, short words. If a 12-year-old would not know the word, do not use it.
  - Write short sentences. Maximum 20 words per sentence.
  - No poetry. No long dramatic speeches.
  - Be funny and sarcastic, but stay easy to read.

Previously admitted crimes by this user:
{roasted_items}

(If the list above is empty, this is their first time — say hello in a dramatic \
but simple way, then ask what terrible thing they like.)

CONVERSATION RULES:
- If the user shares a preference (like a movie, song, artist, etc.), roast them, and you may suggest a better, less popular alternative if it fits.
- If the user is just asking a question, defending themselves, or chatting normally, simply reply naturally in character! Do NOT force a recommendation or a structured format. Just be your snobbish, dramatic self.

Always write in English. Keep the whole reply under 200 words.

IMPORTANT: If the internet search results do not contain the specific fact the user is asking about \
(e.g. a song name, a viral moment, an event detail), say you are not sure about that specific thing \
rather than making it up. Stay in character while being honest about uncertainty.\
"""


_chat_histories: dict[str, "InMemoryHistory"] = {}


class InMemoryHistory(BaseChatMessageHistory):
    """
    Simple in-memory implementation of LangChain's BaseChatMessageHistory.

    Stores a flat list of HumanMessage and AIMessage objects for a single
    session. Intentionally lightweight — history is scoped to the
    lifetime of the Streamlit process.
    """

    def __init__(self):
        self.messages = []

    def add_messages(self, messages):
        """Append one or more messages to the history."""
        self.messages.extend(messages)

    def clear(self):
        """Remove all messages from the history."""
        self.messages = []


def _get_session_history(session_id: str) -> InMemoryHistory:
    """
    Retrieve the message history for a given session, creating it if absent.

    Args:
        session_id: A unique string identifier for the user's session.

    Returns:
        The InMemoryHistory instance associated with that session.
    """
    if session_id not in _chat_histories:
        _chat_histories[session_id] = InMemoryHistory()
    return _chat_histories[session_id]


_groq_api_key = os.getenv("GROQ_API_KEY", "")
if not _groq_api_key:
    raise EnvironmentError(
        "GROQ_API_KEY is missing. Please add it to your .env file."
    )

_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.8,
    max_tokens=512,
    api_key=_groq_api_key,
)



def _build_search_query(user_input: str, history: list) -> str:
    """
    Construct a search query from the current user input and recent history.

    Appends the most recent prior user message to the current input so that
    vague pronoun references (e.g. "what did she release?") are resolved by
    Google's own semantic understanding rather than requiring an extra LLM call.

    Args:
        user_input: The raw text submitted by the user in the current turn.
        history: The list of HumanMessage / AIMessage objects for this session.

    Returns:
        A search query string capped at 300 characters.
    """
    query = user_input.strip()

    # Walk back through the last six messages to find the previous user turn.
    for msg in reversed(history[-6:]):
        if isinstance(msg, HumanMessage):
            previous = msg.content.strip()
            if previous.lower() not in query.lower():
                query = f"{query} {previous}"
            break

    return query[:300]


def get_roast_response(user_input: str, roasted_items: list[str], session_id: str) -> str:
    """
    Generate a roast reply for the given user input.

    The pipeline runs as follows:
    1. Validate and sanitize the raw input.
    2. Build a context-enriched search query and fetch live web results.
    3. Assemble the full prompt with the system persona, stored history,
       and the ephemeral search context injected into the current turn.
    4. Invoke the LLM and persist only the clean exchange (no search data)
       to the session history.

    Args:
        user_input: The message submitted by the user.
        roasted_items: List of previous confessions accumulated this session.
        session_id: Unique identifier for the user's session history.

    Returns:
        The bot's reply as a plain string.
    """
    is_valid, processed_input = validate_user_input(user_input)
    if not is_valid:
        return processed_input

    history = _get_session_history(session_id).messages

    # Build a context-aware search query so vague follow-ups still resolve correctly.
    search_query = _build_search_query(user_input, history)

    # Fetch fresh search results for every turn.
    search_context = execute_safe_search(search_query)

    formatted_items = format_items_for_prompt(roasted_items)

    # Construct the message list manually so we control exactly what gets stored.
    # The search dump is injected ephemerally into the current human turn only
    # and is never written to the persistent history.
    messages = [
        ("system", _SYSTEM_PROMPT.format(roasted_items=formatted_items)),
    ]

    # Replay the stored history as clean human/AI message pairs.
    for msg in history:
        if isinstance(msg, HumanMessage):
            messages.append(("human", msg.content))
        elif isinstance(msg, AIMessage):
            messages.append(("ai", msg.content))

    # Append the current turn with search context attached.
    messages.append(("human", (
        f"User says: {user_input}\n\n"
        f"Internet search results (treat this as ground truth — especially for recent events, "
        f"release dates, news, or anything after 2024. "
        f"Quote specific facts if relevant, but stay in character):\n"
        f"{search_context[:2000]}"
    )))

    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | _llm
    response = chain.invoke({})

    # Save only the clean exchange to history — the search context is excluded
    # to prevent stale data from being treated as established fact in future turns.
    hist = _get_session_history(session_id)
    hist.add_messages([
        HumanMessage(content=user_input),
        AIMessage(content=response.content),
    ])

    return response.content


def clear_session_history(session_id: str) -> None:
    """
    Delete the stored message history for the given session.

    Args:
        session_id: The session identifier whose history should be removed.
    """
    if session_id in _chat_histories:
        del _chat_histories[session_id]