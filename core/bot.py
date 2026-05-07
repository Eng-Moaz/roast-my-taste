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

    def __init__(self):
        self.messages = []

    def add_messages(self, messages):
        self.messages.extend(messages)

    def clear(self):
        self.messages = []


def _get_session_history(session_id: str) -> InMemoryHistory:
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
    Build a search query from the current user input + the previous user message.
    With Serper (Google Search), we don't need LLM-generated queries — Google
    handles semantic understanding natively. We just need to include enough
    context so that pronouns (they, she, he) are resolved by the surrounding words.
    """
    query = user_input.strip()

    # Append the last user message for pronoun resolution context
    for msg in reversed(history[-6:]):
        if isinstance(msg, HumanMessage):
            previous = msg.content.strip()
            if previous.lower() not in query.lower():
                query = f"{query} {previous}"
            break

    return query[:300]


def get_roast_response(user_input: str, roasted_items: list[str], session_id: str) -> str:
    """
    Returns the bot's reply string.
    """
    # Validate the user input first to handle unexpected/wrong inputs gracefully
    is_valid, processed_input = validate_user_input(user_input)
    if not is_valid:
        return processed_input  # Return the graceful fallback message directly

    history = _get_session_history(session_id).messages

    # Build a context-aware search query so vague follow-ups still find results
    search_query = _build_search_query(user_input, history)

    # Search the internet (fresh every turn, using our safe robust wrapper)
    search_context = execute_safe_search(search_query)

    formatted_items = format_items_for_prompt(roasted_items)

    # Build the prompt manually — we manage history ourselves so that ONLY
    # the clean user_input (without search dumps) gets stored in memory.
    messages = [
        ("system", _SYSTEM_PROMPT.format(roasted_items=formatted_items)),
    ]

    # Inject the stored history (clean human/AI pairs only)
    for msg in history:
        if isinstance(msg, HumanMessage):
            messages.append(("human", msg.content))
        elif isinstance(msg, AIMessage):
            messages.append(("ai", msg.content))

    # Current turn: user_input + fresh search context (ephemeral, NOT stored)
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

    # Manually save ONLY the clean exchange (no search dump) to history
    hist = _get_session_history(session_id)
    hist.add_messages([
        HumanMessage(content=user_input),
        AIMessage(content=response.content),
    ])

    return response.content


def clear_session_history(session_id: str) -> None:
    if session_id in _chat_histories:
        del _chat_histories[session_id]