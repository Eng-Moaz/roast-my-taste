import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from core.utils import format_items_for_prompt

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

Always write in English. Keep the whole reply under 200 words.\
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

# We can safely use 3.3 and high temperature again because we aren't using the Agent tool-calling API!
_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.8,
    max_tokens=512,
    api_key=_groq_api_key,
)

_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    MessagesPlaceholder("history"),
    ("human", "User says: {user_input}\n\nInternet Context (use this to make your roast factually accurate, but act like you already knew it): {search_context}"),
])

_base_chain = _prompt | _llm

_chain_with_history = RunnableWithMessageHistory(
    _base_chain,
    _get_session_history,
    input_messages_key="user_input",
    history_messages_key="history",
)


def get_roast_response(user_input: str, roasted_items: list[str], session_id: str) -> str:
    # We do the search manually before asking the LLM. This guarantees 100% stability.
    search_tool = DuckDuckGoSearchRun()
    try:
        search_context = search_tool.invoke(user_input)
    except Exception:
        search_context = "No internet context found."

    formatted_items = format_items_for_prompt(roasted_items)

    response = _chain_with_history.invoke(
        {
            "user_input": user_input,
            "search_context": search_context[:1000], # Keep it brief
            "roasted_items": formatted_items,
        },
        config={"configurable": {"session_id": session_id}},
    )

    return response.content


def clear_session_history(session_id: str) -> None:
    if session_id in _chat_histories:
        del _chat_histories[session_id]