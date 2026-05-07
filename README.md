# 🎭 The Critic

**The Critic** is an elitist, snobbish, and highly sarcastic AI chatbot that aggressively judges your taste in pop culture—movies, music, food, or whatever "crimes" you confess to. Built for an IEEE competition, this project features a modern dark-themed UI with custom CSS, LangChain integration for conversational memory, and real-time DuckDuckGo search integration to ensure the bot stays up-to-date with the latest internet trends and celebrity gossip.

---

## ✨ Features

- **Character Persona**: A dedicated, dramatic "Pretentious Critic" personality powered by LLaMA 3.3 70B via Groq.
- **Internet Search Integration**: Automatically runs Google searches in the background via the **Serper API** to roast you accurately on current events (e.g., recent Coachella drama, new releases, viral TikTok songs).
- **Persistent Memory**: Remembers everything you confess in a session and brings it up later to enhance the joke.
- **Dynamic UI**: A highly customized Streamlit interface featuring:
  - Custom dark mode, typography, and layout.
  - Interactive "Pill" buttons for quick confessions.
  - Polished chat bubbles with typing animations.
- **Robust State Management**: Clean history management ensures search context is fed ephemerally to the LLM without polluting conversational memory, solving multi-turn hallucination issues.

---

## 🛠️ Technology Stack

- **Frontend & UI**: [Streamlit](https://streamlit.io/) with aggressive CSS overrides targeting Streamlit's Shadow DOM and BaseWeb components for a seamless, bracket-free chat input experience.
- **Backend Orchestration**: [LangChain](https://www.langchain.com/) for managing prompts, message history, and tool invocation.
- **LLM Provider**: [Groq](https://groq.com/) (using the `llama-3.3-70b-versatile` model) for lightning-fast, high-quality responses.
- **Search Tool**: [Serper API](https://serper.dev) via LangChain's `GoogleSerperAPIWrapper` — real Google Search results, fully synchronous, no event-loop issues.
- **Dependency Management**: [uv](https://github.com/astral-sh/uv) (configured via `pyproject.toml`).

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- `uv` installed on your machine (`pip install uv`)
- A [Groq API Key](https://console.groq.com/keys)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd roast-my-taste
   ```

2. **Set up your environment variables:**
   Create a `.env` file in the root directory and add your API keys:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   SERPER_API_KEY=your_serper_api_key_here
   ```
   Get your free Serper key (2,500 searches/month) at [serper.dev](https://serper.dev).

3. **Install dependencies and run the app using `uv`:**
   ```bash
   uv run streamlit run app.py
   ```

4. **Open your browser:**
   Navigate to `http://localhost:8502/` (or the port specified by Streamlit) to meet your new worst enemy.

---

## 📂 Project Structure

```text
roast-my-taste/
├── app.py               # Main Streamlit application, UI logic, and CSS.
├── pyproject.toml       # uv dependency configuration.
├── .env                 # Environment variables (not tracked in git).
└── core/
    ├── bot.py           # LangChain logic, prompts, history management, and search integration.
    └── utils.py         # Helper functions for formatting items for the prompt.
```

---

## 🧠 How it Works

1. **The Chat Interface (`app.py`)**: 
   Streamlit captures user input via the chat input box or quick-action "pill" buttons. The UI is completely customized via injected CSS, stripping away default Streamlit decorations for a premium look.
2. **The Smart Search (`bot.py`)**:
   Before querying the LLM, the bot takes the user's input and recent conversation context (using `_build_search_query`) and queries DuckDuckGo. This guarantees the LLM knows about current events even if it wasn't trained on them. Furthermore, if you ask a vague follow-up question (e.g., "what song did *they* play?", or "what did *she* do?"), the bot intelligently tracks pronouns and injects the previous topic's keywords into the search query under the hood to ensure it never loses context.
3. **The LLM Call**:
   The LLM is fed a strict System Prompt enforcing its snobbish personality. The fresh internet search data is injected directly into the prompt *ephemerally*.
4. **Clean Memory**:
   To prevent the LLM from treating old search dumps as established facts, only the raw user input and the AI's direct response are saved to the persistent `InMemoryHistory`.

---

## 📝 Future Improvements

- Add a dedicated tracking metric for "Roast Severity" to dynamically end the chat if the user's taste is deemed *too* terrible.
- Expand quick-prompts to be dynamically generated based on trending Twitter/X topics.
