from core.bot import get_roast_response
import os

def main():
    # Make sure the API key is loaded
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY is not set in your .env file!")
        print("Please add it and try again.")
        return

    print("*** Welcome to The Pretentious Critic! ***")
    print("Tell me what you like, and prepare to be judged.")
    print("(Type 'quit' or 'exit' to leave)\n")
    print("-" * 50)
    
    session_id = "cli-test-session"
    roasted_items = []
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            print("\nCritic: Finally. I couldn't take much more of your taste anyway. *sighs loudly*")
            break
            
        if not user_input.strip():
            continue
            
        try:
            # We call the bot from core/bot.py
            response = get_roast_response(user_input, roasted_items, session_id)
            print(f"\nCritic:\n{response}")
            
            # Keep track of what they told us so the bot can remember
            roasted_items.append(user_input)
            
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()
