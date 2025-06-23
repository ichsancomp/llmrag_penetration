import os
import getpass
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# --- Configuration ---
API_KEY = os.getenv("MISTRAL_API_KEY") or getpass.getpass("Enter Mistral API key: ")
MODEL_NAME = "mistral-large-latest"
mistral = MistralClient(api_key=API_KEY)

class LLMChatbot:
    def __init__(self, mistral: MistralClient, model_name: str):
        self.mistral = mistral
        self.model_name = model_name
        self.chat_history = []  # Stores (user_input, bot_response) pairs

    def format_prompt(self, current_query: str) -> str:
        history_str = "\n".join([
            f"User: {u}\nAssistant: {a}"
            for u, a in self.chat_history
        ])
        prompt = f"""
You are a senior penetration tester. Use best practices from OWASP and HackTricks to answer clearly and professionally.

Conversation History:
{history_str if history_str else "None"}

Current User Question:
{current_query}

Answer:"""
        return prompt

    def ask(self, user_input: str) -> str:
        prompt = self.format_prompt(user_input)
        messages = [
            ChatMessage(role="system", content="You are a cybersecurity expert."),
            ChatMessage(role="user", content=prompt)
        ]
        response = self.mistral.chat(model=self.model_name, messages=messages)
        reply = response.choices[0].message.content.strip()
        self.chat_history.append((user_input, reply))
        return reply


# --- Interactive Chat Loop ---
if __name__ == "__main__":
    print("🧠 LLM-Only Cybersecurity Chatbot (Structured Prompt)\nType 'exit' to quit.\n")
    bot = LLMChatbot(mistral, MODEL_NAME)

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        answer = bot.ask(user_input)
        print("Bot:", answer)
