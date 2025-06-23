from rag_pipeline2 import LLMChatbot
from mistralai.client import MistralClient
mistral = MistralClient(api_key="YnhQ7lYiFLCfKo1HvbZaHvsCHO6yXVig")
bot_llm = LLMChatbot(model_name="mistral-large-latest", mistral=mistral)

while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            break
        elif user_input.lower() == "reset":
            bot_llm.chat_history.clear()
            print(" Memory cleared.\n")
            continue

        answer = bot_llm.ask_norag(user_input)
        print("Bot:", answer)