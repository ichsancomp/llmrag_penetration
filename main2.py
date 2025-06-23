from rag_pipeline2 import Config, RAGClients, RAGChatbot, LLMChatbot, prepare_dataset, upsert_documents, prepare_dataset_evaluation, prepare_dataset_from_web
from evaluate_utils import evaluate_model,print_sample_results, show_classification_report, save_correct_predictions, print_correct_predictions


cfg = Config()
clients = RAGClients(cfg)
llm = LLMChatbot(clients.mistral, cfg.MODEL_NAME)
# Prepare documents from OWASP / HackTricks
#ds = prepare_dataset()
ds = prepare_dataset_from_web()
ds_eval = prepare_dataset_evaluation()

# Upsert embeddings to Pinecone
upsert_documents(clients.index, clients.mistral, ds, cfg.EMBED_MODEL)

# Create RAG chatbot
bot = RAGChatbot(clients.mistral, clients.index, cfg.MODEL_NAME, cfg.EMBED_MODEL)
# Export training data (optional)
#export_fine_tune_data(ds)


# Get step-by-step explanation
#print("\nStep-by-step Reasoning:")
#print(generate_step_by_step_answer(clients.mistral, cfg.MODEL_NAME))

# Evaluate performance (dummy since no ground-truth)

# Ask chatbot a question with RAG


question = "What is privilege escalation?"
print("Bot Answer:", bot.ask(question))
while True:
        
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        reply = bot.ask(user_input)
        print("Bot:", reply)



"""
print("Running evaluation on preemware/pentesting-eval")
eval_ds = prepare_dataset_evaluation()
accuracy, results = evaluate_model(eval_ds, clients.mistral)

print(f"\n Evaluation Accuracy: {accuracy*100:.2f}%")

# Optional: print some results
for r in results[:10]:
    print("Q:", r["question"])
    print("A:", r["choices"])
    print("True:", r["true_answer"], "| Pred:", r["pred_answer"], "| ✅" if r["correct"] else "| ❌")
    print("-" * 50)

accuracy, results = evaluate_model(ds_eval, clients.mistral)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Reporting
print_sample_results(results)
show_classification_report(results)
correct_results = save_correct_predictions(results)
print_correct_predictions(correct_results)
"""