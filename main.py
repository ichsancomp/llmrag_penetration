from rag_pipeline import Config, RAGClients, RAGChatbot, prepare_dataset, upsert_dataset, export_fine_tune_data, generate_step_by_step_answer
from evaluate_utils import evaluate_model,print_sample_results, show_classification_report, save_correct_predictions, print_correct_predictions

cfg = Config()
clients = RAGClients(cfg)
ds = prepare_dataset()
upsert_dataset(clients.index, clients.mistral, ds, cfg.EMBED_MODEL)

bot = RAGChatbot(clients.mistral, clients.index, cfg.MODEL_NAME, cfg.EMBED_MODEL)
export = export_fine_tune_data(ds)

#question = "What is privilege escalation?"
#print(bot.ask(question))
#response = bot.ask(question)
#print(f"Q: {question}\nA: {response}")

question = "What is privilege escalation?"
print("Bot Answer:", bot.ask(question))

print("\nStep-by-step Reasoning:")
print(generate_step_by_step_answer(clients.mistral, cfg.MODEL_NAME))


accuracy, results = evaluate_model(ds, clients.mistral)

print(f"Accuracy: {accuracy * 100:.2f}%")

print_sample_results(results)
show_classification_report(results)
correct_results = save_correct_predictions(results)
print_correct_predictions(correct_results)