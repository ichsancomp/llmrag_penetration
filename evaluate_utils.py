from typing import List, Tuple
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from datasets import Dataset
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import json

OPTION_LETTERS = ["A", "B", "C", "D"]
MODEL_NAME = "mistral-large-latest"

def generate_mc_answer(mistral: MistralClient, question: str, options: List[str]) -> str:
    """
    Sends a multiple-choice question to Mistral model and parses the selected option (A-D).
    """
    prompt = f"Answer the following multiple-choice question by choosing A, B, C, or D.\n\nQuestion:\n{question}\n\nOptions:\n"
    for letter, option in zip(OPTION_LETTERS, options):
        prompt += f"{letter}. {option}\n"
    prompt += "\nAnswer:"

    messages = [
        ChatMessage(role="system", content="You are an expert cybersecurity assistant."),
        ChatMessage(role="user", content=prompt)
    ]

    response = mistral.chat(model=MODEL_NAME, messages=messages)
    answer = response.choices[0].message.content.strip().upper()

    for letter in OPTION_LETTERS:
        if letter in answer:
            return letter
    return ""

def evaluate_model2(dataset: Dataset, mistral: MistralClient) -> Tuple[float, List[dict]]:
    """
    Evaluates the model on a dataset of multiple-choice questions.
    Returns the accuracy and the detailed result list.
    """
    correct = 0
    results = []

    for item in tqdm(dataset, desc="Evaluating"):
        question = item["question"]
        options = item["choices"]
        true_answer = OPTION_LETTERS[item["answer"]]

        pred_answer = generate_mc_answer(mistral, question, options)
        is_correct = (pred_answer == true_answer)
        correct += int(is_correct)

        results.append({
            "question": question,
            "choices": options,
            "true_answer": true_answer,
            "pred_answer": pred_answer,
            "correct": is_correct
        })

    accuracy = correct / len(dataset)
    return accuracy, results

def evaluate_model(dataset: Dataset, mistral: MistralClient) -> Tuple[float, List[dict]]:
    correct = 0
    results = []

    for item in tqdm(dataset, desc="Evaluating"):
        question = item["question"]
        options = item["choices"]
        true_answer = OPTION_LETTERS[item["answer"]]  # Convert int → letter

        pred_answer = generate_mc_answer(mistral, question, options)  # Should return "A"–"D"
        is_correct = (pred_answer == true_answer)
        correct += int(is_correct)

        results.append({
            "question": question,
            "choices": options,
            "true_answer": true_answer,
            "pred_answer": pred_answer,
            "correct": is_correct
        })

    accuracy = correct / len(dataset)
    return accuracy, results


def print_sample_results(results: List[dict], n: int = 50):
    print(f"\nSample results (first {n}):")
    for r in results[:n]:
        print("Question:", r["question"])
        print("Options:", r["choices"])
        print("True Answer:", r["true_answer"])
        print("Predicted:", r["pred_answer"])
        print("Correct" if r["correct"] else "Incorrect")
        print()

def show_classification_report(results: List[dict]):
    y_true = [r["true_answer"] for r in results]
    y_pred = [r["pred_answer"] for r in results]

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=OPTION_LETTERS))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=OPTION_LETTERS))

def save_correct_predictions(results: List[dict], filename: str = "correct_predictions.json"):
    correct_results = [r for r in results if r["correct"]]
    with open(filename, "w") as f:
        json.dump(correct_results, f, indent=2)

    print(f"Saved {len(correct_results)} correct predictions to '{filename}'")
    return correct_results

def print_correct_predictions(correct_results: List[dict]):
    for r in correct_results:
        print("Question:", r["question"])
        for i, opt in enumerate(r["choices"]):
            print(f"  {chr(65+i)}. {opt}")
        print("True Answer:", r["true_answer"], "| Predicted:", r["pred_answer"])
        print("-" * 60)
