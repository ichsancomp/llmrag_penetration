import os
import getpass
from typing import List, Tuple
from datasets import load_dataset
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from pinecone import Pinecone, ServerlessSpec

OPTION_LETTERS = ["A", "B", "C", "D"]
# --- Configuration ---
class Config:
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY") or getpass.getpass("Enter Mistral API key: ")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or getpass.getpass("Enter Pinecone API key: ")
    INDEX_NAME = "rag-cybersec"
    EMBED_MODEL = "mistral-embed"
    MODEL_NAME = "mistral-large-latest"
    REGION = "us-east-1"


# --- Client Initialization ---
import atexit

class RAGClients:
    def __init__(self, config: Config):
        self.mistral = MistralClient(api_key=config.MISTRAL_API_KEY)
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.index = self._init_index(config)

        # Register graceful shutdown of mistral client
        if hasattr(self.mistral, "client") and hasattr(self.mistral.client, "close"):
            atexit.register(self.mistral.client.close)

    def _init_index(self, config: Config):
        spec = ServerlessSpec(cloud="aws", region=config.REGION)
        if config.INDEX_NAME not in [i["name"] for i in self.pc.list_indexes()]:
            dims = len(self.mistral.embeddings(model=config.EMBED_MODEL, input=["test"]).data[0].embedding)
            self.pc.create_index(config.INDEX_NAME, dimension=dims, metric="dotproduct", spec=spec)
        return self.pc.Index(config.INDEX_NAME)



# --- Dataset Preparation ---
def prepare_dataset():
    ds = load_dataset("preemware/pentesting-eval", split="train")
    return ds.map(lambda x, idx: {
        "id": f"q{idx}",
        "metadata": {"question": x["question"], "choices": x["choices"]},
        "text": x["question"] + "\n" + "\n".join(x["choices"])
    }, with_indices=True)


# --- Embedding and Upsert ---
def embed_texts(mistral: MistralClient, model: str, texts: List[str]) -> List[List[float]]:
    return [e.embedding for e in mistral.embeddings(model=model, input=texts).data]


def upsert_dataset(index, mistral: MistralClient, dataset, embed_model: str, batch_size: int = 32):
    from tqdm.auto import tqdm
    for i in tqdm(range(0, len(dataset), batch_size), desc="Embedding + Upload"):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        texts = batch["text"]
        vectors = embed_texts(mistral, embed_model, texts)
        metas = batch["metadata"]
        ids = [str(x) for x in batch["id"]]
        index.upsert(vectors=list(zip(ids, vectors, metas)))


# --- RAG Chatbot ---
class RAGChatbot:
    def __init__(self, mistral: MistralClient, index, model_name: str, embed_model: str):
        self.mistral = mistral
        self.index = index
        self.model_name = model_name
        self.embed_model = embed_model
        self.chat_history: List[Tuple[str, str]] = []

    def retrieve_docs(self, query: str, top_k: int = 3) -> List[str]:
        query_vec = embed_texts(self.mistral, self.embed_model, [query])[0]
        res = self.index.query(vector=query_vec, top_k=top_k, include_metadata=True)
        return [match["metadata"]["question"] for match in res["matches"]]

    def format_prompt(self, context_docs: List[str], current_query: str) -> str:
        history_str = "\n".join([f"User: {u}\nAssistant: {a}" for u, a in self.chat_history])
        context_str = "\n- ".join(context_docs)
        prompt = f"""
    You are a senior penetration tester. Answer concisely and clearly, using industry best practices and OWASP terminology when needed.

    Conversation History:
    {history_str if self.chat_history else 'None'}

    Relevant Knowledge:
    - {context_str}

    Current User Question:
    {current_query}

    Answer:"""
        return prompt

    def ask(self, user_input: str) -> str:
        retrieved = self.retrieve_docs(user_input)
        prompt = self.format_prompt(retrieved, user_input)

        messages = [
            ChatMessage(role="system", content="You are a cybersecurity expert assistant."),
            ChatMessage(role="user", content=prompt)
        ]
        response = self.mistral.chat(model=self.model_name, messages=messages)
        reply = response.choices[0].message.content
        self.chat_history.append((user_input, reply))

        return reply


# --- Optional: Fine-Tuning Export ---
def export_fine_tune_data(dataset, file_path: str = "fine_tune_data.jsonl"):
    import json
    fine_tune_data = []
    option_letters = ["A", "B", "C", "D"]
    for example in dataset.select(range(240)):
        prompt = example["question"] + "\n" + "\n".join(example["choices"])
        answer = option_letters[example["answer"]] if "answer" in example else ""
        fine_tune_data.append({"prompt": prompt.strip(), "completion": answer})

    with open(file_path, "w") as f:
        for ex in fine_tune_data:
            f.write(json.dumps(ex) + "\n")
from mistralai.models.chat_completion import ChatMessage

def generate_step_by_step_answer2(mistral: MistralClient, model: str = "mistral-large-latest"):
    prompt = """
Think step-by-step. Consider what privilege escalation means in penetration testing, and then decide which answer makes the most sense.

Question:
What is the primary purpose of privilege escalation?

Options:
A. To avoid detection by antivirus software  
B. To gain access to higher-level system functions  
C. To establish persistence on the target system  
D. To create a reverse shell

Let’s think step by step:
"""

    messages = [
        ChatMessage(role="system", content="You are an expert cybersecurity assistant."),
        ChatMessage(role="user", content=prompt)
    ]

    chat_response = mistral.chat(
        model=model,
        messages=messages
    )
    return chat_response.choices[0].message.content


def generate_step_by_step_answer(mistral: MistralClient, model: str = "mistral-large-latest") -> str:
    prompt = """
You are a cybersecurity expert. Answer the following question by reasoning step-by-step, but conclude with a single letter (A, B, C, or D) as your final answer.

Question:
What is the primary purpose of privilege escalation?

Options:
A. To avoid detection by antivirus software  
B. To gain access to higher-level system functions  
C. To establish persistence on the target system  
D. To create a reverse shell

Let's think step by step:
"""

    messages = [
        ChatMessage(role="system", content="You are an expert cybersecurity assistant."),
        ChatMessage(role="user", content=prompt)
    ]

    chat_response = mistral.chat(
        model=model,
        messages=messages
    )

    # Extract and post-process answer
    full_response = chat_response.choices[0].message.content.strip()
    print("Full step-by-step reasoning:\n", full_response)

    # Extract only A-D answer from the last line (if any)
    last_line = full_response.strip().splitlines()[-1].strip().upper()
    for letter in OPTION_LETTERS:
        if last_line.startswith(letter):
            return letter

    # Fallback: Look for A-D anywhere in response (last resort)
    for letter in OPTION_LETTERS:
        if letter in full_response:
            return letter

    return "?"  # Unknown or unclear output