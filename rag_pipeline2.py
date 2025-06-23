import os
import getpass
from typing import List, Tuple
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import atexit
from datasets import load_dataset
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests


OPTION_LETTERS = ["A", "B", "C", "D"]

# --- Configuration ---
class Config:
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY") or getpass.getpass("Enter Mistral API key: ")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or getpass.getpass("Enter Pinecone API key: ")
    INDEX_NAME = "rag-owasp"
    EMBED_MODEL = "mistral-embed"
    MODEL_NAME = "mistral-large-latest"
    REGION = "us-east-1"


# --- Client Initialization ---
class RAGClients:
    def __init__(self, config: Config):
        self.mistral = MistralClient(api_key=config.MISTRAL_API_KEY)
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.index = self._init_index(config)
        if hasattr(self.mistral, "client") and hasattr(self.mistral.client, "close"):
            atexit.register(self.mistral.client.close)

    def _init_index(self, config: Config):
        spec = ServerlessSpec(cloud="aws", region=config.REGION)
        if config.INDEX_NAME not in [i["name"] for i in self.pc.list_indexes()]:
            dims = len(self.mistral.embeddings(model=config.EMBED_MODEL, input=["test"]).data[0].embedding)
            self.pc.create_index(config.INDEX_NAME, dimension=dims, metric="dotproduct", spec=spec)
        return self.pc.Index(config.INDEX_NAME)


# --- Load Documents ---

def prepare_dataset_file(file_path="data/source_docs.txt"):
    with open(file_path, "r") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    docs = [Document(page_content=chunk, metadata={"source": "OWASP_HackTricks"}) for chunk in chunks]
    return docs

def prepare_dataset():
    print(f"Fetching Dataset `preemware/pentesting-eval`")
    ds = load_dataset("preemware/pentesting-eval", split="train")
    return ds.map(lambda x, idx: {
        "id": f"q{idx}",
        "metadata": {"question": x["question"], "choices": x["choices"]},
        "text": x["question"] + "\n" + "\n".join(x["choices"])
    }, with_indices=True)



def prepare_dataset_from_web(url="https://owasp.org/www-project-top-ten/"):
    print(f"Fetching OWASP content from: {url}")
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract main text content
    paragraphs = soup.find_all(["p", "li", "h2", "h3"])
    content = "\n".join([p.get_text(strip=True) for p in paragraphs])

    # Chunk it for RAG
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(content)

    # Create document objects
    docs = [Document(page_content=chunk, metadata={"source": "OWASP_Web"}) for chunk in chunks]
    return docs
def prepare_dataset_evaluation():
    ds = load_dataset("preemware/pentesting-eval", split="train")
    return ds.map(lambda x, idx: {
        "id": f"q{idx}",
        "metadata": {"question": x["question"], "choices": x["choices"]},
        "text": x["question"] + "\n" + "\n".join(x["choices"])
    }, with_indices=True)

# --- Embedding and Upserting ---
def embed_texts(mistral: MistralClient, model: str, texts: List[str]) -> List[List[float]]:
    return [e.embedding for e in mistral.embeddings(model=model, input=texts).data]


def upsert_documents(index, mistral: MistralClient, docs, embed_model: str):
    texts = [doc.page_content for doc in docs]
    vectors = embed_texts(mistral, embed_model, texts)
    ids = [f"chunk-{i}" for i in range(len(docs))]
    metas = [doc.metadata for doc in docs]
    index.upsert(vectors=list(zip(ids, vectors, metas)))


# --- Chatbot Logic ---
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
        return [match["metadata"].get("source", "") + ": " + match["metadata"].get("text", match["id"]) for match in res["matches"]]

    def format_prompt(self, context_docs: List[str], current_query: str) -> str:
        history_str = "\n".join([f"User: {u}\nAssistant: {a}" for u, a in self.chat_history])
        context_str = "\n- ".join(context_docs)
        prompt = f"""
    You are a senior penetration tester. Use best practices from OWASP and HackTricks.

    Conversation History:
    {history_str if history_str else "None"}

    Relevant Knowledge:
    - {context_str}

    Current User Question:
    {current_query}

    Answer:"""
        return prompt

    def ask(self, user_input: str) -> str:
        context_docs = self.retrieve_docs(user_input)
        prompt = self.format_prompt(context_docs, user_input)

        messages = [
            ChatMessage(role="system", content="You are a cybersecurity expert assistant."),
            ChatMessage(role="user", content=prompt)
        ]

        response = self.mistral.chat(model=self.model_name, messages=messages)
        reply = response.choices[0].message.content
        self.chat_history.append((user_input, reply))  # Save to history
        return reply
    



    
def generate_mc_answer(mistral: MistralClient, question: str, options: List[str]) -> str:
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
    return "?"

# without RAG

    
class LLMChatbot:
    def __init__(self, mistral: MistralClient, model_name: str):
        self.mistral = mistral
        self.model_name = model_name
        self.chat_history = []  # Stores (user_input, bot_response) pairs

    def format_prompt_norag(self, current_query: str) -> str:
        history_str = "\n".join([
            f"User: {u}\nAssistant: {a}"
            for u, a in self.chat_history
        ])
        prompt = f"""
You are a senior penetration tester

Conversation History:
{history_str if history_str else "None"}

Current User Question:
{current_query}

Answer:"""
        return prompt

    def ask_norag(self, user_input: str) -> str:
        prompt = self.format_prompt_norag(user_input)
        messages = [
            ChatMessage(role="system", content="You are a cybersecurity expert."),
            ChatMessage(role="user", content=prompt)
        ]
        response = self.mistral.chat(model=self.model_name, messages=messages)
        reply = response.choices[0].message.content.strip()
        self.chat_history.append((user_input, reply))
        return reply
