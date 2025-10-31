# RAG

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/xidic81/rag/blob/main/ragqdrant.ipynb)

📚 RAG: Giving AI a Search Engine

Imagine a really smart computer program called a Large Language Model (LLM)—like the engine behind ChatGPT.

The Problem with Simple AI Chatbots

These smart programs are great, but they have two main problems:

    Their Knowledge is Frozen: They only know what they were trained on, which might be years old. They can't tell you about last week's news or details about your company's latest products.

    They Guess: If you ask a question they haven't seen before, they sometimes make things up (this is called "hallucination").

Example: If you ask a standard chatbot, "What is my mother's name?" it can't answer, because your mother's name wasn't in its public training data.

How RAG Fixes This

RAG (which stands for Retrieval-Augmented Generation) is a powerful method that gives the AI a private search engine so it can look things up before it answers.

It works by combining two main parts:

    The Retriever (The Searcher): This part has access to a huge library of your specific, up-to-date information (like your company documents, new articles, or your personal files). When you ask a question, the Retriever quickly finds the most relevant information from that library.

    The Generator (The Chatbot): This is the LLM itself. It takes your original question AND the retrieved information (the relevant facts it just looked up) and uses only those facts to generate a complete, accurate answer.

The result? The chatbot doesn't guess, and it can answer questions based on the latest or most private knowledge you give it.

🧠 Building Your Simple RAG System

To build a basic RAG system, we need three main tools that work together:

1. The Key Components

Component	Simple Description	What It Does
Embedding Model	The "Idea Translator" 💡	This is a special AI that converts plain text (like a sentence) into a long list of numbers called a vector. This vector is like a mathematical fingerprint that captures the meaning of the text.
Vector Database	The "Smart Filing Cabinet" 🗄️	This is a special storage system where we keep all the text and its corresponding vectors. It's designed to search based on these number lists very quickly. (The article will build a basic one, but powerful ones exist, like Qdrant).
Chatbot (LLM)	The "Answer Generator" 🤖	This is the main AI (like a smaller version of Llama or GPT) that reads the retrieved facts and writes the final, easy-to-read answer for the user.

📝 Step 1: The Indexing Phase (Building the Searchable Library)

This is the one-time preparation step where you get your knowledge ready to be searched.

How it Works:

    Break it Down (Chunking): You start with your big collection of information (your dataset). You break it down into small, digestible pieces, like single paragraphs or sentences. These small pieces are called chunks.

    Translate to Math (Vectorizing): You feed each of these small chunks into the Embedding Model. This model turns the text into its numerical vector.

    Store the Map: You then save both the original text chunk and its new vector into the Vector Database.

Why Vectors are Smart

Instead of storing knowledge in a way that requires exact keyword matching (like a simple Ctrl+F search), the Vector Database stores the meaning as numbers.
Chunk (The Original Text)	Embedding Vector (The Math Fingerprint of its Meaning)
Italy and France produce over 40% of all wine in the world.	[0.1, 0.04, -0.34, 0.21, ...]
The Taj Mahal in India is made entirely out of marble.	[-0.12, 0.03, 0.9, -0.1, ...]

When a user asks a question, the system turns the question into a vector, and then asks the database: "Which stored vectors are closest to this question vector?" Closeness in math means similarity in meaning.

To find out how "close" two vectors are, we use a technique called Cosine Similarity. Don't worry about the formula—just know that the higher the similarity score, the more relevant the stored information is to the user's question!

🔎 Step 2: The Retrieval Phase (Finding the Answer)

This is the moment the RAG system actually answers a question.

    Question to Vector: You type a question (your Input Query). Just like with the knowledge chunks, the Embedding Model instantly turns your question into a Query Vector (a list of numbers representing the question's meaning).

    Smart Search: The system takes this Query Vector and checks it against all the vectors stored in the Vector Database (Qdrant). It finds the chunks whose vectors are the closest match to your question vector.

    Top Facts Found: The database hands back the Top N (usually 3 or 5) most relevant facts, along with their high similarity scores. These few facts are the crucial Retrieved Knowledge that the final chatbot will use.

    💻 Let's Code It! (Using Python and Qdrant)

We'll write a simple Python script using Qdrant and a placeholder for the text-to-vector conversion.

1. Setup and Preparation

Just install the Qdrant client:
Bash

Install the Qdrant client for the Vector Database
pip install qdrant-client

2. Loading the Dataset

We'll use a placeholder function, emb_model, to simulate the job of the Embedding Model. For a real RAG system, you would replace this with calls to an API (like OpenAI, Cohere) or a local library (like Hugging Face Transformers).

```python
from FlagEmbedding import FlagModel
from qdrant_client import QdrantClient, models
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


emb_model = FlagModel('BAAI/bge-large-en', use_fp16=True)

client = QdrantClient(
    url="https://35628111-12ca-49e6-b072-...",  # use your Qdrant Cloud URL
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6Ikp...",  # your API key if using Qdrant Cloud
)
print(client.get_collections())
--- Load dataset ---
dataset = []
with open('cat-facts.txt', 'r') as file:
    dataset = [line.strip() for line in file if line.strip()]
print(f'Loaded {len(dataset)} entries')

vector_size = len(emb_model.encode("test"))  # get embedding dimension
```
3. Implement the Vector Database (with Qdrant)

This section initializes Qdrant and implements the Indexing Phase by using the mock embedding function to load the data.
```Python
# --- Create Qdrant collection ---
collection_name = "cat_facts"
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=vector_size,
        distance=models.Distance.COSINE
    )
)

# --- Insert text chunks into Qdrant ---
points = []
for idx, chunk in enumerate(tqdm(dataset, desc="Embedding & uploading")):
    embedding = emb_model.encode(chunk)
    points.append(
        models.PointStruct(
            id=idx,  # unique ID for each vector
            vector=embedding,
            payload={"text": chunk}
        )
    )

# Upload all vectors in one batch
client.upsert(
    collection_name=collection_name,
    points=points
)

print(f"✅ Successfully added {len(points)} chunks to Qdrant!")

model_name = "deepseek-ai/deepseek-llm-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda:0")
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

messages = [
    {"role": "user", "content": "Who are you?"}
]
input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
outputs = model.generate(input_tensor.to(model.device), max_new_tokens=100)

result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
print(result)

query = "Why do cats purr?"
query_vector = emb_model.encode(query)

results = client.search(
    collection_name=collection_name,
    query_vector=query_vector,
    limit=3
)

print("\n🔍 Top results:")
for hit in results:
    print(f"Score: {hit.score:.3f} | Text: {hit.payload['text']}")
```
🏃 Step 4: Implement the Retrieval Function

This function handles the Retrieval Phase by taking a user query, turning it into a vector (again, using the mock function), and searching Qdrant.
```python
# --- Optional: Query example ---
def retrieve(query, top_n=3):
  query = "Why do cats purr?"
  query_vector = emb_model.encode(query)

  results = client.search(
    collection_name=collection_name,
    query_vector=query_vector,
    limit=3
  )

  print("\n🔍 Top results:")
  for hit in results:
    print(f"Score: {hit.score:.3f} | Text: {hit.payload['text']}")
  return results
```
💬 Step 5: Generation Phase (The Final Answer)

In a real RAG application, you would now use the retrieved facts to build a prompt for your Language Model (LLM), which would then generate the final, informed answer.
```python
input_query = input('Ask me a question: ')
retrieved_knowledge = retrieve(input_query)

for hit in retrieved_knowledge:
    print(f"Score: {hit.score:.3f} | Text: {hit.payload['text']}")

text = f"""
{" ".join([f'{hit.payload['text']}. ' for hit in retrieved_knowledge])}
.Think the above information are the only knowledge you have, don't add any additional information, if the information above not sufficient answer 'I am sorry I don't know', now respond to this : {input_query}. Do not add something like 'Based on the information provided'
"""
messages = [
    {"role":"user","content":"Jakarta is the capital city of Indonesia. Think the above information are the only knowledge you have, don't add any additional information, if the information above not sufficient answer 'I am sorry I don't know', now respond to this : 'What is capital city of Indonesia'. Do not add something like 'Based on the information provided"},
    {"role": "assistant", "content": "The capital city of Indonesia is Jakarta"},
    {"role":"user","content":"Dragon is a mythological animal from some countries like China and Japan. Think the above information are the only knowledge you have, don't add any additional information, if the information above not sufficient answer 'I am sorry I don't know', now respond to this : 'What is capital city of Japan?'. Do not add something like 'Based on the information provided"},
    {"role": "assistant", "content": "I am sorry, I don't know about that"},
    {"role": "user", "content": text}
]
input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
outputs = model.generate(input_tensor.to(model.device), max_new_tokens=100)

result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
print("THE OUTPUT FROM THE CHATBOT : ")
print(result)
```
![Demonstration of the RAG Qdrant](gif/raggg.gif)


## 📚 Article Sources / References

This RAG (Retrieval-Augmented Generation) project is inspired by and references the following implementation guide:

* **[Code a simple RAG from scratch]** by **Ngxson**.
  Available at: [Hugging Face Blog](https://huggingface.co/blog/ngxson/make-your-own-rag)
