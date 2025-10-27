from FlagEmbedding import FlagModel
from qdrant_client import QdrantClient, models
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


emb_model = FlagModel('BAAI/bge-large-en', use_fp16=True)

client = QdrantClient(
    url="https://35628111-12ca-49e6-b072-55bc62c5744b.europe-west3-0.gcp.cloud.qdrant.io:6333",  # or your Qdrant Cloud URL
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Ef4XkbdC-ewbGLrB8l_yIZPS6Iwz_wJilGysBZkQEoM"                  # or your API key if using Qdrant Cloud
)
print(client.get_collections())
# --- 3. Load dataset ---
dataset = []
with open('cat-facts.txt', 'r') as file:
    dataset = [line.strip() for line in file if line.strip()]
print(f'Loaded {len(dataset)} entries')

# --- 4. Embedding model ---
# Make sure you have emb_model defined
# e.g., from sentence_transformers import SentenceTransformer
# emb_model = SentenceTransformer("all-MiniLM-L6-v2")

vector_size = len(emb_model.encode("test"))  # get embedding dimension

# --- 5. Create Qdrant collection ---
collection_name = "cat_facts"
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=vector_size,
        distance=models.Distance.COSINE
    )
)

# --- 6. Insert text chunks into Qdrant ---
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

# --- 7. Optional: Query example ---
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