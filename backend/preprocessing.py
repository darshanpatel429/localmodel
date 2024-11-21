import json
from sentence_transformers import SentenceTransformer
import pickle

# Load the JSON data
with open("backend/ethics.json", "r") as file:
    data = json.load(file)

# Prepare text chunks for embedding
texts = []
metadata = []

for key, policy in data.items():
    title = policy.get("Title:", "Unknown Title")
    url = policy.get("URL", "No URL")
    content = policy.get("Content", "")

    # Split content into smaller chunks (if lengthy)
    chunks = content.split("\n")
    for chunk in chunks:
        if chunk.strip():  # Ignore empty lines
            texts.append(chunk.strip())
            metadata.append({"title": title, "url": url})

# Create embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, show_progress_bar=True)

# Save embeddings and metadata
with open("embeddings.pkl", "wb") as f:
    pickle.dump((embeddings, texts, metadata), f)