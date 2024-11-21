from sentence_transformers import SentenceTransformer, util
import pickle
import torch

# Load the same SentenceTransformer model used in preprocessing
model = SentenceTransformer('all-MiniLM-L6-v2')  # Replace with the actual model name

# Load embeddings and metadata
with open("./trained_model/embeddings.pkl", "rb") as f:
    embeddings, texts, metadata = pickle.load(f)
    embeddings = torch.tensor(embeddings, dtype=torch.float32)  # Convert to PyTorch FloatTensor

# Choose device: GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move embeddings to the chosen device
embeddings = embeddings.to(device)
model = model.to(device)  # Ensure the model also uses the same device

def retrieve_response(query):
    # Generate the embedding for the query using the SentenceTransformer model
    query_embedding = model.encode(query, convert_to_tensor=True, device=device)

    # Compute cosine similarities
    scores = util.cos_sim(query_embedding, embeddings)[0]
    top_idx = scores.argmax()

    # Threshold for unknown responses
    if scores[top_idx] >= 0.4:
        return f"{texts[top_idx]}\n\nReference: {metadata[top_idx]['url']}"
    else:
        return "I'm sorry, I don't have enough information to answer that."


if __name__ == "__main__":
    import sys
    import json

    input_data = sys.stdin.read()
    try:
        # Parse the JSON input
        data = json.loads(input_data)
        user_query = data.get("message", "")

        # Generate a response
        response = retrieve_response(user_query)

        # Output the response as JSON
        print(json.dumps({"response": response}))
    except Exception as e:
        print(json.dumps({"error": str(e)}))