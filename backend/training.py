import pickle
from transformers import AutoTokenizer

# Load embeddings and metadata
with open("embeddings.pkl", "rb") as f:
    embeddings, texts, metadata = pickle.load(f)

# Define the model and tokenizer
model_id = "neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w4a16"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Save the tokenizer and embeddings for inference
tokenizer.save_pretrained("./trained_model")
with open("./trained_model/embeddings.pkl", "wb") as f:
    pickle.dump((embeddings, texts, metadata), f)

print("Training complete. Model, tokenizer, and embeddings saved.")