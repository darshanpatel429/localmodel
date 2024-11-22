import torch
import ollama
import json
import os
import sys

# ANSI escape codes for colors
CYAN = '\033[96m'
RESET_COLOR = '\033[0m'

# Utility for logging to stderr
def log_debug(message):
    print(message, file=sys.stderr)

# Load the ethics.json file
def load_ethics_json(filepath):
    log_debug("Loading ethics.json...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    log_debug(f"Loaded {len(data)} entries from ethics.json.")
    return data

# Generate embeddings for the ethics content
def generate_embeddings(ethics_data):
    log_debug("Generating embeddings for ethics content...")
    embeddings = []
    content_list = []

    for i, (key, value) in enumerate(ethics_data.items()):
        content = value.get("Content", "")
        title = value.get("Title:", "Untitled Policy")
        link = value.get("URL", "")
        if content:
            entry = f"{title}\n\n{content}\n\nReference: {link}"
            content_list.append(entry)
            try:
                log_debug(f"Generating embedding for entry {i+1}/{len(ethics_data)}: {title}")
                response = ollama.embeddings(model='mxbai-embed-large', prompt=entry)
                embeddings.append(response["embedding"])
            except Exception as e:
                log_debug(f"Error generating embedding for entry {i+1}: {e}")
                continue  # Skip this entry if there's an error

    log_debug(f"Generated embeddings for {len(embeddings)} entries.")
    return torch.tensor(embeddings), content_list

# Get relevant context for the user query
def get_relevant_context(user_input, embeddings, content_list, top_k=3):
    log_debug("Fetching relevant context for the user query...")
    if embeddings.nelement() == 0:  # Check if embeddings exist
        log_debug("No embeddings available.")
        return []
    try:
        input_embedding = ollama.embeddings(model='mxbai-embed-large', prompt=user_input)["embedding"]
        cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), embeddings)
        top_indices = torch.topk(cos_scores, k=min(top_k, len(cos_scores)))[1].tolist()
        log_debug(f"Top {top_k} relevant contexts identified.")
        return [content_list[idx] for idx in top_indices]
    except Exception as e:
        log_debug(f"Error fetching relevant context: {e}")
        return []

# Chat function to handle the response
def ollama_chat(user_input, embeddings, content_list, system_message, ollama_model="mistral"):
    log_debug(f"Processing chat query: {user_input}")
    relevant_context = get_relevant_context(user_input, embeddings, content_list)
    context_str = "\n".join(relevant_context) if relevant_context else "No relevant context found."

    log_debug(CYAN + "Relevant Context Found:\n" + context_str + RESET_COLOR)

    input_with_context = user_input + f"\n\nRelevant Context:\n{context_str}" if relevant_context else user_input
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": input_with_context}]

    try:
        response = ollama.chat(
            model=ollama_model,
            messages=messages
        )
        if hasattr(response, 'message') and hasattr(response.message, 'content'):
            return response.message.content.strip()
        else:
            log_debug(f"Unexpected response structure: {response}")
            return "Unable to generate a response."
    except Exception as e:
        log_debug(f"Error generating chat response: {e}")
        return "An error occurred while generating the response."

# Initialization
log_debug("Initializing...")
ethics_filepath = "./backend/ethics.json"  # Path to ethics.json
try:
    ethics_data = load_ethics_json(ethics_filepath)
    vault_embeddings, vault_content = generate_embeddings(ethics_data)
    log_debug("Initialization complete.")
except Exception as e:
    log_debug(f"Error during initialization: {e}")
    vault_embeddings = torch.tensor([])
    vault_content = []

system_message = "You are an assistant expert at providing answers to user questions based on the given ethics document."

# API endpoint function for server.js
def handle_query(query):
    log_debug(f"Received query: {query}")
    response = ollama_chat(query, vault_embeddings, vault_content, system_message)
    log_debug(f"Response generated: {response}")
    return {"response": response}

# Main execution for debugging
if __name__ == "__main__":
    try:
        input_data = sys.stdin.read()
        query_data = json.loads(input_data)
        query = query_data.get("query", "")

        if not query:
            print(json.dumps({"error": "No query provided."}))
        else:
            log_debug(f"Received query: {query}")
            result = handle_query(query)
            print(json.dumps(result))  # Ensure JSON output
    except Exception as e:
        print(json.dumps({"error": str(e)}))