import json
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from torch.utils.data import Dataset

# Custom Dataset Class for JSON
class EthicsDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length=8192):
        with open(json_file, "r") as f:
            self.data = json.load(f)

        self.texts = []
        for key, value in self.data.items():
            content = value.get("Content", "")
            title = value.get("Title:", "Untitled Policy")
            link = value.get("URL", "")
            if content:
                entry = f"{title}\n\n{content}\n\nReference: {link}\n"
                self.texts.append(entry)

        self.examples = []
        for text in self.texts:
            tokenized = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            self.examples.append(tokenized["input_ids"].squeeze(0))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# Model and tokenizer setup
model_id = "neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w4a16"
max_model_len = 8192  # Max model input length

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Ensure pad_token is set

# Initialize the dataset
dataset = EthicsDataset("backend/ethics.json", tokenizer, max_length=max_model_len)

# vLLM backend initialization
llm = LLM(model=model_id, tensor_parallel_size=1, max_model_len=max_model_len)

# Training simulation
print("Starting training...")
for i, input_ids in enumerate(dataset):
    # Simulate a training step
    text = tokenizer.decode(input_ids, skip_special_tokens=True)
    print(f"Training on example {i + 1}/{len(dataset)}: {text[:200]}...")  # Print a sample for debugging
    # Here you can add code for fine-tuning using a custom training loop if needed.

print("Training complete. Saving the model...")
llm.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")