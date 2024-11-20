from transformers import GPT2Tokenizer, GPT2LMHeadModel
from optimum.onnxruntime import ORTModelForCausalLM
import os

# Paths for your trained model and ONNX export
model_path = "./trained_model"
onnx_path = "./trained_model/onnx/"

# Ensure ONNX directory exists
os.makedirs(onnx_path, exist_ok=True)

# Load the fine-tuned GPT-2 model
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Set pad_token to eos_token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Resize the embeddings if pad_token was added
model.resize_token_embeddings(len(tokenizer))

# Export the model to ONNX format
onnx_model = ORTModelForCausalLM.from_pretrained(
    model_path,
    from_transformers=True,
    provider="CPUExecutionProvider",  # Add provider explicitly if needed
)
onnx_model.save_pretrained(onnx_path)

print(f"ONNX model exported to {onnx_path}")