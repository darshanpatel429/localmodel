import os
import sys
import json
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# Load the tokenizer (absolute path)
tokenizer = AutoTokenizer.from_pretrained(os.path.abspath("./trained_model"))
tokenizer.pad_token = tokenizer.eos_token  # Ensure pad_token is set

# Load the ONNX model (absolute path)
session = ort.InferenceSession(
    os.path.abspath("./trained_model/onnx/model.onnx"),
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]  # Adjust based on environment
)

def generate_response(input_text):
    # Tokenize the input
    inputs = tokenizer(
        input_text,
        return_tensors="np",  # NumPy arrays for ONNX Runtime
        padding=True,
        truncation=True
    )

    # Prepare the ONNX inputs
    onnx_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }

    # Add placeholders for past_key_values
    for i in range(32):  # Assuming 32 transformer layers for LLaMA
        onnx_inputs[f"past_key_values.{i}.key"] = np.zeros((1, 32, 0, 128), dtype=np.float32)  # Adjust size as needed
        onnx_inputs[f"past_key_values.{i}.value"] = np.zeros((1, 32, 0, 128), dtype=np.float32)  # Adjust size as needed

    # Generate position IDs (optional)
    onnx_inputs["position_ids"] = np.arange(inputs["input_ids"].shape[1], dtype=np.int64).reshape(1, -1)

    # Run inference
    outputs = session.run(None, onnx_inputs)

    # Decode the output logits
    logits = outputs[0]
    predicted_ids = np.argmax(logits, axis=-1)
    response = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

    return response

if __name__ == "__main__":
    # Read input from stdin
    input_data = sys.stdin.read()
    try:
        # Parse JSON input
        data = json.loads(input_data)
        user_query = data.get("message", "")

        # Generate response
        response = generate_response(user_query)

        # Return output as JSON
        result = {"response": response}
        print(json.dumps(result))  # Output JSON
        sys.stdout.flush()
    except Exception as e:
        # Handle errors gracefully
        error_result = {"error": str(e)}
        print(json.dumps(error_result))
        sys.stdout.flush()