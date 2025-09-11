from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

model_name = "meta-llama/Llama-3.2-3B-Instruct"

# Export to ONNX
ort_model = ORTModelForCausalLM.from_pretrained(
    model_name,
    export=True,
    use_auth_token=True  # You'll need HF access token for Llama models
)

# Save the exported model
ort_model.save_pretrained("./llama-3.2-3b-instruct-onnx")

# Also save the tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.save_pretrained("./llama-3.2-3b-instruct-onnx")