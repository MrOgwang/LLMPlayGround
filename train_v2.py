import torch
import sentencepiece as spm
import json
from transformers import AutoTokenizer, AutoModel
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig

# Step 1: Load model configuration from params.json
with open("F:/LLM/llama32-IB/params.json", "r") as f:
    config = json.load(f)

print("Model Configuration:", config)  # Optional: Check architecture

model_path = 'F:/LLM/llama32-IB/'

#tokenizer = spm.SentencePieceProcessor(model_file="F:/LLM/llama32-IB/tokenizer.model") #load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model_weights = torch.load("F:/LLM/llama32-IB/consolidated.00.pth", map_location="cpu") #load model weights

print("Loaded model weights.")

# Optional: Explore the loaded state dict (model weights)
print(model_weights.keys())

# Step 4: Tokenize input text
input_text = "What is the capital of Kenya?"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
print("Input IDs:", input_ids)

# load the model with the specific configs. 
model = LlamaForCausalLM.from_pretrained("F:/LLM/llama32-IB/")
model.load_state_dict(model_weights, strict=False)  # Adjust as needed

# Step 6: Run the model with your input
with torch.no_grad():
    output = model(input_ids.float())  # Ensure input type matches your model

print("Output:", output)
