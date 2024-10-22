import torch
import sentencepiece as spm
import json
from transformers import AutoTokenizer, AutoModel
from custommodel import LlamaModel

directory = "C:/Users/Wycliffe/.llama/checkpoints/Llama3.2-1B-Instruct"

with open(directory + "/params.json", "r") as f:
    config = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(directory)
model_weights = torch.load(directory + "/consolidated.00.pth", map_location="cpu") #load model weights

input_text = "What is the capital of Kenya?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
print("Input IDs:", input_ids)
print(input_ids.shape)

model = LlamaModel(config)
model.load_state_dict(model_weights, strict=False)

with torch.no_grad():
    output = model(input_ids.int())

print("Output:", output)

token_ids = torch.argmax(output, dim=-1)
print("Decoded Token IDs:", token_ids)

answer = tokenizer.decode(token_ids[0].tolist(), skip_special_tokens=True)
print("Model's Response:", answer)