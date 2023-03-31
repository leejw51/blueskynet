import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Input text
input_text = "Hello, how are you today?"

# Encode the input text
input_ids = tokenizer.encode(input_text, return_tensors="pt")
attention_mask = torch.ones_like(input_ids)
# display attention mask
print("attention_mask=",attention_mask)

# Generate text using the model
output_ids = model.generate(input_ids, max_length=50, do_sample=True, attention_mask=attention_mask)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Display the output text
print(output_text)

