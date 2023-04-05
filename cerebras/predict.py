from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("cerebras/Cerebras-GPT-13B")
model = AutoModelWithLMHead.from_pretrained("cerebras/Cerebras-GPT-13B")

# Set the context for the model to generate from
context = "hello world, who are you?"

# Encode the context into token IDs
input_ids = tokenizer.encode(context, return_tensors="pt")

# Generate text
output = model.generate(input_ids, max_length=1000, do_sample=True)

# Decode the generated token IDs back into text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
