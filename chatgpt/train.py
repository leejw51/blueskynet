import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Prepare training data
text = "The quick brown fox jumps over the lazy dog"
input_ids = tokenizer.encode(text, return_tensors='pt')

# Fine-tune model on small dataset
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

for i in range(100):
    loss = model(input_ids, labels=input_ids)[0]
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Generate text from fine-tuned model
model.eval()
generated_text = model.generate(
    input_ids,
    max_length=50,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
    top_k=0,
    repetition_penalty=1.0,
    num_return_sequences=1,
)

print(tokenizer.decode(generated_text[0], skip_special_tokens=True))
