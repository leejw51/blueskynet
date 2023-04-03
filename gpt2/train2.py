import torch
import os
import logging
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# Custom Dataset class
class TensorDataset(Dataset):
    def __init__(self, file_path):
        self.data = torch.load(file_path)
        self.input_ids = self.data['input_ids'].squeeze()
        self.attention_mask = self.data['attention_mask'].squeeze()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
        }

# Function to generate text using the model
def generate_text(prompt, model, tokenizer, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Load and encode text data
data = "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog."
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
encoding = tokenizer.encode_plus(data, return_tensors='pt')

# Save the encoded data to a file
os.makedirs('./data', exist_ok=True)
torch.save(encoding, './data/encoding.pt')

# Create a TensorDataset from the encoded data
dataset = TensorDataset(file_path='./data/encoding.pt')

# Initialize model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Initialize the DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Set training arguments and initialize the Trainer
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    save_steps=5000,
    save_total_limit=2,
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Train the model
trainer.train()

# Generate text using the trained model
test_prompt = "The quick brown fox"
generated_text = generate_text(test_prompt, model, tokenizer)
print(f"Generated text: {generated_text}")
