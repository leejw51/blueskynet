import torch
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from torch.utils.data import Dataset
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


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
    
# Load the text data
data = "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog."

# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Encode the text data
encoding = tokenizer.encode_plus(data, return_tensors='pt')

# Save the encoded data to a file
os.makedirs('./data', exist_ok=True)
torch.save(encoding, './data/encoding.pt')

# Create a TextDataset from the encoded data
#dataset = TextDataset(file_path='./data/encoding.pt', tokenizer=tokenizer, block_size=tokenizer.model_max_length)
# Create a TensorDataset from the encoded data
dataset = TensorDataset(file_path='./data/encoding.pt')


# Initialize the DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Initialize the Trainer
training_args = TrainingArguments(
    output_dir='./results',     # output directory
    overwrite_output_dir=True,  # overwrite the content of the output directory
    num_train_epochs=1,         # number of training epochs
    per_device_train_batch_size=1,  # batch size for training
    save_steps=5000,           # after how many steps to save model checkpoint
    save_total_limit=2,        # number of maximum checkpoints to save
)
trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    data_collator=data_collator,          # data collator
    train_dataset=dataset,               # training dataset
)

# Train the model
trainer.train()
