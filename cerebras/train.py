from transformers import AutoTokenizer, AutoModelWithLMHead
import torch

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("cerebras/Cerebras-GPT-13B")
model = AutoModelWithLMHead.from_pretrained("cerebras/Cerebras-GPT-13B")

# Load the question-answer dataset
dataset = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "Who wrote the novel 'Pride and Prejudice'?", "answer": "Jane Austen"},
    # Add more question-answer pairs here
]

# Encode the questions and answers into token IDs
input_ids = []
attention_masks = []
labels = []
for example in dataset:
    # Encode the question and answer
    encoded_dict = tokenizer(
        example["question"],
        example["answer"],
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    input_ids.append(encoded_dict["input_ids"])
    attention_masks.append(encoded_dict["attention_mask"])
    
    # Encode the label (0 for incorrect, 1 for correct)
    if example["answer"].lower() in example["question"].lower():
        labels.append(1)
    else:
        labels.append(0)

# Convert the input IDs, attention masks, and labels to tensors
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# Define the training arguments
training_args = {
    "output_dir": "./results",
    "overwrite_output_dir": True,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,
    "save_steps": 1000,
    "save_total_limit": 1,
}

# Define the training dataset
training_dataset = torch.utils.data.TensorDataset(input_ids, attention_masks, labels)

# Define the trainer
trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=training_dataset,
    data_collator=lambda data: {"input_ids": torch.stack([item[0] for item in data]),
                                "attention_mask": torch.stack([item[1] for item in data]),
                                "labels": torch.stack([item[2] for item in data])},
    compute_metrics=compute_metrics
)

# Define the metrics function
def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    predictions = torch.argmax(predictions, axis=1)
    accuracy = torch.sum(predictions == labels) / len(labels)
    return {"accuracy": accuracy}

# Train the model
trainer.train()
