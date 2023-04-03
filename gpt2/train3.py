from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, pipeline
import torch
from transformers import DataCollatorWithPadding


def custom_data_collator(features):
    data_collator = DataCollatorWithPadding(tokenizer)
    batch = data_collator(features)
    
    batch["start_positions"] = torch.stack([f["start_positions"] for f in features])
    batch["end_positions"] = torch.stack([f["end_positions"] for f in features])
    
    return batch


# Load the question-answering model and tokenizer
model_name = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Define a sample dataset
dataset = [
    {
        "context": "The quick brown fox jumps over the lazy dog",
        "question": "What animal jumps over the lazy dog?",
        "answer": "fox",
    },
    {
        "context": "The capital of France is Paris",
        "question": "What is the capital of France?",
        "answer": "Paris",
    },
]

def tokenize_data(example):
    encoding = tokenizer(example["question"], example["context"], return_tensors='pt', padding='max_length', max_length=512, truncation=True)

    answer_start = example["context"].index(example["answer"])
    answer_end = answer_start + len(example["answer"])

    start_positions = encoding.char_to_token(answer_start)
    end_positions = encoding.char_to_token(answer_end)

    if start_positions is None or end_positions is None:
        start_positions = 0
        end_positions = 0

    return {
        "input_ids": encoding["input_ids"].squeeze(),
        "attention_mask": encoding["attention_mask"].squeeze(),
        "start_positions": torch.tensor([start_positions]),
        "end_positions": torch.tensor([end_positions]),
    }



tokenized_dataset = [tokenize_data(example) for example in dataset]

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Define an evaluation dataset
eval_dataset = tokenized_dataset



# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=eval_dataset, # Add this line to specify the evaluation dataset
    data_collator=custom_data_collator,
)

# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()

# Save the trained model
model.save_pretrained("trained_qa_model/")
tokenizer.save_pretrained("trained_qa_model/")

# Use the trained model to answer a sample question
qa_pipeline = pipeline("question-answering", model="./trained_qa_model", tokenizer="./trained_qa_model")
question = "What animal jumps over the lazy dog?"
context = "The quick brown fox jumps over the lazy dog"
result = qa_pipeline(question=question, context=context)
print(result["answer"])
