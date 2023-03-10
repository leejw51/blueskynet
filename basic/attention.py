import torch
from transformers import AutoTokenizer, AutoModel

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Define some example text
text = "The quick brown fox jumps over the lazy dog."

# Tokenize the text
tokens = tokenizer(text, return_tensors='pt')

# Pass the tokenized text through the model
outputs = model(**tokens)

# Extract the last hidden state output
last_hidden_state = outputs.last_hidden_state

# Define linear projections for the Query, Key, and Value
query_proj = torch.nn.Linear(
    model.config.hidden_size, model.config.hidden_size)
key_proj = torch.nn.Linear(model.config.hidden_size, model.config.hidden_size)
value_proj = torch.nn.Linear(
    model.config.hidden_size, model.config.hidden_size)

# Compute the Query, Key, and Value representations using linear projections
query = query_proj(last_hidden_state)
key = key_proj(last_hidden_state)
value = value_proj(last_hidden_state)

# Compute the attention weights using the Query, Key, and Value representations
attention_weights = torch.matmul(
    query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
attention_weights = torch.softmax(attention_weights, dim=-1)

# Compute the attention vector using the attention weights and Value representations
attention_vector = torch.matmul(attention_weights, value)
