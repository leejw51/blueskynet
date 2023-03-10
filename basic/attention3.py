import torch
import torch.nn as nn

# Import the nn.TransformerEncoder class
from torch.nn import TransformerEncoder

# print torch.__version__
print("torch version=", torch.__version__)

# Set the word embedding vector length
embedding_dim = 256

encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

# Generate a sequence of input tokens with the new embedding dimensionality
input_tokens = torch.rand(10, 32, embedding_dim)

# Define a linear layer to project the embeddings to the `d_model` dimensionality
projection_layer = nn.Linear(embedding_dim, 512)

# Pass the input embeddings through the projection layer to convert them to the `d_model` dimensionality
input_embeddings = projection_layer(input_tokens)

# Pass the projected embeddings through the transformer encoder
encoder_output = encoder(input_embeddings)

print(encoder_output.shape)

