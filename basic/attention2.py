import torch
import torch.nn as nn

# Import the nn.TransformerEncoder class
from torch.nn import TransformerEncoder

# print torch.__version__
print("torch version=", torch.__version__)


encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)


input_tokens =  torch.rand(10, 32, 512)
encoder_output = encoder(input_tokens)

print(encoder_output.shape) 
