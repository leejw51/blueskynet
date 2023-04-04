import torch
import torch.nn as nn
import torch.optim as optim
import math


# Sample Data
src = torch.rand((10, 32, 512)) # Source tensor of shape (sequence length, batch size, embedding dimension)
trg = torch.rand((20, 32, 512)) # Target tensor of shape (sequence length, batch size, embedding dimension)
output_dim = 512 # Output dimension


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=200):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# Transformer
class Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self.fc_out = nn.Linear(d_model, output_dim)
        
    def forward(self, src, trg):
        src = self.pos_encoder(src)
        trg = self.pos_encoder(trg)
        memory = self.encoder(src)
        output = self.decoder(trg, memory)
        output = self.fc_out(output)
        return output


# Training
model = Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(10):
    optimizer.zero_grad()
    output = model(src, trg[:-1])
    loss = criterion(output, trg[1:])
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1} Loss: {loss.item():.4f}')
