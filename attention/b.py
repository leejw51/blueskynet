import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torch.optim as optim

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        pe = self.pe[:, :seq_len, :]
        return x + pe

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_size = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # Linear transformations
        q = self.q_linear(q).view(bs, -1, self.num_heads, self.head_size)
        k = self.k_linear(k).view(bs, -1, self.num_heads, self.head_size)
        v = self.v_linear(v).view(bs, -1, self.num_heads, self.head_size)

        # Transpose to prepare for dot product attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores and apply mask
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_size)
        if mask is not None:
            #mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # Update the mask tensor
            mask = mask.unsqueeze(1).unsqueeze(2).repeat(1, self.num_heads, 1, 1)  # Update the mask tensor shape
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention probabilities
        attention_probs = F.softmax(scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Apply attention to value vectors
        attention = torch.matmul(attention_probs, v)

        # Concatenate and apply final linear transformation
        attention = attention.transpose(1, 2).contiguous().view(bs, -1, self.num_heads * self.head_size)
        output = self.out(attention)
        output = self.dropout(output)
        return output
    

class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)  # Add the required arguments here
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x



class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        norm = (x - mean) / (std + self.eps)
        return norm * self.gamma + self.beta        
    
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.encoder_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feedforward = PositionwiseFeedforward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)    

    def forward(self, x, encoder_output, self_mask=None, encoder_mask=None):
        # Self-attention layer
        x2 = self.norm1(x)
        attn_output = self.self_attn(x2, x2, x2, mask=self_mask)
        x = x + self.dropout(attn_output)

        # Encoder attention layer
        x2 = self.norm2(x)
        attn_output = self.encoder_attn(x2, encoder_output, encoder_output, mask=encoder_mask)
        x = x + self.dropout(attn_output)

        # Position-wise feedforward layer
        x2 = self.norm3(x)
        ff_output = self.feedforward(x2)
        x = x + self.dropout(ff_output)

        return x        
    

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = TransformerEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model    

    def forward(self, x, encoder_output, mask=None, encoder_mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, encoder_output, self_mask=mask, encoder_mask=encoder_mask)

        x = self.out(x)
        return x        

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feedforward = PositionwiseFeedforward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention layer
        x2 = self.norm1(x)
        attn_output = self.self_attn(x2, x2, x2, mask=mask)
        x = x + self.dropout(attn_output)

        # Position-wise feedforward layer
        x2 = self.norm2(x)
        ff_output = self.feedforward(x2)
        x = x + self.dropout(ff_output)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = TransformerEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask=mask)

        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = TransformerDecoder(vocab_size, d_model, num_heads, d_ff, num_layers, dropout)

    def forward(self, src, trg, src_mask=None, trg_mask=None, src_padding_mask=None, trg_padding_mask=None):
        encoder_output = self.encoder(src, mask=src_padding_mask)
        decoder_output = self.decoder(trg, encoder_output, mask=trg_mask, encoder_mask=src_padding_mask)

        return decoder_output
    

# Set hyperparameters
vocab_size = 10
d_model = 64
num_heads = 2
d_ff = 512
num_layers = 3
dropout = 0.1

# Initialize the Transformer model
model = Transformer(vocab_size, d_model, num_heads, d_ff, num_layers, dropout)

# Set the loss function and optimizer
loss_fn = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Generate some sample data
max_seq_len = 50
num_sequences = 10
input_sequences = np.random.randint(1, vocab_size, size=(num_sequences, max_seq_len))
target_sequences = np.concatenate([input_sequences[:, 1:], np.zeros((num_sequences, 1))], axis=1)

# Split the data into training and testing sets
train_test_split = int(0.8 * num_sequences)
X_train, X_test = input_sequences[:train_test_split], input_sequences[train_test_split:]
Y_train, Y_test = target_sequences[:train_test_split], target_sequences[train_test_split:]

# Convert the data to PyTorch tensors
X_train, X_test = torch.LongTensor(X_train), torch.LongTensor(X_test)
Y_train, Y_test = torch.LongTensor(Y_train), torch.LongTensor(Y_test)

# Train the model
num_epochs = 100
batch_size = 32
for epoch in range(num_epochs):
    # Shuffle the training data
    permutation = torch.randperm(X_train.size(0))

    # Train the model on batches of the data
    for i in range(0, X_train.size(0), batch_size):
        #show progress
        if i % 100 == 0:
            print("Epoch: {}, Batch: {}".format(epoch, i))

        indices = permutation[i:i+batch_size]
        batch_X, batch_Y = X_train[indices], Y_train[indices]

        # Set the model to training mode and zero the gradients
        model.train()
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(batch_X, batch_Y[:, :-1], src_padding_mask=(batch_X == 0), trg_padding_mask=(batch_Y[:, :-1] == 0))

        # Compute loss and gradients
        loss = loss_fn(y_pred.reshape(-1, vocab_size), batch_Y[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()

    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test, Y_test[:, :-1], src_padding_mask=(X_test == 0), trg_padding_mask=(Y_test[:, :-1] == 0))
        test_loss = loss_fn(y_pred_test.reshape(-1,vocab_size), Y_test[:, 1:].reshape(-1))
    # Print the training and test losses
    print(f'Epoch: {epoch+1}, Training Loss: {loss.item()}, Test Loss: {test_loss.item()}')
       
