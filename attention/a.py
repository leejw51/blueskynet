import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_size = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

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
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention probabilities
        attention_probs = F.softmax(scores, dim=-1)
        attention = torch.matmul(attention_probs, v)

        # Concatenate and apply final linear transformation
        attention = attention.transpose(1, 2).contiguous().view(bs, -1, self.num_heads * self.head_size)
        output = self.out(attention)
        return output


class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feedforward = PositionwiseFeedforward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention layer
        attn_output = self.self_attn(x, x, x, mask=mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Position-wise feed
        ff_output = self.feedforward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        encoded = self.encoder(x, mask=mask)
        decoded = self.decoder(encoded.mean(dim=1))
        return decoded
            

# Create a simple dataset
input_size = 8
seq_length = 10
num_samples = 1000
X = torch.randn(num_samples, seq_length, input_size)
Y = X.sum(dim=-1).sum(dim=-1, keepdim=True)

# Split the dataset into training and testing sets
train_test_split = int(0.8 * num_samples)
X_train, X_test = X[:train_test_split], X[train_test_split:]
Y_train, Y_test = Y[:train_test_split], Y[train_test_split:]

# Initialize the Transformer model
num_layers = 2
d_model = input_size
num_heads = 2
d_ff = 16
dropout = 0.1
model = Transformer(num_layers, d_model, num_heads, d_ff, dropout)

# Set the loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, Y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            y_pred_test = model(X_test)
            test_loss = loss_fn(y_pred_test, Y_test)
        print(f'Epoch: {epoch+1}, Training Loss: {loss.item()}, Test Loss: {test_loss.item()}')
