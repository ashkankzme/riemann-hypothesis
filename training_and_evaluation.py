import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
DATA_PATH = './data/zeros_2m.txt'  # Path to the zeros data file
CHECKPOINT_DIR = './checkpoints'  # Directory to save checkpoints
LOG_INTERVAL = 100  # Log every N batches
SEED = 42  # Random seed
BATCH_SIZE = 64  # Batch size
NUM_EPOCHS = 10  # Number of training epochs
LEARNING_RATE = 1e-4  # Learning rate
MAX_SEQ_LEN = 100  # Maximum sequence length
EMBED_DIM = 128  # Embedding dimension
NHEAD = 8  # Number of heads in multi-head attention
NHID = 256  # Dimension of the feedforward network
NLAYERS = 3  # Number of Transformer layers
DROPOUT = 0.1  # Dropout rate

# Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Read the zeros data
def read_zeros(data_path):
    zeros = []
    with open(data_path, 'r') as f:
        for line in f:
            zero = float(line.strip())
            zeros.append(zero)
    return zeros

print("Loading data...")
zeros = read_zeros(DATA_PATH)
print(f"Total zeros loaded: {len(zeros)}")

# Split the data into train/dev/test sets
def split_data(zeros, train_ratio=0.8, dev_ratio=0.05):
    total = len(zeros)
    train_end = int(total * train_ratio)
    dev_end = train_end + int(total * dev_ratio)
    train_zeros = zeros[:train_end]
    dev_zeros = zeros[train_end:dev_end]
    test_zeros = zeros[dev_end:]
    return train_zeros, dev_zeros, test_zeros

train_zeros, dev_zeros, test_zeros = split_data(zeros)
print(f"Data split into train ({len(train_zeros)}), dev ({len(dev_zeros)}), test ({len(test_zeros)})")

# Dataset class
class ZetaZerosDataset(Dataset):
    def __init__(self, zeros, max_seq_len=MAX_SEQ_LEN):
        self.zeros = zeros
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.zeros) - 1  # Cannot predict after the last zero

    def __getitem__(self, idx):
        # Random sequence length between 10 and max_seq_len
        seq_len = random.randint(10, self.max_seq_len)
        start_idx = max(0, idx - seq_len + 1)
        sequence = self.zeros[start_idx:idx+1]  # Include idx
        target = self.zeros[idx+1]  # The next zero
        sequence = torch.tensor(sequence, dtype=torch.float)
        target = torch.tensor(target, dtype=torch.float)
        return sequence, target

# Collate function for variable length sequences
def collate_fn(batch):
    sequences, targets = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    targets = torch.stack(targets)
    # Create attention masks
    masks = torch.arange(padded_sequences.size(1))[None, :] < lengths[:, None]
    return padded_sequences, masks, targets

# Create datasets and data loaders
train_dataset = ZetaZerosDataset(train_zeros)
dev_dataset = ZetaZerosDataset(dev_zeros)
test_dataset = ZetaZerosDataset(test_zeros)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# Transformer Model for regression
class TransformerRegressor(nn.Module):
    def __init__(self, embed_dim, nhead, nhid, nlayers, dropout=0.1):
        super(TransformerRegressor, self).__init__()
        self.model_type = 'Transformer'

        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(embed_dim, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)

        self.embedding = nn.Linear(1, embed_dim)
        self.decoder = nn.Linear(embed_dim, 1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = src.unsqueeze(-1)  # Add feature dimension
        src = self.embedding(src) * math.sqrt(EMBED_DIM)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)  # Shape for transformer: [seq_len, batch_size, embed_dim]
        output = self.transformer_encoder(src, src_key_padding_mask=~src_mask)
        output = output.permute(1, 0, 2)
        # Use the output corresponding to the last element in the sequence
        idx = (src_mask.sum(1) - 1).unsqueeze(1).unsqueeze(2).expand(-1, -1, EMBED_DIM)
        last_output = output.gather(1, idx).squeeze(1)
        output = self.decoder(last_output)
        return output.squeeze()

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=MAX_SEQ_LEN):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        if embed_dim % 2 == 1:
            # If embed_dim is odd, we need to handle the last term separately
            pe[:, -1] = torch.cos(position.squeeze() * div_term[-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, embed_dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Initialize the model, loss function, and optimizer
model = TransformerRegressor(EMBED_DIM, NHEAD, NHID, NLAYERS, DROPOUT).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Function to save checkpoint
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

# Function to load checkpoint
def load_checkpoint(model, optimizer, filename):
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded checkpoint '{filename}' (epoch {start_epoch})")
        return start_epoch
    else:
        print(f"No checkpoint found at '{filename}', starting from scratch")
        return 0

# Training loop
def train(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, mask, target) in enumerate(train_loader):
        data, mask, target = data.to(device), mask.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data, mask)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % LOG_INTERVAL == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]"
                  f"\tLoss: {avg_loss:.6f}")
    avg_loss = total_loss / len(train_loader)
    print(f"====> Epoch: {epoch} Average loss: {avg_loss:.6f}")

# Evaluation loop
def evaluate(model, data_loader, set_name='Dev'):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, mask, target in data_loader:
            data, mask, target = data.to(device), mask.to(device), target.to(device)
            output = model(data, mask)
            loss = criterion(output, target)
            total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    print(f"====> {set_name} set loss: {avg_loss:.6f}")
    return avg_loss

# Main training and evaluation
start_epoch = load_checkpoint(model, optimizer, os.path.join(CHECKPOINT_DIR, 'model_checkpoint.pth.tar'))
best_dev_loss = float('inf')

for epoch in range(start_epoch + 1, NUM_EPOCHS + 1):
    train(model, train_loader, optimizer, epoch)
    dev_loss = evaluate(model, dev_loader, set_name='Dev')

    # Save checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'model_checkpoint_epoch_{epoch}.pth.tar')
    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, filename=checkpoint_path)

    # Save the best model
    if dev_loss < best_dev_loss:
        best_dev_loss = dev_loss
        best_model_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth.tar')
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved to {best_model_path}")

# Evaluate on the test set
print("Evaluating on test set...")
model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'best_model.pth.tar')))
test_loss = evaluate(model, test_loader, set_name='Test')
