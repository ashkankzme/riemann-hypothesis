import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
DATA_PATH = './drive/MyDrive/Colab Notebooks/data/zeros_2m.txt'  # Path to the zeros data file
CHECKPOINT_DIR = './drive/MyDrive/Colab Notebooks/checkpoints'  # Directory to save checkpoints
LOG_INTERVAL = 100  # Log every N batches
SEED = 42  # Random seed
BATCH_SIZE = 512  # Batch size
NUM_EPOCHS = 3  # Number of training epochs
LEARNING_RATE = 1e-4  # Learning rate
MAX_SEQ_LEN = 100  # Maximum input sequence length (number of zeros)
MAX_ZERO_LEN = 50   # Maximum length of a zero in characters
EMBED_DIM = 128  # Embedding dimension
NHEAD = 8  # Number of heads in multi-head attention
NHID = 256  # Dimension of the feedforward network
NLAYERS = 3  # Number of Transformer layers
DROPOUT = 0.1  # Dropout rate
STEPS_PER_CHECKPOINT = 50  # Save checkpoint every N steps

# Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Read the zeros data and convert to strings
def read_zeros(data_path):
    zeros = []
    with open(data_path, 'r') as f:
        for line in f:
            zero = line.strip()
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

# Build character vocabulary
def build_vocab(zeros):
    chars = set(''.join(zeros))
    char2idx = {ch: idx + 3 for idx, ch in enumerate(sorted(chars))}
    char2idx['<pad>'] = 0  # Padding token
    char2idx['<sos>'] = 1  # Start of sequence token
    char2idx['<eos>'] = 2  # End of sequence token
    return char2idx

char2idx = build_vocab(zeros)
idx2char = {idx: ch for ch, idx in char2idx.items()}
vocab_size = len(char2idx)
print(f"Vocabulary size: {vocab_size}")

# Dataset class
class ZetaZerosDataset(Dataset):
    def __init__(self, zeros, max_seq_len=MAX_SEQ_LEN, max_zero_len=MAX_ZERO_LEN):
        self.zeros = zeros
        self.max_seq_len = max_seq_len
        self.max_zero_len = max_zero_len

    def __len__(self):
        return len(self.zeros) - 1  # Cannot predict after the last zero

    def __getitem__(self, idx):
        # Random sequence length between 10 and max_seq_len
        seq_len = random.randint(10, self.max_seq_len)
        start_idx = max(0, idx - seq_len + 1)
        zero_sequence = self.zeros[start_idx:idx+1]  # Include idx
        target_zero = self.zeros[idx+1]  # The next zero

        # Convert zeros to sequences of character indices
        # Input sequence
        zero_sequence_indices = [char2idx['<sos>']]
        for zero in zero_sequence:
            zero_indices = [char2idx[ch] for ch in zero]
            zero_sequence_indices.extend(zero_indices + [char2idx['<eos>']])

        zero_sequence_indices = torch.tensor(zero_sequence_indices, dtype=torch.long)
        # Target sequence (including <sos> and <eos>)
        target_zero_indices = [char2idx['<sos>']] + [char2idx[ch] for ch in target_zero] + [char2idx['<eos>']]
        target_zero_indices = torch.tensor(target_zero_indices, dtype=torch.long)

        return zero_sequence_indices, target_zero_indices

# Collate function for variable length sequences
def collate_fn(batch):
    zero_sequences, target_zeros = zip(*batch)

    # Pad sequences
    padded_sequences = pad_sequence(zero_sequences, batch_first=True, padding_value=char2idx['<pad>'])
    sequence_masks = padded_sequences != char2idx['<pad>']

    padded_targets = pad_sequence(target_zeros, batch_first=True, padding_value=char2idx['<pad>'])
    target_masks = padded_targets != char2idx['<pad>']

    return padded_sequences, sequence_masks, padded_targets, target_masks

# Create datasets and data loaders
train_dataset = ZetaZerosDataset(train_zeros)
dev_dataset = ZetaZerosDataset(dev_zeros)
test_dataset = ZetaZerosDataset(test_zeros)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# Transformer model for sequence-to-sequence prediction
class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, nhid, nlayers, dropout=0.1):
        super(TransformerSeq2Seq, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=char2idx['<pad>'])
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        self.pos_decoder = PositionalEncoding(embed_dim, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=nhid, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=nhid, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=nlayers)

        self.fc_out = nn.Linear(embed_dim, vocab_size)

        self.embed_scale = np.sqrt(embed_dim)

    def forward(self, src, tgt, src_key_padding_mask, tgt_key_padding_mask):
        src_emb = self.embedding(src) * self.embed_scale
        src_emb = self.pos_encoder(src_emb)

        tgt_emb = self.embedding(tgt) * self.embed_scale
        tgt_emb = self.pos_decoder(tgt_emb)

        src = src_emb.transpose(0, 1)  # [seq_len, batch_size, embed_dim]
        tgt = tgt_emb.transpose(0, 1)  # [seq_len, batch_size, embed_dim]

        memory = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        output = self.transformer_decoder(
            tgt,
            memory,
            tgt_mask=generate_square_subsequent_mask(tgt.size(0)).to(tgt.device),
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        output = self.fc_out(output)  # [tgt_len, batch_size, vocab_size]
        output = output.transpose(0, 1)  # [batch_size, tgt_len, vocab_size]
        return output

def generate_square_subsequent_mask(sz):
    """Generate a square mask for the sequence. The masked positions are filled with float('-inf')."""
    mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
    return mask

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)

        if embed_dim % 2 == 1:
            pe[:, -1] = torch.cos(position.squeeze() * div_term[-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, embed_dim]
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)

# Initialize the model, loss function, and optimizer
model = TransformerSeq2Seq(vocab_size, EMBED_DIM, NHEAD, NHID, NLAYERS, DROPOUT).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=char2idx['<pad>'])
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Function to save checkpoint
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

# Function to load checkpoint
def load_checkpoint(model, optimizer):
    # Find the latest checkpoint
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('model_checkpoint_step_')]
    if not checkpoints:
        print("No checkpoints found, starting from scratch")
        return 0, 0

    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    checkpoint_path = os.path.join(CHECKPOINT_DIR, latest_checkpoint)

    print(f"Loading checkpoint '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print(f"Loaded checkpoint (epoch {checkpoint['epoch']}, step {checkpoint['step']})")
    return checkpoint['epoch'], checkpoint['step']

# Training loop
def train(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    global_step = (epoch - 1) * len(train_loader)

    for batch_idx, (src, src_mask, tgt, tgt_mask) in enumerate(train_loader):
        src, src_mask = src.to(device), src_mask.to(device)
        tgt, tgt_mask = tgt.to(device), tgt_mask.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        tgt_input_mask = tgt_mask[:, :-1]
        tgt_output_mask = tgt_mask[:, 1:]

        optimizer.zero_grad()
        output = model(src, tgt_input, src_key_padding_mask=~src_mask, tgt_key_padding_mask=~tgt_input_mask)

        output = output.reshape(-1, vocab_size)
        tgt_output = tgt_output.reshape(-1)

        loss = criterion(output, tgt_output)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        global_step += 1

        if batch_idx % LOG_INTERVAL == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"Train Epoch: {epoch} [{batch_idx * len(src)}/{len(train_loader.dataset)}]"
                  f"\tLoss: {avg_loss:.6f}")

        # Save checkpoint every STEPS_PER_CHECKPOINT steps
        if global_step % STEPS_PER_CHECKPOINT == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'model_checkpoint_step_{global_step}.pth.tar')
            save_checkpoint({
                'epoch': epoch,
                'step': global_step,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename=checkpoint_path)

    avg_loss = total_loss / len(train_loader)
    print(f"====> Epoch: {epoch} Average loss: {avg_loss:.6f}")

# Evaluation loop
def evaluate(model, data_loader, set_name='Dev'):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, src_mask, tgt, tgt_mask in data_loader:
            src, src_mask = src.to(device), src_mask.to(device)
            tgt, tgt_mask = tgt.to(device), tgt_mask.to(device)

            # Prepare the input and target sequences for the decoder
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            tgt_input_mask = tgt_mask[:, :-1]
            tgt_output_mask = tgt_mask[:, 1:]

            output = model(src, tgt_input, src_key_padding_mask=~src_mask, tgt_key_padding_mask=~tgt_input_mask)

            # Flatten the output and target tensors
            output = output.reshape(-1, vocab_size)
            tgt_output = tgt_output.reshape(-1)

            loss = criterion(output, tgt_output)
            total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    print(f"====> {set_name} set loss: {avg_loss:.6f}")
    return avg_loss

# Main training and evaluation
start_epoch, global_step = load_checkpoint(model, optimizer)
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
