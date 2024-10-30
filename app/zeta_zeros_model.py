# zeta_zero_model.py

import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
DATA_PATH = '../data/zeros_2m.txt'  # Path to the zeros data file
CHECKPOINT_DIR = '../models'  # Directory to save checkpoints
LOG_INTERVAL = 100  # Log every N batches
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

# Split the data into train/dev/test sets
def split_data(zeros, train_ratio=0.8, dev_ratio=0.05):
    total = len(zeros)
    train_end = int(total * train_ratio)
    dev_end = train_end + int(total * dev_ratio)
    train_zeros = zeros[:train_end]
    dev_zeros = zeros[train_end:dev_end]
    test_zeros = zeros[dev_end:]
    return train_zeros, dev_zeros, test_zeros

# Build character vocabulary
def build_vocab(zeros):
    chars = set(''.join(zeros))
    char2idx = {ch: idx + 3 for idx, ch in enumerate(sorted(chars))}
    char2idx['<pad>'] = 0  # Padding token
    char2idx['<sos>'] = 1  # Start of sequence token
    char2idx['<eos>'] = 2  # End of sequence token
    return char2idx

# Dataset class
class ZetaZerosDataset(Dataset):
    def __init__(self, zeros, char2idx, max_seq_len=MAX_SEQ_LEN, max_zero_len=MAX_ZERO_LEN):
        self.zeros = zeros
        self.char2idx = char2idx
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
        zero_sequence_indices = [self.char2idx['<sos>']]
        for zero in zero_sequence:
            zero_indices = [self.char2idx[ch] for ch in zero]
            zero_sequence_indices.extend(zero_indices + [self.char2idx['<eos>']])

        zero_sequence_indices = torch.tensor(zero_sequence_indices, dtype=torch.long)
        # Target sequence (including <sos> and <eos>)
        target_zero_indices = [self.char2idx['<sos>']] + [self.char2idx[ch] for ch in target_zero] + [self.char2idx['<eos>']]
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

# Transformer model for sequence-to-sequence prediction
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

def generate_square_subsequent_mask(sz):
    """Generate a square mask for the sequence. The masked positions are filled with float('-inf')."""
    mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
    return mask

class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, nhid, nlayers, dropout=0.1, char2idx=None):
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

# Functions to save and load checkpoints
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(model, optimizer, CHECKPOINT_DIR, map_location=device):
    # Find the latest checkpoint
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('model_checkpoint_step_')]
    if not checkpoints:
        print("No checkpoints found, starting from scratch")
        return 0, 0

    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    checkpoint_path = os.path.join(CHECKPOINT_DIR, latest_checkpoint)

    print(f"Loading checkpoint '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print(f"Loaded checkpoint (epoch {checkpoint['epoch']}, step {checkpoint['step']})")
    return checkpoint['epoch'], checkpoint['step']
