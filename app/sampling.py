# sample.py

import torch
import numpy as np
import random
import os

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import necessary components from train_and_evaluation.py
from training_and_evaluation import (
    read_zeros,
    split_data,
    build_vocab,
    ZetaZerosDataset,
    TransformerSeq2Seq,
    generate_square_subsequent_mask,
    DATA_PATH,
    CHECKPOINT_DIR,
    MAX_SEQ_LEN,
    MAX_ZERO_LEN,
    EMBED_DIM,
    NHEAD,
    NHID,
    NLAYERS,
    DROPOUT,
)

# Read the zeros data and convert to strings
print("Loading data...")
zeros = read_zeros(DATA_PATH)
print(f"Total zeros loaded: {len(zeros)}")

# Split the data into train/dev/test sets
train_zeros, dev_zeros, test_zeros = split_data(zeros)
print(f"Data split into train ({len(train_zeros)}), dev ({len(dev_zeros)}), test ({len(test_zeros)})")

# Build character vocabulary
char2idx = build_vocab(zeros)
idx2char = {idx: ch for ch, idx in char2idx.items()}
vocab_size = len(char2idx)
print(f"Vocabulary size: {vocab_size}")

# Prepare test dataset
test_dataset = ZetaZerosDataset(test_zeros, max_seq_len=MAX_SEQ_LEN, max_zero_len=MAX_ZERO_LEN)
print(f"Total test examples: {len(test_dataset)}")

# Initialize the model with the same parameters
model = TransformerSeq2Seq(vocab_size, EMBED_DIM, NHEAD, NHID, NLAYERS, DROPOUT)

# Load the trained model
best_model_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth.tar')
print(f"Loading model from {best_model_path}")
model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
model = model.to(device)
model.eval()

# Function for top-p (nucleus) sampling
def nucleus_sampling(logits, top_p=0.9):
    # logits: Tensor of shape [vocab_size]
    # Compute softmax probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Sort the probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_keep = cumulative_probs <= top_p
    if not torch.any(sorted_indices_to_keep):
        # If no tokens meet the criterion, keep at least the top one
        sorted_indices_to_keep[0] = True
    sorted_probs = sorted_probs[sorted_indices_to_keep]
    sorted_indices = sorted_indices[sorted_indices_to_keep]

    # Normalize the probabilities
    normalized_probs = sorted_probs / torch.sum(sorted_probs)

    # Sample from the truncated distribution
    next_token = np.random.choice(sorted_indices.cpu().numpy(), p=normalized_probs.cpu().numpy())

    return next_token

# Generate and decode zeros in the test set
TOP_P = 0.9  # Top-p (nucleus) sampling threshold
MAX_GENERATE_LENGTH = 50  # Maximum length of generated zero

print("Generating zeros for the first 100 test examples using top-p sampling...")
for idx in range(100):
    # Retrieve the input sequence and target zero from the test dataset
    src_indices, tgt_indices = test_dataset[idx]
    src_indices = src_indices.to(device)
    src_len = src_indices.size(0)
    src_mask = torch.ones(1, src_len, dtype=torch.bool).to(device)  # [1, src_len]

    # Prepare source input
    src = src_indices.unsqueeze(0)  # [1, src_len]

    # Initialize the target sequence with <sos> token
    generated_indices = [char2idx['<sos>']]

    # Generate the sequence
    for _ in range(MAX_GENERATE_LENGTH):
        tgt_input = torch.tensor(generated_indices, dtype=torch.long).unsqueeze(0).to(device)  # [1, tgt_len]
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)
        tgt_key_padding_mask = torch.zeros(1, tgt_input.size(1), dtype=torch.bool).to(device)  # [1, tgt_len]

        # Perform forward pass
        with torch.no_grad():
            output = model(src, tgt_input, src_key_padding_mask=None, tgt_key_padding_mask=None)
            # Get the last token logits
            logits = output[0, -1, :]  # [vocab_size]

        # Apply top-p sampling to get the next token
        next_token = nucleus_sampling(logits, top_p=TOP_P)

        generated_indices.append(next_token)

        # Stop if <eos> token is generated
        if next_token == char2idx['<eos>']:
            break

    # Decode the generated indices to characters
    generated_zero = ''.join([idx2char[token] for token in generated_indices[1:-1]])  # Exclude <sos> and <eos>

    # Get the ground truth zero
    ground_truth_zero_indices = tgt_indices.cpu().numpy()
    ground_truth_zero = ''.join([idx2char[token] for token in ground_truth_zero_indices[1:-1]])  # Exclude <sos> and <eos>

    print(f"Index: {idx+1}")
    print(f"Predicted Zero: {generated_zero}")
    print(f"Ground Truth Zero: {ground_truth_zero}")
    print("---------------")
