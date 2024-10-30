# train_and_evaluate.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import necessary components from zeta_zero_model.py
from zeta_zero_model import (
    read_zeros,
    split_data,
    build_vocab,
    ZetaZerosDataset,
    TransformerSeq2Seq,
    collate_fn,
    save_checkpoint,
    load_checkpoint,
    device,
    DATA_PATH,
    CHECKPOINT_DIR,
    LOG_INTERVAL,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    MAX_SEQ_LEN,
    MAX_ZERO_LEN,
    EMBED_DIM,
    NHEAD,
    NHID,
    NLAYERS,
    DROPOUT,
    STEPS_PER_CHECKPOINT,
    char2idx,   # If char2idx is defined in zeta_zero_model.py
)

if __name__ == "__main__":
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

    # Create datasets and data loaders
    train_dataset = ZetaZerosDataset(train_zeros, char2idx)
    dev_dataset = ZetaZerosDataset(dev_zeros, char2idx)
    test_dataset = ZetaZerosDataset(test_zeros, char2idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Initialize the model, loss function, and optimizer
    model = TransformerSeq2Seq(vocab_size, EMBED_DIM, NHEAD, NHID, NLAYERS, DROPOUT, char2idx=char2idx).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=char2idx['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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
    start_epoch, global_step = load_checkpoint(model, optimizer, CHECKPOINT_DIR)
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
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'best_model.pth.tar'), map_location=device))
    test_loss = evaluate(model, test_loader, set_name='Test')
    print(f"Test set loss: {test_loss:.6f}")