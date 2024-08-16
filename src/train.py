from trax.models.transformer import TransformerLM
from trax.supervised import training
from trax import layers as tl
from trax.optimizers import Adam
from config import Config
from data_preparation import load_json_file, generate_train_validation_test_splits, generate_batch, generate_and_store_trajectories
import itertools


def generate_decoder_only_transformer_lm(config):
    return TransformerLM(
        vocab_size=len(config.vocabulary),
        d_model=config.d_token_embedding,
        d_ff=config.d_dense_layer,
        n_layers=config.layers,
        n_heads=config.num_heads,
        max_len=config.context_window_size,
        dropout=config.dropout_rate,
        dropout_shared_axes=(0, 1),  # recommended in the trax documentation to save memory
        mode='train',
        ff_activation=tl.activation_fns.FastGelu
    )


if __name__ == '__main__':
    config = Config()

    # print("Generating and storing trajectories...")
    # generate_and_store_trajectories('../data/zeros_2m.txt', '../data/zz_trajectories_2m.json')
    #
    # print("Storing the results...")
    # time.sleep(2)  # just to be safe, we make sure the file is ready to be read by the next step

    print("Loading the trajectories...")
    zeta_zero_trajectories = load_json_file('../data/zz_trajectories_2m.json')

    # prepare the training, validation, and test splits
    print("Preparing the training, validation, and test splits...")

    training_trajectories, validation_trajectories, _ = (
        generate_train_validation_test_splits(zeta_zero_trajectories))

    infinite_train_generator = itertools.cycle(generate_batch(training_trajectories, config.batch_size))
    infinite_eval_generator = itertools.cycle(generate_batch(validation_trajectories, config.batch_size))

    # since we are doing autoregressive modeling, a decoder-only transformer is sufficient (similar to gpt-2)
    decoderLM = generate_decoder_only_transformer_lm(config)

    # todo use validation set to tune hyperparameters

    # Training task.
    train_task = training.TrainTask(
        labeled_data=infinite_train_generator,
        loss_layer=tl.CrossEntropyLoss(),
        optimizer=Adam(0.01),
        n_steps_per_checkpoint=500,
    )

    # Evaluaton task.
    eval_task = training.EvalTask(
        labeled_data=infinite_eval_generator,
        metrics=[tl.CrossEntropyLoss(), tl.Accuracy()],
        n_eval_batches=20  # For less variance in eval numbers.
    )

    # Train the model.
    training_loop = training.Loop(
        decoderLM,
        train_task,
        eval_tasks=[eval_task],
        output_dir=config.training_output_dir)

    # do one epoch, which is len(training_trajectories) / config.batch_size
    epoch_steps = (len(training_trajectories) // config.batch_size) + 1

    # begin training
    print("Training the model...")
    training_loop.run(n_steps=epoch_steps)
