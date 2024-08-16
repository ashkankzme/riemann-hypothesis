from trax.models.transformer import TransformerDecoder
from trax.layers.activation_fns import FastGelu
from config import Config
from dataـpreparation import load_json_file, generate_train_validation_test_splits, generate_and_store_trajectories


if __name__ == '__main__':
    config = Config()

    # print("Generating and storing trajectories...")
    # generate_and_store_trajectories('../data/zeros_2m.txt', '../data/zz_trajectories_2m.json')
    #
    # print("Storing the results...")
    # time.sleep(2)  # just to be safe, we make sure the file is ready to be read by the next step

    print("Loading the trajectories...")
    zeta_zero_trajectories = load_json_file('../data/zz_trajectories_2m.json')

    # print max trajectory length
    print(f"max trajectory length is: {max([len(trajectory_item['trajectory']) for trajectory_item in zeta_zero_trajectories])}")

    # prepare the training, validation, and test splits
    print("Preparing the training, validation, and test splits...")

    training_trajectories, validation_trajectories, test_trajectories = (
        generate_train_validation_test_splits(zeta_zero_trajectories))

    decoder = TransformerDecoder( # todo use transformerlm instead
        vocab_size=len(config.vocabulary),
        d_model=config.d_token_embedding,
        d_ff=config.d_dense_layer,
        n_layers=config.layers,
        n_heads=config.num_heads,
        max_len=config.context_window_size,
        dropout=config.dropout_rate,
        dropout_shared_axes=(0, 1),  # recommended in the trax documentation to save memory
        mode='train',
        ff_activation=FastGelu()
    )

