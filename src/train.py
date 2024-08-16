import json
from trax.models.transformer import TransformerDecoder
from trax.layers.activation_fns import FastGelu
from config import Config


def load_json_file(filepath: str) -> list[list[int]]:
    with open(filepath, 'r') as file:
        return json.load(file)


if __name__ == '__main__':
    zeta_zero_trajectories = load_json_file('../data/zz_trajectories_2m.json')
    config = Config()

    # split the data into training (90%), validation (1%), and test (9%) sets
    training_split = 0.9
    validation_split = 0.01
    test_split = 0.09

    # calculate the indices of the end of each set based on the percentage of the total trajectories
    total_trajectories = len(zeta_zero_trajectories)
    training_end = int(total_trajectories * training_split)
    validation_end = training_end + int(total_trajectories * validation_split)

    training_trajectories = zeta_zero_trajectories[:training_end]
    validation_trajectories = zeta_zero_trajectories[training_end:validation_end]
    test_trajectories = zeta_zero_trajectories[validation_end:]

    decoder = TransformerDecoder(
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

