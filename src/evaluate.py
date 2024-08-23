from train import generate_decoder_only_transformer_lm
from config import Config
from data_preparation import load_json_file, generate_train_validation_test_splits, generate_batch
import itertools
import numpy as np
from trax.supervised.decoding import autoregressive_sample


if __name__ == '__main__':
    config = Config()

    number_of_tokens_to_predict = 4  # todo add this to the config

    print("Loading the trajectories...")
    zeta_zero_trajectories = load_json_file('../data/zz_trajectories_2m.json')

    # prepare the training, validation, and test splits
    print("Preparing the training, validation, and test splits...")

    _, _, test_trajectories = (generate_train_validation_test_splits(zeta_zero_trajectories))
    test_trajectories = test_trajectories[:config.batch_size]
    infinite_test_generator = itertools.cycle(generate_batch(test_trajectories, config.batch_size))

    # since we are doing autoregressive modeling, a decoder-only transformer is sufficient (similar to gpt-2)
    decoderLM = generate_decoder_only_transformer_lm(config)
    # predict_signature = ShapeDtype((1, 1), dtype=np.int32)
    model_path = config.training_output_dir + '/model.pkl.gz'
    decoderLM.init_from_file(model_path, weights_only=True)
    # decoderLM.init_from_file(model_path, input_signature=predict_signature, weights_only=True)

    # Run the model
    for _ in range(5):

        trajectories = next(infinite_test_generator)[0]  # get the x out of (x, y, mask)
        padding_token_id = config.vocabulary.index('p')
        # find the first padding token index in the trajectory list for each row, return as a list of indices
        last_token_before_padding = [len(list(itertools.takewhile(lambda x: x != padding_token_id, trajectory_row)))
                                     for trajectory_row in trajectories]

        for i, trajectory in enumerate(trajectories):
            input_ids, output_ids = (np.array([trajectory[:last_token_before_padding[i]-number_of_tokens_to_predict]]),
                                     trajectory[last_token_before_padding[i]-number_of_tokens_to_predict:])
            prediction = autoregressive_sample(decoderLM, input_ids, batch_size=1, temperature=0.5, max_length=32,
                                            eos_id=config.vocabulary.index('e'))

            print(f"Predicted output: {''.join([config.vocabulary[token] for token in prediction[0]])}")
            print(f"Expected output: {''.join([config.vocabulary[token] for token in output_ids])}")