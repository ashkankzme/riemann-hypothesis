from config import Config
import random
import json
import time
from trax.fastmath import numpy as np


def load_json_file(filepath: str) -> list[list[int]]:
    with open(filepath, 'r') as file:
        return json.load(file)


def read_txt_file_line_by_line(filepath: str) -> list[str]:
    with open(filepath, 'r') as file:
        # strip the trailing spaces and newlines
        return [line.strip() for line in file.readlines()]


def generate_and_store_trajectories(input_path, output_path):
    zeta_zeros = read_txt_file_line_by_line(input_path)
    vocabulary = config.vocabulary
    trajectories = []
    masks = []
    random.seed(config.seed)

    i = 0
    offset = 0
    offset_range = config.trajectory_offset_range
    while i < len(zeta_zeros):
        trajectory = [vocabulary.index('b')]
        mask = []

        offset = random.randint(offset_range[0], offset_range[1])
        # if i + offset is greater than the length of the zeros, then we  reset offset to len(zeta_zeros) - i
        if i + offset > len(zeta_zeros):
            offset = len(zeta_zeros) - i

        slice_of_zeros = zeta_zeros[i:i + offset]

        for j, zero in enumerate(slice_of_zeros):
            j_as_a_string = str(i + j)
            trajectory += [vocabulary.index(digit) for digit in j_as_a_string] + [vocabulary.index(':')]
            trajectory += [vocabulary.index(digit) for digit in zero]
            if j < len(slice_of_zeros) - 1:
                trajectory.append(vocabulary.index(' '))
            else:
                trajectory.append(vocabulary.index('e'))
                mask_length = len(zero) + 1 + len(j_as_a_string) + 1
                mask += [0] * (len(trajectory) - mask_length)
                mask += [1] * mask_length

        # add padding to the trajectory to make it of length 192
        trajectory += [vocabulary.index('p')] * (config.context_window_size - len(trajectory))
        mask += [0] * (config.context_window_size - len(mask))

        trajectories.append({'trajectory': trajectory, 'mask': mask})

        # advance the index by the offset
        i += offset

    # shuffle the trajectories
    random.shuffle(trajectories)

    # store the trajectories as a json file
    with open(output_path, 'w') as file:
        json.dump(trajectories, file)


def generate_batch(trajectories, batch_size):
    for i in range(0, len(trajectories), batch_size):
        # split the trajectories list of dictionaries into batches of trajectories and masks
        batch = trajectories[i:i + batch_size]
        x = np.array([trajectory['trajectory'] for trajectory in batch])
        mask = np.array([trajectory['mask'] for trajectory in batch])
        yield x, x, mask


def generate_train_validation_test_splits(zeta_zero_trajectories):
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

    return training_trajectories, validation_trajectories, test_trajectories


if __name__ == '__main__':
    config = Config()

    print("Generating and storing trajectories...")
    generate_and_store_trajectories('../data/zeros_2m.txt', '../data/zz_trajectories_2m.json')

    print("Storing the results...")
    time.sleep(2)  # just to be safe, we make sure the file is ready to be read by the next step

    print("Loading the trajectories...")
    zeta_zero_trajectories = load_json_file('../data/zz_trajectories_2m.json')

    # print max trajectory length
    print(f"max trajectory length is: {max([len(trajectory_item['trajectory']) for trajectory_item in zeta_zero_trajectories])}")

    # prepare the training, validation, and test splits
    print("Preparing the training, validation, and test splits...")

    training_trajectories, validation_trajectories, test_trajectories = (
        generate_train_validation_test_splits(zeta_zero_trajectories))

    # print a sample and some statistics of the training trajectories
    print(f"Number of training trajectories: {len(training_trajectories)}")
    print(f"Number of validation trajectories: {len(validation_trajectories)}")
    print(f"Number of test trajectories: {len(test_trajectories)}")
    print(f"Sample of a training trajectory: {training_trajectories[0]}")
    print(f"Sample of a validation trajectory: {validation_trajectories[0]}")
    print(f"Sample of a test trajectory: {test_trajectories[0]}")