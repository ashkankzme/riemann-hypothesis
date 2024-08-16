from config import Config
import random
import json


def read_txt_file_line_by_line(filepath: str) -> list[str]:
    with open(filepath, 'r') as file:
        # strip the trailing spaces and newlines
        return [line.strip() for line in file.readlines()]


if __name__ == '__main__':
    config = Config()
    zeta_zeros = read_txt_file_line_by_line('../data/zeros_2m.txt')
    vocabulary = config.vocabulary
    trajectories = []
    random.seed(config.seed)

    i = 0
    offset = 0
    offset_range = config.trajectory_offset_range
    while i < len(zeta_zeros):
        trajectory = [vocabulary.index('b')]

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
        trajectory.append(vocabulary.index('e'))
        trajectories.append(trajectory)

        # advance the index by the offset
        i += offset

    # store the trajectories as a json file
    with open('../data/zz_trajectories_2m.json', 'w') as file:
        json.dump(trajectories, file)

    # print max trajectory length
    print(f"max trajectory length is: {max([len(trajectory) for trajectory in trajectories])}")