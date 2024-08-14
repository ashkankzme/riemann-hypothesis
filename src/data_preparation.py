from config import Config
import random


def read_txt_file_line_by_line(filepath: str) -> list[str]:
    with open(filepath, 'r') as file:
        # strip the trailing spaces and newlines
        return [line.strip() for line in file.readlines()]


zeta_zeros = read_txt_file_line_by_line('../data/zeros.txt')
# all the digits + the decimal point + the begin and end special tokens
vocabulary = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', ':', 'b', 'e']

if __name__ == '__main__':
    config = Config()

    random.seed(config.seed)
    trajectories = []


    i = 0
    for i, zero in enumerate(zeta_zeros):
        trajectory = []
        for j, digit in enumerate(zero):
            trajectory.append(vocabulary.index(digit))
        trajectories.append(trajectory)
