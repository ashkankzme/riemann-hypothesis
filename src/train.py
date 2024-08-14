import json


def load_json_file(filepath: str) -> list[list[int]]:
    with open(filepath, 'r') as file:
        return json.load(file)


if __name__ == '__main__':
    zeta_zero_trajectories = load_json_file('../data/zz_trajectories_2m.json')
    print(f"Number of trajectories: {len(zeta_zero_trajectories)}")
