class Config:
    # Number of encoder-decoder layer pairs
    layers: int = 6

    # Number of units in dense layer
    mlp_dim: int = 2048

    # Tokens length
    length: int = 18

    # Number of embdedding dim
    features: int = 512

    # Batch size
    batch: int = 16

    # Number of heads in multihead attention
    num_heads: int = 8

    # Number of dims per head = features/num_heads
    head_dim: int = 64

    # Bias
    use_bias: bool = False

    # Droput rate
    dropout_rate: float = 0.2

    # Dropout or not
    training: bool = False

    # Random seed
    seed: int = 72

    # Percentage of the dataset to use for validation
    validation_split: float = 0.025

    # Number of epochs to train the model
    training_epochs: int = 1

    # Vocabulary: all the digits + the decimal point + the begin and end, and other special tokens
    vocabulary: list[str] = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', ':', 'b', 'e', ' ']

    # Trajectory offset range
    # some back of the envelope math that in order to stay under 192 tokens context windo,
    # we need to limit the range of the trajectory offsets between 1 and 7
    # the worse case scenario filling the most context length is offset being 7
    # 7 * (7 digit id + 1 : token + 7 digits + 1 . token + 9 digits + 1 space token) +
    # 1 begin token + 1 end token - 1 space token = 185
    # all calculations are based on the 2m zeros dataset
    trajectory_offset_range: list[int] = [1, 7]

    # Context window size
    context_window_size: int = 192