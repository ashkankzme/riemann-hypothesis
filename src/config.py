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
