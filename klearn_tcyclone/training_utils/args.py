"""Flags for training."""

from absl import flags

FLAGS = flags.FLAGS

_SEED = flags.DEFINE_integer("seed", 123, "The random seed.")
_MODEL = flags.DEFINE_string(
    "model", "KNF", "dataset classes: RRR, Randomized_RRR, Nystroem_RRR, KNF"
)
_DATASET = flags.DEFINE_string("dataset", "TCTracks", "dataset classes: TCTracks")
_YEAR_RANGE = flags.DEFINE_list(
    "year_range",
    [1990, 2021],
    # [2000, 2021],
    "Year range for the TCTracks data: defaults to [2001, 2002]",
)
_GLOBAL_LOCAL_COMBINATION = flags.DEFINE_string(
    "global_local_combination",
    "additive",
    "possible combinations: additive, multiplicative",
)
_LEARNING_RATE = flags.DEFINE_float(
    "learning_rate", 0.001, "The initial learning rate."
)
_DECAY_RATE = flags.DEFINE_float("decay_rate", 0.9, "The learning decay rate.")
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 32, "The batch size.")
_NUM_EPOCHS = flags.DEFINE_integer("num_epochs", 20, "The maximum number of epochs.")
_MIN_EPOCHS = flags.DEFINE_integer(
    "min_epochs", 10, "The minimum number of epochs the model is trained with."
)
_DATA_DIR = flags.DEFINE_string(
    "data_dir", "data_prep/M4/", "Data directory containing train and test data."
)
_REGULARIZE_RANK = flags.DEFINE_bool(
    "regularize_rank", False, "Whether to regularize dynamics module rank."
)
_USE_REVIN = flags.DEFINE_bool(
    "use_revin", True, "Whether to use reinversible normalization."
)
_USE_INSTANCENORM = flags.DEFINE_bool(
    "use_instancenorm", False, "Whether to use instance normalization."
)
_ADD_GLOBAL_OPERATOR = flags.DEFINE_bool(
    "add_global_operator", True, "Whether to use a gloabl Koopman operator."
)
_ADD_CONTROL = flags.DEFINE_bool(
    "add_control", True, "Whether to use a control module."
)
_DATA_FREQ = flags.DEFINE_string(
    "data_freq", "None", "The frequency of the time series data."
)
_DROPOUT_RATE = flags.DEFINE_float("dropout_rate", 0.0, "The dropout rate.")
_LATENT_DIM = flags.DEFINE_integer(
    "latent_dim", 32, "The dimension of latent Koopman space."
)
_NUM_STEPS = flags.DEFINE_integer(
    "num_steps",
    5,
    "The number of steps of predictions in one autoregressive call.",
)
_CONTROL_HIDDEN_DIM = flags.DEFINE_integer(
    "control_hidden_dim",
    64,
    "The hidden dimension of the module for learning adjustment matrix.",
)
_NUM_LAYERS = flags.DEFINE_integer(
    "num_layers", 4, "The number of layers in the encoder and decoder."
)
_CONTROL_NUM_LAYERS = flags.DEFINE_integer(
    "control_num_layers",
    3,
    "The number of layers in the module for learning adjustment matrix.",
)
_JUMPS = flags.DEFINE_integer(
    "jumps", 5, "The number of skipped steps when genrating sliced samples."
)
# Note that input_length must be divisible by input_dim.
_INPUT_DIM = flags.DEFINE_integer(
    "input_dim", 7, "The number of observations taken by the encoder at each step"
)
_INPUT_LENGTH = flags.DEFINE_integer(
    "input_length", 21, "The lookback window length for learning Koopman operator."
)
_HIDDEN_DIM = flags.DEFINE_integer(
    "hidden_dim", 128, "The hidden dimension of the encoder and decoder."
)
_TRAIN_OUTPUT_LENGTH = flags.DEFINE_integer(
    "train_output_length", 10, "The training output length for backpropogation."
)
_TEST_OUTPUT_LENGTH = flags.DEFINE_integer(
    "test_output_length", 10, "The forecasting horizon on the test set."
)
_NUM_HEADS = flags.DEFINE_integer("num_heads", 1, "Transformer number of heads")
_TRANSFORMER_DIM = flags.DEFINE_integer(
    "transformer_dim", 64, "Transformer feedforward dimension."
)
_TRANSFORMER_NUM_LAYERS = flags.DEFINE_integer(
    "transformer_num_layers", 3, "Number of Layers in Transformer Encoder."
)
_NUM_SINS = flags.DEFINE_integer("num_sins", -1, "number of sine functions.")
_NUM_POLY = flags.DEFINE_integer("num_poly", -1, "number of sine functions.")
_NUM_EXP = flags.DEFINE_integer("num_exp", -1, "number of sine functions.")

_CONTEXT_LENGTH = flags.DEFINE_integer(
    "context_length", 42, "The context length for Koopman kernels."
)

_TIKHONOV_REG = flags.DEFINE_float(
    "tikhonov_reg", 1e-6, "Tikhonov regularization coefficient."
)
_KOOPMAN_KERNEL_RANK = flags.DEFINE_integer(
    "koopman_kernel_rank", 25, "The rank of the Koopman kernel."
)
_KOOPMAN_KERNEL_REDUCED_RANK = flags.DEFINE_bool(
    "koopman_kernel_reduced_rank", True, "Whether to use reduced rank."
)
_KOOPMAN_KERNEL_NUM_CENTERS = flags.DEFINE_integer(
    "koopman_kernel_num_centers", 250, "The number of centers of the Koopman kernel."
)
_KOOPMAN_KERNEL_LENGTH_SCALE = flags.DEFINE_float(
    "koopman_kernel_length_scale", 50.0, "The length scale of the Koopman kernel."
)
_KOOPMAN_KERNEL_SVD_SOLVER = flags.DEFINE_string(
    "koopman_kernel_svd_solver", "randomized", "Which svd solver to use."
)
_KOOPMAN_KERNEL_NUM_TRAIN_STOPS = flags.DEFINE_integer(
    "koopman_kernel_num_train_stops", 6, "The number of training stops."
)


ALL_FLAGS = [
    _SEED,
    _MODEL,
    _DATASET,
    _YEAR_RANGE,
    _GLOBAL_LOCAL_COMBINATION,
    _LEARNING_RATE,
    _DECAY_RATE,
    _BATCH_SIZE,
    _NUM_EPOCHS,
    _MIN_EPOCHS,
    _DATA_DIR,
    _REGULARIZE_RANK,
    _USE_REVIN,
    _USE_INSTANCENORM,
    _ADD_GLOBAL_OPERATOR,
    _ADD_CONTROL,
    _DATA_FREQ,
    _DROPOUT_RATE,
    _LATENT_DIM,
    _NUM_STEPS,
    _CONTROL_HIDDEN_DIM,
    _NUM_LAYERS,
    _CONTROL_NUM_LAYERS,
    _JUMPS,
    _INPUT_DIM,
    _INPUT_LENGTH,
    _HIDDEN_DIM,
    _TRAIN_OUTPUT_LENGTH,
    _TEST_OUTPUT_LENGTH,
    _NUM_HEADS,
    _TRANSFORMER_DIM,
    _TRANSFORMER_NUM_LAYERS,
    _NUM_SINS,
    _NUM_POLY,
    _NUM_EXP,
    _CONTEXT_LENGTH,
    _TIKHONOV_REG,
    _KOOPMAN_KERNEL_RANK,
    _KOOPMAN_KERNEL_REDUCED_RANK,
    _KOOPMAN_KERNEL_NUM_CENTERS,
    _KOOPMAN_KERNEL_LENGTH_SCALE,
    _KOOPMAN_KERNEL_SVD_SOLVER,
    _KOOPMAN_KERNEL_NUM_TRAIN_STOPS,
]

# KNF_FLAGS = [
#     _SEED,
#     _MODEL,
#     _DATASET,
#     _YEAR_RANGE,
#     _GLOBAL_LOCAL_COMBINATION,
#     _LEARNING_RATE,
#     _DECAY_RATE,
#     _BATCH_SIZE,
#     _NUM_EPOCHS,
#     _MIN_EPOCHS,
#     _DATA_DIR,
#     _REGULARIZE_RANK,
#     _USE_REVIN,
#     _USE_INSTANCENORM,
#     _ADD_GLOBAL_OPERATOR,
#     _ADD_CONTROL,
#     _DATA_FREQ,
#     _DROPOUT_RATE,
#     _LATENT_DIM,
#     _NUM_STEPS,
#     _CONTROL_HIDDEN_DIM,
#     _NUM_LAYERS,
#     _CONTROL_NUM_LAYERS,
#     _JUMPS,
#     _INPUT_DIM,
#     _INPUT_LENGTH,
#     _HIDDEN_DIM,
#     _TRAIN_OUTPUT_LENGTH,
#     _TEST_OUTPUT_LENGTH,
#     _NUM_HEADS,
#     _TRANSFORMER_DIM,
#     _TRANSFORMER_NUM_LAYERS,
#     _NUM_SINS,
#     _NUM_POLY,
#     _NUM_EXP,
# ]
