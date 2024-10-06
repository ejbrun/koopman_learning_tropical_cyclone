from absl import flags

FLAGS = flags.FLAGS

_SEED = flags.DEFINE_integer("seed", 123, "The random seed.")
_MODEL = flags.DEFINE_string(
    "model", "KNF", "dataset classes: RRR, Randomized_RRR, Nystroem-RRR, KNF"
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
