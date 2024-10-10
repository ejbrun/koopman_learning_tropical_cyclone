"""Setting flags."""

from absl.flags import FlagValues

from klearn_tcyclone.training_utils.args import ALL_FLAGS


def set_flags(FLAGS: FlagValues) -> dict:
    model = FLAGS.model
    year_range = [int(s) for s in FLAGS.year_range]
    if model == "KNF":
        # Currently implemented flag dictionary

        flag_dict = {
            "seed": FLAGS.seed,
            "model": FLAGS.model,
            "dataset": FLAGS.dataset,
            "year_range": year_range,
            "global_local_combination": FLAGS.global_local_combination,
            "learning_rate": FLAGS.learning_rate,
            "decay_rate": FLAGS.decay_rate,
            "batch_size": FLAGS.batch_size,
            "num_epochs": FLAGS.num_epochs,
            "min_epochs": FLAGS.min_epochs,
            "data_dir": FLAGS.data_dir,
            "regularize_rank": FLAGS.regularize_rank,
            "use_revin": FLAGS.use_revin,
            "use_instancenorm": FLAGS.use_instancenorm,
            "add_global_operator": FLAGS.add_global_operator,
            "add_control": FLAGS.add_control,
            "data_freq": FLAGS.data_freq,
            "dropout_rate": FLAGS.dropout_rate,
            "latent_dim": FLAGS.latent_dim,
            "num_steps": FLAGS.num_steps,
            "control_hidden_dim": FLAGS.control_hidden_dim,
            "num_layers": FLAGS.num_layers,
            "control_num_layers": FLAGS.control_num_layers,
            "jumps": FLAGS.jumps,
            "input_dim": FLAGS.input_dim,
            "input_length": FLAGS.input_length,
            "hidden_dim": FLAGS.hidden_dim,
            "train_output_length": FLAGS.train_output_length,
            "test_output_length": FLAGS.test_output_length,
            "num_heads": FLAGS.num_heads,
            "transformer_dim": FLAGS.transformer_dim,
            "transformer_num_layers": FLAGS.transformer_num_layers,
            "num_sins": FLAGS.num_sins,
            "num_poly": FLAGS.num_poly,
            "num_exp": FLAGS.num_exp,
        }

    else:
        raise NotImplementedError
    return flag_dict


def get_default_flag_values():
    flag_dict = {}
    for flag in ALL_FLAGS:
        flag_dict[flag.name] = flag.default

    return flag_dict


def extend_by_default_flag_values(flag_dict: dict) -> dict:
    flag_dict_defaults = get_default_flag_values()
    for key in flag_dict_defaults.keys():
        if key not in flag_dict.keys():
            flag_dict[key] = flag_dict_defaults[key]
    return flag_dict
