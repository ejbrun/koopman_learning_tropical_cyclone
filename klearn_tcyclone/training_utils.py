"""Setting flags."""

from absl.flags import FlagValues

def set_flags(FLAGS: FlagValues)->dict:
    model = FLAGS.model
    year_range = [
        int(s) for s in FLAGS.year_range
    ]
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
        }

        # Full flag dictionary
        # output_dim = FLAGS.input_dim
        # flag_dict = {
        #     "input_dim": FLAGS.input_dim,
        #     "input_length": FLAGS.input_length,
        #     "latent_dim": FLAGS.latent_dim,
        #     "train_output_length": FLAGS.train_output_length,
        #     "num_steps": FLAGS.num_steps,
        #     "control_hidden_dim": FLAGS.control_hidden_dim,
        #     "control_num_layers": FLAGS.control_num_layers,
        #     "encoder_hidden_dim": FLAGS.hidden_dim,
        #     "decoder_hidden_dim": FLAGS.hidden_dim,
        #     "encoder_num_layers": FLAGS.num_layers,
        #     "decoder_num_layers": FLAGS.num_layers,
        #     "use_revin": FLAGS.use_revin,
        #     "use_instancenorm": FLAGS.use_instancenorm,
        #     "regularize_rank": FLAGS.regularize_rank,
        #     "add_global_operator": FLAGS.add_global_operator,
        #     "add_control": FLAGS.add_control,
        #     "output_dim": output_dim,
        #     "batch_size": FLAGS.batch_size,
        #     "num_epochs": FLAGS.num_epochs,
        #     "learning_rate": FLAGS.learning_rate,
        #     "freq": FLAGS.data_freq,
        #     "data_dir": FLAGS.data_dir,
        # }
    else:
        raise NotImplementedError
    return flag_dict