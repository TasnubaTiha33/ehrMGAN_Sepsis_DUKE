config = {
    "default_config": {
        "batch_size": 256,
        "num_pre_epochs": 300,
        "num_epochs": 500,
        "epoch_ckpt_freq": 50,
        "epoch_loss_freq": 50,

        # GAN training loops
        "d_rounds": 2,
        "g_rounds": 3,
        "v_rounds": 1,

        # Learning rates (slower = more stable)
        "v_lr_pre": 0.0005,
        "v_lr": 0.0001,
        "g_lr": 0.00005,
        "d_lr": 0.00005,

        # Loss weights
        "alpha_re": 1,
        "alpha_kl": 0.5,
        "alpha_mt": 0.1,
        "alpha_ct": 0.1,
        "alpha_sm": 1,

        # Continuous GAN strength (unchanged)
        "c_beta_adv": 1,
        "c_beta_fm": 20,

        # 🔧 Discrete GAN (stronger for rare med support)
        "d_beta_adv": 2,
        "d_beta_fm": 40,

        # Architecture
        "enc_size": 128,
        "dec_size": 128,
        "enc_layers": 3,
        "dec_layers": 3,
        "gen_num_units": 512,
        "gen_num_layers": 3,
        "dis_num_units": 256,
        "dis_num_layers": 3,

        "keep_prob": 0.8,
        "l2_scale": 0.001,
        "conditional": False,
        "num_labels": 1
    }
}
