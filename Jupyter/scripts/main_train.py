import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import pickle
import os
import gzip
import timeit
import argparse

from networks import C_VAE_NET, D_VAE_NET, C_GAN_NET, D_GAN_NET
from m3gan import m3gan
from utils import renormlizer


def load_npz_stats(stats_path):
    """Load normalization stats safely; return dict, missing keys set to None."""
    stats = np.load(stats_path, allow_pickle=True)
    keys = stats.files
    print(f"Keys inside {os.path.basename(stats_path)}: {keys}")
    return {
        "mean":       stats["mean"]       if "mean"       in keys else None,
        "std":        stats["std"]        if "std"        in keys else None,
        "feat_names": stats["feat_names"] if "feat_names" in keys else None,
        "min_val":    stats["min_val"]    if "min_val"    in keys else None,
        "max_val":    stats["max_val"]    if "max_val"    in keys else None,
    }


def maybe_save_npz_stats(stats_path, mean=None, std=None, min_val=None, max_val=None, feat_names=None):
    """Update or create norm_stats.npz with any provided arrays."""
    save_dict = {}
    if os.path.exists(stats_path):
        old = np.load(stats_path, allow_pickle=True)
        for k in old.files:
            save_dict[k] = old[k]

    if mean is not None:       save_dict["mean"] = mean
    if std is not None:        save_dict["std"] = std
    if min_val is not None:    save_dict["min_val"] = min_val
    if max_val is not None:    save_dict["max_val"] = max_val
    if feat_names is not None: save_dict["feat_names"] = feat_names

    np.savez(stats_path, **save_dict)
    print(f"Updated stats saved to {stats_path} with keys: {list(save_dict.keys())}")


def generate_large_data(model, num_sample, batch_size, time_steps, c_dim, d_dim,
                        conditional=False, labels=None):
    """
    Generate a very large number of samples by chunking.
    Uses np.memmap to avoid holding everything in RAM.
    Returns paths to saved .npy files.
    """
    rounds = num_sample // batch_size
    remainder = num_sample % batch_size

    # Memmap files (created on disk)
    out_dir = "data/fake/large_gen_tmp"
    os.makedirs(out_dir, exist_ok=True)
    c_path = os.path.join(out_dir, "c_gen_data_mm.npy")
    d_path = os.path.join(out_dir, "d_gen_data_mm.npy")

    c_mm = np.memmap(c_path, dtype=np.float32, mode='w+', shape=(num_sample, time_steps, c_dim))
    d_mm = np.memmap(d_path, dtype=np.float32, mode='w+', shape=(num_sample, time_steps, d_dim))

    idx = 0
    for r in range(rounds + (1 if remainder > 0 else 0)):
        this_bs = batch_size if r < rounds else remainder
        if this_bs == 0:
            break

        # Generate
        d_gen_data, c_gen_data = model.generate_data(num_sample=this_bs,
                                                     labels=labels[idx: idx + this_bs] if (conditional and labels is not None) else None)

        c_mm[idx: idx + this_bs] = c_gen_data
        d_mm[idx: idx + this_bs] = d_gen_data
        idx += this_bs

        if (r + 1) % 50 == 0:
            print(f"  Generated {idx}/{num_sample} samples...")

    # Flush to disk
    del c_mm
    del d_mm
    return c_path, d_path


def renorm_large_file(c_mm_path, num_sample, time_steps, c_dim, max_val, min_val, out_path):
    """
    Renormalize a memmapped continuous file in chunks and save final npy.
    We create another memmap, apply renorm per chunk, then save to npy.
    """
    chunk = 50000  # adjust if you have more memory
    c_in = np.memmap(c_mm_path, dtype=np.float32, mode='r', shape=(num_sample, time_steps, c_dim))
    c_out = np.memmap(out_path, dtype=np.float32, mode='w+', shape=(num_sample, time_steps, c_dim))

    for start in range(0, num_sample, chunk):
        end = min(start + chunk, num_sample)
        c_out[start:end] = renormlizer(c_in[start:end], max_val, min_val)

    del c_in
    del c_out
    return out_path


def main(args):
    # Reset TF graph
    tf.reset_default_graph()

    # Data paths
    real_data_path = os.path.join('ehrMGANdata', 'real', args.disease)

    # Load real data
    with gzip.open(os.path.join(real_data_path, 'vital_sign_48hrs.pkl'), 'rb') as f:
        continuous_x = pickle.load(f)['data']   # [N, T, c_dim]
    with gzip.open(os.path.join(real_data_path, 'med_interv_48hrs.pkl'), 'rb') as f:
        discrete_x = pickle.load(f)['data']     # [N, T, d_dim]

    # statics.pkl
    statics_path = os.path.join(real_data_path, 'statics.pkl')
    if os.path.exists(statics_path):
        try:
            with gzip.open(statics_path, 'rb') as f:
                statics_label = pickle.load(f)
            statics_label = np.asarray(statics_label).reshape([-1, 1])
        except Exception as e:
            print(" statics.pkl exists but cannot be loaded. Using zeros. Error:", e)
            statics_label = np.zeros((continuous_x.shape[0], 1))
    else:
        print(" statics.pkl is missing. Using zeros.")
        statics_label = np.zeros((continuous_x.shape[0], 1))

    # Shapes / params
    time_steps = continuous_x.shape[1]
    c_dim = continuous_x.shape[2]
    d_dim = discrete_x.shape[2]

    shared_latent_dim = 25
    c_z_size = shared_latent_dim
    d_z_size = shared_latent_dim
    c_noise_dim = max(int(c_dim / 2), 1)
    d_noise_dim = max(int(d_dim / 2), 1)

    # Build nets
    c_vae = C_VAE_NET(batch_size=args.batch_size, time_steps=time_steps, dim=c_dim, z_dim=c_z_size,
                      enc_size=args.enc_size, dec_size=args.dec_size,
                      enc_layers=args.enc_layers, dec_layers=args.dec_layers,
                      keep_prob=args.keep_prob, l2scale=args.l2_scale,
                      conditional=args.conditional, num_labels=args.num_labels)

    c_gan = C_GAN_NET(batch_size=args.batch_size, noise_dim=c_noise_dim, dim=c_dim,
                      gen_num_units=args.gen_num_units, gen_num_layers=args.gen_num_layers,
                      dis_num_units=args.dis_num_units, dis_num_layers=args.dis_num_layers,
                      keep_prob=args.keep_prob, l2_scale=args.l2_scale,
                      gen_dim=c_z_size, time_steps=time_steps,
                      conditional=args.conditional, num_labels=args.num_labels)

    d_vae = D_VAE_NET(batch_size=args.batch_size, time_steps=time_steps, dim=d_dim, z_dim=d_z_size,
                      enc_size=args.enc_size, dec_size=args.dec_size,
                      enc_layers=args.enc_layers, dec_layers=args.dec_layers,
                      keep_prob=args.keep_prob, l2scale=args.l2_scale,
                      conditional=args.conditional, num_labels=args.num_labels)

    d_gan = D_GAN_NET(batch_size=args.batch_size, noise_dim=d_noise_dim, dim=d_dim,
                      gen_num_units=args.gen_num_units, gen_num_layers=args.gen_num_layers,
                      dis_num_units=args.dis_num_units, dis_num_layers=args.dis_num_layers,
                      keep_prob=args.keep_prob, l2_scale=args.l2_scale,
                      gen_dim=d_z_size, time_steps=time_steps,
                      conditional=args.conditional, num_labels=args.num_labels)

    checkpoint_dir = "data/checkpoint/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    start = timeit.default_timer()

    # TF session config
    run_config = tf.ConfigProto()
    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        run_config.gpu_options.allow_growth = True
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    with tf.Session(config=run_config) as sess:
        model = m3gan(sess=sess,
                      batch_size=args.batch_size,
                      time_steps=time_steps,
                      num_pre_epochs=args.num_pre_epochs,
                      num_epochs=args.num_epochs,
                      checkpoint_dir=checkpoint_dir,
                      epoch_ckpt_freq=args.epoch_ckpt_freq,
                      epoch_loss_freq=args.epoch_loss_freq,
                      c_dim=c_dim, c_noise_dim=c_noise_dim,
                      c_z_size=c_z_size, c_data_sample=continuous_x,
                      c_vae=c_vae, c_gan=c_gan,
                      d_dim=d_dim, d_noise_dim=d_noise_dim,
                      d_z_size=d_z_size, d_data_sample=discrete_x,
                      d_vae=d_vae, d_gan=d_gan,
                      d_rounds=args.d_rounds, g_rounds=args.g_rounds, v_rounds=args.v_rounds,
                      v_lr_pre=args.v_lr_pre, v_lr=args.v_lr, g_lr=args.g_lr, d_lr=args.d_lr,
                      alpha_re=args.alpha_re, alpha_kl=args.alpha_kl, alpha_mt=args.alpha_mt,
                      alpha_ct=args.alpha_ct, alpha_sm=args.alpha_sm,
                      c_beta_adv=args.c_beta_adv, c_beta_fm=args.c_beta_fm,
                      d_beta_adv=args.d_beta_adv, d_beta_fm=args.d_beta_fm,
                      conditional=args.conditional, num_labels=args.num_labels,
                      statics_label=statics_label)

        model.build()
        model.train()

        # Big generation (1,000,000 by default):
        print(f"Generating {args.num_gen:,} samples...")
        c_path_mm, d_path_mm = generate_large_data(model,
                                                   num_sample=args.num_gen,
                                                   batch_size=args.batch_size,
                                                   time_steps=time_steps,
                                                   c_dim=c_dim,
                                                   d_dim=d_dim,
                                                   conditional=args.conditional,
                                                   labels=statics_label if args.conditional else None)

        import matplotlib.pyplot as plt

    # Load synthetic data (small batch for evaluation)
    c_gen_eval = np.memmap(c_path_mm, dtype=np.float32, mode='r', shape=(args.num_gen, time_steps, c_dim))[:1000]
    d_gen_eval = np.memmap(d_path_mm, dtype=np.float32, mode='r', shape=(args.num_gen, time_steps, d_dim))[:1000]

    # Load real data (already loaded earlier as continuous_x and discrete_x)
    c_real_eval = continuous_x[:1000]
    d_real_eval = discrete_x[:1000]

    # Flatten for comparison
    c_gen_flat = c_gen_eval.reshape(-1, c_gen_eval.shape[-1])
    c_real_flat = c_real_eval.reshape(-1, c_real_eval.shape[-1])
    d_gen_flat = d_gen_eval.reshape(-1, d_gen_eval.shape[-1])
    d_real_flat = d_real_eval.reshape(-1, d_real_eval.shape[-1])

    # Check for NaN or Inf in generated data
    print("Real contains NaN:", np.isnan(c_real_eval).any())
    print("Real contains inf:", np.isinf(c_real_eval).any())
    print("Synthetic contains NaN:", np.isnan(c_gen_eval).any())
    print("Synthetic contains inf:", np.isinf(c_gen_eval).any())

    #MMD function
    def compute_mmd(x, y, gamma=1.0, sample_size=1000):
        def rbf_kernel(a, b):
            a = np.asarray(a, dtype=np.float64)
            b = np.asarray(b, dtype=np.float64)
            sq_dist = np.sum(a**2, axis=1).reshape(-1, 1) + np.sum(b**2, axis=1) - 2 * np.dot(a, b.T)
            sq_dist = np.maximum(sq_dist, 0)
            return np.exp(-gamma * sq_dist)

        # Safety checks
        if np.isnan(x).any() or np.isinf(x).any():
            print("❌ x contains NaN or Inf!")
            return float('inf')
        if np.isnan(y).any() or np.isinf(y).any():
            print("❌ y contains NaN or Inf!")
            return float('inf')

        # Subsample (to avoid memory crash)
        n = min(len(x), len(y), sample_size)
        idx_x = np.random.choice(len(x), n, replace=False)
        idx_y = np.random.choice(len(y), n, replace=False)
        x_sample = x[idx_x]
        y_sample = y[idx_y]

        k_xx = rbf_kernel(x_sample, x_sample).mean()
        k_yy = rbf_kernel(y_sample, y_sample).mean()
        k_xy = rbf_kernel(x_sample, y_sample).mean()
        return k_xx + k_yy - 2 * k_xy

    # --- KL Divergence ---
    def kl_divergence(p, q, eps=1e-10):
        p = np.clip(p, eps, 1)
        q = np.clip(q, eps, 1)
        return np.sum(p * np.log(p / q))

    # --- Compute metrics ---
    mmd_score = compute_mmd(c_real_flat, c_gen_flat, sample_size=1000)
    kl_score = kl_divergence(d_real_flat.mean(0), d_gen_flat.mean(0))

    print(f"\n📊 Final MMD (continuous): {mmd_score:.4f}")
    print(f"📊 Final KL (discrete meds): {kl_score:.4f}")

    # --- Plot Histograms ---
    feat_names = np.arange(c_gen_flat.shape[-1])
    selected_feats = feat_names[:5]  # First 5 features

    for idx in selected_feats:
        plt.figure(figsize=(6, 4))
        plt.hist(c_real_flat[:, idx], bins=50, alpha=0.5, label="Real", density=True)
        plt.hist(c_gen_flat[:, idx], bins=50, alpha=0.5, label="Synthetic", density=True)
        plt.title(f"Feature {idx} Histogram")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # --- Plot medication frequencies ---
    med_top_idx = np.argsort(d_real_flat.mean(0))[-5:]
    for idx in med_top_idx:
        plt.figure(figsize=(4, 3))
        plt.bar(["Real", "Synthetic"], [d_real_flat[:, idx].mean(), d_gen_flat[:, idx].mean()],
                color=["blue", "orange"])
        plt.title(f"Medication Feature {idx}")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()



    # Normalization stats
    stats_path = os.path.join(real_data_path, "norm_stats.npz")
    if not os.path.exists(stats_path):
        print(f"{stats_path} not found. Creating new stats from raw continuous data.")
        mean_val = np.mean(continuous_x, axis=(0, 1))
        std_val  = np.std (continuous_x, axis=(0, 1)) + 1e-8
        min_val  = np.min (continuous_x, axis=(0, 1))
        max_val  = np.max (continuous_x, axis=(0, 1))
        maybe_save_npz_stats(stats_path, mean=mean_val, std=std_val,
                             min_val=min_val, max_val=max_val,
                             feat_names=np.arange(c_dim))
    stats = load_npz_stats(stats_path)

    print("\n✅ norm_stats.npz successfully loaded!")
    print("Available keys:", stats.keys())

    # Check and preview values
    if stats["mean"] is not None:
        print("▶ Mean (first 5):", stats["mean"][:5])
    else:
        print("❌ Mean is missing!")

    if stats["std"] is not None:
        print("▶ Std  (first 5):", stats["std"][:5])
    else:
        print("❌ Std is missing!")

    if stats["feat_names"] is not None:
        print("▶ Feature Names:", stats["feat_names"])
    else:
        print("❌ Feature names missing!")

    if stats["min_val"] is not None and stats["max_val"] is not None:
        print("▶ Min (first 5):", stats["min_val"][:5])
        print("▶ Max (first 5):", stats["max_val"][:5])
    else:
        print("❌ Min or Max is missing!")


    if stats["min_val"] is None or stats["max_val"] is None:
        print("min_val/max_val missing in stats. Computing from raw continuous data...")
        min_val_con = np.min(continuous_x, axis=(0, 1))
        max_val_con = np.max(continuous_x, axis=(0, 1))
        maybe_save_npz_stats(stats_path, min_val=min_val_con, max_val=max_val_con)
    else:
        min_val_con = stats["min_val"]
        max_val_con = stats["max_val"]

    # Renormalize the huge continuous memmap file and save final arrays
    final_dir = 'data/fake/'
    os.makedirs(final_dir, exist_ok=True)
    c_final_path = os.path.join(final_dir, "c_gen_data.npy")
    d_final_path = os.path.join(final_dir, "d_gen_data.npy")

    print("Renormalizing large continuous file...")
    renorm_large_file(c_mm_path=c_path_mm,
                      num_sample=args.num_gen,
                      time_steps=time_steps,
                      c_dim=c_dim,
                      max_val=max_val_con,
                      min_val=min_val_con,
                      out_path=c_final_path)

    # Move / copy discrete memmap to final location (no renorm needed for discrete)
    # We'll just copy into a regular .npy for convenience
    d_mm = np.memmap(d_path_mm, dtype=np.float32, mode='r', shape=(args.num_gen, time_steps, d_dim))
    np.save(d_final_path, d_mm)
    del d_mm

    stop = timeit.default_timer()
    print('✅ Time taken: ', stop - start)
    print(f"Saved:\n  Continuous: {c_final_path}\n  Discrete:   {d_final_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="real",
                        choices=['mimic', 'eicu', 'hirid', 'real'])
    parser.add_argument('--disease', type=str, default="mydata",
                        help='Subdirectory in ehrMGANdata/real/')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU index to use. -1 means CPU.')

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_pre_epochs', type=int, default=500)
    parser.add_argument('--num_epochs', type=int, default=800)
    parser.add_argument('--epoch_ckpt_freq', type=int, default=100)
    parser.add_argument('--epoch_loss_freq', type=int, default=100)
    parser.add_argument('--d_rounds', type=int, default=1)
    parser.add_argument('--g_rounds', type=int, default=3)
    parser.add_argument('--v_rounds', type=int, default=1)

    parser.add_argument('--v_lr_pre', type=float, default=0.0005)
    parser.add_argument('--v_lr', type=float, default=0.0001)
    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--d_lr', type=float, default=0.0001)

    parser.add_argument('--alpha_re', type=float, default=1)
    parser.add_argument('--alpha_kl', type=float, default=0.5)
    parser.add_argument('--alpha_mt', type=float, default=0.1)
    parser.add_argument('--alpha_ct', type=float, default=0.1)
    parser.add_argument('--alpha_sm', type=float, default=1)

    parser.add_argument('--c_beta_adv', type=float, default=1)
    parser.add_argument('--c_beta_fm', type=float, default=20)
    parser.add_argument('--d_beta_adv', type=float, default=1)
    parser.add_argument('--d_beta_fm', type=float, default=20)

    parser.add_argument('--enc_size', type=int, default=128)
    parser.add_argument('--dec_size', type=int, default=128)
    parser.add_argument('--enc_layers', type=int, default=3)
    parser.add_argument('--dec_layers', type=int, default=3)

    parser.add_argument('--gen_num_units', type=int, default=512)
    parser.add_argument('--gen_num_layers', type=int, default=3)
    parser.add_argument('--dis_num_units', type=int, default=256)
    parser.add_argument('--dis_num_layers', type=int, default=3)

    parser.add_argument('--keep_prob', type=float, default=0.8)
    parser.add_argument('--l2_scale', type=float, default=0.001)

    parser.add_argument('--conditional', type=bool, default=False)
    parser.add_argument('--num_labels', type=int, default=1)

    # NEW: how many synthetic samples to create after training
    parser.add_argument('--num_gen', type=int, default=1000000,
                        help='Number of synthetic samples to generate after training.')
    args = parser.parse_args()
    main(args)
