import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def nt_xent_loss(out, out_aug, batch_size, hidden_norm=False, temperature=1.0):
    """
    NT-Xent loss adapted from SimCLR.
    Compatible with TF 2.x using compat.v1
    """

    if hidden_norm:
        out = tf.nn.l2_normalize(out, -1)
        out_aug = tf.nn.l2_normalize(out_aug, -1)

    INF = 1e9
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    masks = tf.one_hot(tf.range(batch_size), batch_size)
    masksINF = masks * INF

    logits_aa = tf.matmul(out, out, transpose_b=True) / temperature
    logits_bb = tf.matmul(out_aug, out_aug, transpose_b=True) / temperature

    logits_aa = logits_aa - masksINF
    logits_bb = logits_bb - masksINF

    logits_ab = tf.matmul(out, out_aug, transpose_b=True) / temperature
    logits_ba = tf.matmul(out_aug, out, transpose_b=True) / temperature

    loss_a = tf.losses.softmax_cross_entropy(labels, tf.concat([logits_ab, logits_aa], axis=1))
    loss_b = tf.losses.softmax_cross_entropy(labels, tf.concat([logits_ba, logits_bb], axis=1))

    return loss_a + loss_b
