import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def init_matrix(shape):
    """Return a normally initialized matrix of a given shape."""
    return tf.random_normal(shape, stddev=0.1)

def recurrent_unit_bilateral(input_dim, hidden_dim, scope_name, i):
    """Define the recurrent cell in the Bilateral LSTM."""

    params = []

    with tf.variable_scope(scope_name + str(i), reuse=tf.AUTO_REUSE):
        Wi = tf.Variable(init_matrix([input_dim, hidden_dim]), name='Wi_' + str(i))
        Ui = tf.Variable(init_matrix([hidden_dim, hidden_dim]), name='Ui_' + str(i))
        Vi = tf.Variable(init_matrix([hidden_dim, hidden_dim]), name='Vi_' + str(i))

        Wf = tf.Variable(init_matrix([input_dim, hidden_dim]), name='Wf_' + str(i))
        Uf = tf.Variable(init_matrix([hidden_dim, hidden_dim]), name='Uf_' + str(i))
        Vf = tf.Variable(init_matrix([hidden_dim, hidden_dim]), name='Vf_' + str(i))

        Wo = tf.Variable(init_matrix([input_dim, hidden_dim]), name='Wo_' + str(i))
        Uo = tf.Variable(init_matrix([hidden_dim, hidden_dim]), name='Uo_' + str(i))
        Vo = tf.Variable(init_matrix([hidden_dim, hidden_dim]), name='Vo_' + str(i))

        Wc = tf.Variable(init_matrix([input_dim, hidden_dim]), name='Wc_' + str(i))
        Uc = tf.Variable(init_matrix([hidden_dim, hidden_dim]), name='Uc_' + str(i))
        Vc = tf.Variable(init_matrix([hidden_dim, hidden_dim]), name='Vc_' + str(i))

    params.extend([Wi, Ui, Vi, Wf, Uf, Vf, Wo, Uo, Vo, Wc, Uc, Vc])

    def unit(x, hidden_memory_tm1, hidden_memory_tm2):
        previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)
        previous_hidden_state_, _ = tf.unstack(hidden_memory_tm2)

        # Input Gate
        i_gate = tf.sigmoid(tf.matmul(x, Wi) + tf.matmul(previous_hidden_state, Ui) + tf.matmul(previous_hidden_state_, Vi))

        # Forget Gate
        f_gate = tf.sigmoid(tf.matmul(x, Wf) + tf.matmul(previous_hidden_state, Uf) + tf.matmul(previous_hidden_state_, Vf))

        # Output Gate
        o_gate = tf.sigmoid(tf.matmul(x, Wo) + tf.matmul(previous_hidden_state, Uo) + tf.matmul(previous_hidden_state_, Vo))

        # Cell Candidate
        c_ = tf.nn.tanh(tf.matmul(x, Wc) + tf.matmul(previous_hidden_state, Uc) + tf.matmul(previous_hidden_state_, Vc))

        # Final Cell State and Hidden State
        c = f_gate * c_prev + i_gate * c_
        current_hidden_state = o_gate * tf.nn.tanh(c)

        return tf.stack([current_hidden_state, c])

    return unit
