import tensorflow as tf
import tensorflow.contrib.layers as layers


# def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
#     # This model takes as input an observation and returns values of all actions
#     with tf.variable_scope(scope, reuse=reuse):
#         out = input
#
#         out = layers.fully_connected(out, num_outputs=num_units, activation_fn=None)
#         # nonlinear activation
#         out = tf.nn.relu(out)
#
#         out = layers.fully_connected(out, num_outputs=num_units, activation_fn=None)
#         # nonlinear activation
#         out = tf.nn.relu(out)
#
#         out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
#         return out


# def discriminator(x, scope, reuse=False, num_units=64, alpha=0.01, batch_norm=True):
#     with tf.variable_scope(scope, reuse=reuse):
#         if batch_norm:
#             h1 = layers.fully_connected(x, num_outputs=num_units, activation_fn=None, biases_initializer=None)
#             # for discriminator, always set is_training=True
#             # h1 = layers.batch_norm(h1, center=True, scale=True, is_training=True)
#             h1 = tf.layers.batch_normalization(h1, center=True, scale=True, training=True)
#         else:
#             h1 = layers.fully_connected(x, num_outputs=num_units, activation_fn=None)
#
#         # Leaky ReLU
#         h1 = tf.maximum(alpha * h1, h1)
#
#         if batch_norm:
#             h2 = layers.fully_connected(h1, num_outputs=num_units, activation_fn=None, biases_initializer=None)
#             # for discriminator, always set is_training=True
#             # h2 = layers.batch_norm(h2, center=True, scale=True, is_training=True)
#             h2 = tf.layers.batch_normalization(h2, center=True, scale=True, training=True)
#         else:
#             h2 = layers.fully_connected(h1, num_outputs=num_units, activation_fn=None)
#         # Leaky ReLU
#         h2 = tf.maximum(alpha * h2, h2)
#
#         logits = layers.fully_connected(h2, num_outputs=1, activation_fn=None)
#         out = tf.sigmoid(logits)
#         return out, logits


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, layer_norm=False, alpha=0.01):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input

        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=None)
        if layer_norm:
            print("Using layer_norm for actor...")
            # layers.batch_norm()
            out = layers.layer_norm(out, center=True, scale=True)

        # nonlinear activation
        # Leaky ReLU
        out = tf.maximum(alpha * out, out)
        # out = tf.nn.relu(out)

        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=None)

        if layer_norm:
            out = layers.layer_norm(out, center=True, scale=True)

        # nonlinear activation
        # Leaky ReLU
        out = tf.maximum(alpha * out, out)
        # out = tf.nn.relu(out)

        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def discriminator(x, scope, reuse=False, num_units=64, alpha=0.01, layer_norm=True):
    with tf.variable_scope(scope, reuse=reuse):

        h1 = layers.fully_connected(x, num_outputs=num_units, activation_fn=None)
        if layer_norm:
            h1 = layers.layer_norm(h1, center=True, scale=True)

        # Leaky ReLU
        h1 = tf.maximum(alpha * h1, h1)

        h2 = layers.fully_connected(h1, num_outputs=num_units, activation_fn=None)
        if layer_norm:
            h2 = layers.layer_norm(h2, center=True, scale=True)
        # Leaky ReLU
        h2 = tf.maximum(alpha * h2, h2)

        logits = layers.fully_connected(h2, num_outputs=1, activation_fn=None)
        out = tf.sigmoid(logits)
        return out, logits


def w_discriminator(x, scope, reuse=False, num_units=64, alpha=0.01, layer_norm=True):
    with tf.variable_scope(scope, reuse=reuse):

        h1 = layers.fully_connected(x, num_outputs=num_units, activation_fn=None)
        if layer_norm:
            h1 = layers.layer_norm(h1, center=True, scale=True)

        # Leaky ReLU
        h1 = tf.maximum(alpha * h1, h1)

        h2 = layers.fully_connected(h1, num_outputs=num_units, activation_fn=None)
        if layer_norm:
            h2 = layers.layer_norm(h2, center=True, scale=True)
        # Leaky ReLU
        h2 = tf.maximum(alpha * h2, h2)

        out = layers.fully_connected(h2, num_outputs=1, activation_fn=None)
        return out

def w2_discriminator(x, scope, reuse=False, num_units=64, alpha=0.01, layer_norm=True):
    with tf.variable_scope(scope, reuse=reuse):

        h1 = layers.fully_connected(x, num_outputs=num_units, activation_fn=None)
        if layer_norm:
            h1 = layers.layer_norm(h1, center=True, scale=True)

        # Leaky ReLU
        h1 = tf.maximum(alpha * h1, h1)

        h2 = layers.fully_connected(h1, num_outputs=num_units, activation_fn=None)
        if layer_norm:
            h2 = layers.layer_norm(h2, center=True, scale=True)
        # Leaky ReLU
        h2 = tf.maximum(alpha * h2, h2)

        out = layers.fully_connected(h2, num_outputs=1, activation_fn=None)
        out = tf.tanh(out)
        return out
