import tensorflow as tf
import math
from utility import *

# config e.g. dilations: [1,4,16,] In most cases[1,4,] is enough
def nextitnet_residual_block(input_, dilation, layer_id,
                             residual_channels, kernel_size,
                             causal=True, train=True):
    resblock_type = "decoder"
    resblock_name = "nextitnet_residual_block{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
    with tf.variable_scope(resblock_name):
        dilated_conv = conv1d(input_, residual_channels,
                              dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv1"
                              )
        input_ln = normalize(dilated_conv, scope="layer_norm1")
        # input_ln=tf.contrib.layers.layer_norm(dilated_conv,reuse=not train, trainable=train)  #performance is not good, paramter wrong?
        relu1 = tf.nn.relu(input_ln)

        dilated_conv = conv1d(relu1, residual_channels,
                              2 * dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv2"
                              )

        input_ln = normalize(dilated_conv, scope="layer_norm2")
        # input_ln = tf.contrib.layers.layer_norm(dilated_conv, reuse=not train, trainable=train)
        relu1 = tf.nn.relu(input_ln)

        return input_ + relu1


def conv1d(input_, output_channels,
           dilation=1, kernel_size=1, causal=False,
           name="dilated_conv"):
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', [1, kernel_size, input_.get_shape()[-1], output_channels],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02, seed=1))
        bias = tf.get_variable('bias', [output_channels],
                               initializer=tf.constant_initializer(0.0))

        if causal:
            padding = [[0, 0], [(kernel_size - 1) * dilation, 0], [0, 0]]
            padded = tf.pad(input_, padding)
            input_expanded = tf.expand_dims(padded, dim=1)
            out = tf.nn.atrous_conv2d(input_expanded, weight, rate=dilation, padding='VALID') + bias
        else:
            input_expanded = tf.expand_dims(input_, dim=1)
            # out = tf.nn.atrous_conv2d(input_expanded, w, rate = dilation, padding = 'SAME') + bias
            out = tf.nn.conv2d(input_expanded, weight, strides=[1, 1, 1, 1], padding="SAME") + bias

        return tf.squeeze(out, [1])




























