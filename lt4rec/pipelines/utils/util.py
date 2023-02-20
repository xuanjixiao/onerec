# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def tf_global_config(intra_threads, inter_threads):
    import tensorflow as tf

    config = tf.ConfigProto(
        allow_soft_placement=True,
        intra_op_parallelism_threads=intra_threads,
        inter_op_parallelism_threads=inter_threads,
    )
    config.gpu_options.allow_growth = True
    tf.Session(config=config).__enter__()


def execute_shell_cmds(shell_cmds):
    import os

    print("Execute shell commands: %s" % shell_cmds)
    if os.system(shell_cmds) != 0:
        raise RuntimeError("Failed to execute shell commands: %s" % shell_cmds)
