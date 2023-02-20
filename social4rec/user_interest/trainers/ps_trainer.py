# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class ParameterServerTrainer(object):

    def __init__(self,
                 dataset,
                 transform_functions,
                 train_fn,
                 valiate_steps,
                 log_steps,
                 learning_rate,
                 train_epochs=1,
                 save_checkpoints_dir=None,
                 restore_checkpoint_path=None,
                 validate_at_start=False):
        pass

    def run(self):
        pass
