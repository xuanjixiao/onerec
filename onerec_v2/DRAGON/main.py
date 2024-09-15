# coding: utf-8


"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='DRAGON', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')

    config_dict = {
        #'dropout': [0.2],
        #'reg_weight': [1e-04, 1e-03],
        # 'learning_rate': [0.000001],
        #'reg_weight': [0.0001,0.00001],
        #'n_layers': [2],
        #'reg_weight': [0.01],
        'learning_rate': [0.0001],
        'reg_weight': [0.001],
        'gpu_id': 0,
    }

    args, _ = parser.parse_known_args()
    print("config_dict", config_dict)
    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)


# nohup python main.py --dataset baby >log/chuchub_log.log &
