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


    # 新增四个 ablation 参数解析
    parser.add_argument('--use_homogeneity', type=eval, default=False, help='whether to use homogeneity info')
    parser.add_argument('--use_diversity', type=eval, default=False, help='whether to use diversity info')
    parser.add_argument('--use_align_loss', type=eval, default=False, help='whether to use v-t align loss')
    parser.add_argument('--use_residual', type=eval, default=False, help='whether to use residual')

    args = parser.parse_args()

    # 从命令行动态构建 config_dict
    config_dict = {
        'use_homogeneity': args.use_homogeneity,
        'use_diversity': args.use_diversity,
        'use_align_loss': args.use_align_loss,
        'use_residual': args.use_residual,
        'learning_rate': [0.0005],
        'reg_weight': 0.001,
        'mix_bpr_weight_loss': [0.1],
        'dragon_bpr_weight': [0.1],
        'align_weight_loss': [0.1],
        'diver_weight_loss': [0.1,],
    }
        # use_homogeneity: True  # 是否使用同质性信息
        # use_diversity: True    # 是否使用多样性信息
        # use_align_loss: True   # 是否使用 v_t_align_loss
        # use_residual: True     # 是否使用残差连接




    args, _ = parser.parse_known_args()
    print("config_dict", config_dict)
    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)


# nohup python main.py --dataset baby >log/test_add_dragonv3.log 2>&1 &