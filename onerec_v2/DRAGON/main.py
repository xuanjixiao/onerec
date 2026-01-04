# coding: utf-8

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""
import ast
import os
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="DRAGON", help="name of model")
    parser.add_argument("--dataset", "-d", type=str, default="sports_sparse", help="name of dataset")

    # Ablation 开关参数（True / False）
    parser.add_argument("--use_homogeneity", type=ast.literal_eval, default=True, help="whether to use homogeneity info (True/False)")
    parser.add_argument("--use_diversity", type=ast.literal_eval, default=True, help="whether to use diversity info (True/False)")
    parser.add_argument("--use_align_loss", type=ast.literal_eval, default=True, help="whether to use visual-text alignment loss (True/False)")
    parser.add_argument("--use_residual", type=ast.literal_eval, default=True, help="whether to use residual connection (True/False)")

    # List 与 Float 参数
    # parser.add_argument("--lr", type=ast.literal_eval, default=[0.0005], help='learning rate list, e.g. "[0.0005, 0.001]"')
    # parser.add_argument("--mix_bpr_weight_loss", type=ast.literal_eval, default=[0.1], help='mix BPR loss weight list, e.g. "[0.1, 0.2]"')
    # parser.add_argument("--dragon_bpr_weight", type=ast.literal_eval, default=[0.1], help='dragon BPR weight list, e.g. "[0.1, 0.2]"')
    # parser.add_argument("--align_weight_loss", type=ast.literal_eval, default=[0.1], help='alignment loss weight list, e.g. "[0.1, 0.2]"')
    # parser.add_argument("--diver_weight_loss", type=ast.literal_eval, default=[0.1], help='diversity loss weight list, e.g. "[0.1, 0.2]"')


    args = parser.parse_args()

    # # 从命令行动态构建 config_dict
    # config_dict = {
    #     'use_homogeneity': args.use_homogeneity,
    #     'use_diversity': args.use_diversity,
    #     'use_align_loss': args.use_align_loss,
    #     'use_residual': args.use_residual,
    #     'learning_rate': args.lr,
    #     'mix_bpr_weight_loss': args.mix_bpr_weight_loss,
    #     'dragon_bpr_weight': args.dragon_bpr_weight,
    #     'align_weight_loss': args.align_weight_loss,
    #     'diver_weight_loss': args.diver_weight_loss,
    # }



    if args.dataset == 'baby_sparse':
        config_dict = {
            'use_homogeneity': args.use_homogeneity,
            'use_diversity': args.use_diversity,
            'use_align_loss': args.use_align_loss,
            'use_residual': args.use_residual,
            'learning_rate': [0.05],
            'mix_bpr_weight_loss': [1.0],
            'dragon_bpr_weight': [0.01],
            'align_weight_loss': [0.2],
            'diver_weight_loss': [0.2],
            'seed': [999],
        }
    elif args.dataset == 'clothing_sparse':
        config_dict = {
            'use_homogeneity': args.use_homogeneity,
            'use_diversity': args.use_diversity,
            'use_align_loss': args.use_align_loss,
            'use_residual': args.use_residual,
            'learning_rate': [0.05],
            'mix_bpr_weight_loss': [0.001],
            'dragon_bpr_weight': [0.1],
            'align_weight_loss': [0.01],
            'diver_weight_loss': [0.1],
            'seed': [999],
        }
    elif args.dataset == 'sports_sparse':
        config_dict = {
            'use_homogeneity': args.use_homogeneity,
            'use_diversity': args.use_diversity,
            'use_align_loss': args.use_align_loss,
            'use_residual': args.use_residual,
            'learning_rate': [0.05],
            'mix_bpr_weight_loss': [0.1],
            'dragon_bpr_weight': [0.5],
            'align_weight_loss': [0.05],
            'diver_weight_loss': [0.001],
            'seed': [999],
        }

    # config_dict = {
    #     'gpu_id': 0,
    # }

    args, _ = parser.parse_known_args()
    print("config_dict", config_dict)
    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)


# nohup python main.py --dataset baby >log/test_add_dragonv3.log 2>&1 &