import tensorflow as tf
import numpy as np
import pandas as pd
import os
import argparse
import trfl
from utility import *
from NextItNetModules import *
import pdb


data_directory = "./data"
# print(data_directory)
# replay_buffer = pd.read_pickle(os.path.join(data_directory, 'replay_buffer.df'))
# print("replay_buffer.shape:",replay_buffer.shape)
# eval_sessions=pd.read_pickle(os.path.join(data_directory, 'sampled_val.df'))
# eval_ids = eval_sessions.session_id.unique()
# print("len(eval_ids):",len(eval_ids))
# test_sessions=pd.read_pickle(os.path.join(data_directory, 'sampled_test.df'))
# test_ids = eval_sessions.session_id.unique()
# print("len(test_ids):",len(test_ids))




eval_sessions=pd.read_pickle(os.path.join(data_directory, 'sampled_val.df'))
eval_ids = eval_sessions.session_id.unique()
groups = eval_sessions.groupby('session_id')


pdb.set_trace()
print()