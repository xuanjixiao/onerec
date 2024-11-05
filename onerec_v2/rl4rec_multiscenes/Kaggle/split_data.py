import os
import numpy as np
import pandas as pd
from utility import to_pickled_df


if __name__ == '__main__':
    data_directory = 'data'
    # sampled_buys=pd.read_pickle(os.path.join(data_directory, 'sampled_buys.df'))
    #
    # buy_sessions=sampled_buys.session_id.unique()
    sorted_events = pd.read_pickle(os.path.join(data_directory, 'sorted_events.df'))

    total_sessions=sorted_events.session_id.unique()
    np.random.shuffle(total_sessions)

    fractions = np.array([0.8, 0.1, 0.1])
    # split into 3 parts
    train_ids, val_ids, test_ids = np.array_split(
        total_sessions, (fractions[:-1].cumsum() * len(total_sessions)).astype(int))

    train_sessions=sorted_events[sorted_events['session_id'].isin(train_ids)]
    val_sessions=sorted_events[sorted_events['session_id'].isin(val_ids)]
    test_sessions=sorted_events[sorted_events['session_id'].isin(test_ids)]

    to_pickled_df(data_directory, sampled_train=train_sessions)
    to_pickled_df(data_directory, sampled_val=val_sessions)
    to_pickled_df(data_directory,sampled_test=test_sessions)