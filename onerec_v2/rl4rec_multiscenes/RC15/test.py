import os

import pandas as pd
from utility import to_pickled_df


if __name__ == '__main__':
    data_directory = 'data'

    sampled_sessions = pd.read_pickle(os.path.join(data_directory, 'sampled_sessions.df'))
    total_count=sampled_sessions.shape[0]
    purchase_count = sampled_sessions[sampled_sessions['is_buy'] ==1].shape[0]
    click_count=total_count-purchase_count
    print('total clicks: %d, total purchase:%d' % (click_count, purchase_count))

