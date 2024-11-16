import os

import pandas as pd
from utility import to_pickled_df


if __name__ == '__main__':
    data_directory = 'data'

    sampled_clicks = pd.read_pickle(os.path.join(data_directory, 'sampled_clicks.df'))
    sampled_buys=pd.read_pickle(os.path.join(data_directory, 'sampled_buys.df'))

    sampled_clicks=sampled_clicks.drop(columns=['category'])
    sampled_buys=sampled_buys.drop(columns=['price','quantity'])

    sampled_clicks['is_buy']=0
    sampled_buys['is_buy']=1

    merge_session=pd.concat([sampled_clicks, sampled_buys], ignore_index=True)
    merge_session=merge_session.sort_values(by=['session_id','timestamp'])

    merge_session.to_csv('data/sampled_sessions.csv', index = None, header=True)

    to_pickled_df(data_directory, sampled_sessions=merge_session)