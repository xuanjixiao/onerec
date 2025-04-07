import os
import pandas as pd
import tensorflow as tf
from utility import to_pickled_df, pad_history

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('history_length',10,'uniform history length')

if __name__ == '__main__':

    data_directory = 'data'

    length=FLAGS.history_length

    # reply_buffer = pd.DataFrame(columns=['state','action','reward','next_state','is_done'])
    sorted_events=pd.read_pickle(os.path.join(data_directory, 'sorted_events.df'))
    item_ids=sorted_events.item_id.unique()
    pad_item=len(item_ids)

    train_sessions = pd.read_pickle(os.path.join(data_directory, 'sampled_train.df'))
    groups=train_sessions.groupby('session_id')
    ids=train_sessions.session_id.unique()

    state, len_state, action, is_buy, next_state, len_next_state, is_done = [], [], [], [], [],[],[]

    for id in ids:
        group=groups.get_group(id)
        history=[]
        for index, row in group.iterrows():
            s=list(history)
            len_state.append(length if len(s)>=length else 1 if len(s)==0 else len(s))
            s=pad_history(s,length,pad_item)
            a=row['item_id']
            is_b=row['is_buy']
            state.append(s)
            action.append(a)
            is_buy.append(is_b)
            history.append(row['item_id'])
            next_s=list(history)
            len_next_state.append(length if len(next_s)>=length else 1 if len(next_s)==0 else len(next_s))
            next_s=pad_history(next_s,length,pad_item)
            next_state.append(next_s)
            is_done.append(False)
        is_done[-1]=True

    dic={'state':state,'len_state':len_state,'action':action,'is_buy':is_buy,'next_state':next_state,'len_next_states':len_next_state,
         'is_done':is_done}
    reply_buffer=pd.DataFrame(data=dic)
    to_pickled_df(data_directory, replay_buffer=reply_buffer)

    dic={'state_size':[length],'item_num':[pad_item]}
    data_statis=pd.DataFrame(data=dic)
    to_pickled_df(data_directory,data_statis=data_statis)