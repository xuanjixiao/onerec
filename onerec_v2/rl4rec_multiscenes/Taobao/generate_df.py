import pandas as pd
import numpy as np
import pdb
from sklearn.preprocessing import LabelEncoder
import os
# from utility import to_pickled_df, pad_history

data_dir = "./data"
os.system('rm ./data/*.df')
os.system('rm ./data/*.json')



def pad_history(itemlist,length,pad_item):
    if len(itemlist)>=length:
        return itemlist[-length:]
    if len(itemlist)<length:
        temp = [pad_item] * (length-len(itemlist))
        itemlist.extend(temp)
        return itemlist
    
def to_pickled_df(data_directory, **kwargs):
    for name, df in kwargs.items():
        # df.to_pickle(os.path.join(data_directory, name + '.df'))
        # df.to_csv(os.path.join(data_directory, name + '.csv'))
        df.to_json(os.path.join(data_directory, name + '.json'))


        
# read orignal file
raw_sample_df = pd.read_csv(os.path.join(data_dir,'raw_sample.csv'))
print(raw_sample_df.shape)
ad_feature = pd.read_csv(os.path.join(data_dir,'ad_feature.csv'))
print(ad_feature.shape)


# filt by len and sample session_id
info_df = pd.merge(left=raw_sample_df,right=ad_feature,how="left",left_on="adgroup_id",right_on="adgroup_id")
info_df['session_id'] = info_df['user'].astype(str) + info_df['time_stamp'].astype(str)
info_df = info_df[['time_stamp','adgroup_id','clk','cate_id','session_id']]
info_df['valid_session'] = info_df.session_id.map(info_df.groupby('session_id')['adgroup_id'].size() > 2)
sample_info_df = info_df.loc[info_df.valid_session].drop('valid_session', axis=1)
print(sample_info_df.shape)

sampled_session_id = np.random.choice(sample_info_df.session_id.unique(), 200000, replace=False)
sample_info_df = sample_info_df.loc[sample_info_df.session_id.isin(sampled_session_id)]
print(sample_info_df.shape)

# sort by session_id and time_stamp
sample_info_df = sample_info_df.sort_values(by=['session_id','time_stamp'])

# convert item_id and cate_id
cate_id_encoder = LabelEncoder()
adgroup_id_encoder = LabelEncoder()
sample_info_df['cate_id'] = sample_info_df['cate_id'].astype(int)
sample_info_df['adgroup_id'] = sample_info_df['adgroup_id'].astype(int)
sample_info_df['cate_id'] = cate_id_encoder.fit_transform(sample_info_df.cate_id)
sample_info_df['adgroup_id'] = adgroup_id_encoder.fit_transform(sample_info_df.adgroup_id)


# split data to train, eval, test
total_ids=sample_info_df.session_id.unique()
np.random.shuffle(total_ids)

fractions = np.array([0.8, 0.1, 0.1])
# split into 3 parts
train_ids, val_ids, test_ids = np.array_split(
    total_ids, (fractions[:-1].cumsum() * len(total_ids)).astype(int))

train_sessions=sample_info_df[sample_info_df['session_id'].isin(train_ids)]
val_sessions=sample_info_df[sample_info_df['session_id'].isin(val_ids)]
test_sessions=sample_info_df[sample_info_df['session_id'].isin(test_ids)]


# generate replay bufffer file
length=20

# reply_buffer = pd.DataFrame(columns=['state','action','reward','next_state','is_done'])
# sampled_sessions=pd.read_pickle(os.path.join(data_directory, 'sampled_sessions.df'))
item_ids=sample_info_df.adgroup_id.unique()
category_ids=sample_info_df.cate_id.unique()

pad_item=len(item_ids)
pad_category=len(category_ids)

# train_sessions = pd.read_pickle(os.path.join(data_directory, 'sampled_train.df'))
groups=train_sessions.groupby('session_id')
ids=train_sessions.session_id.unique()

state, len_state, action, true_item, is_click, next_state, len_next_state, is_done = [], [], [], [], [], [], [], []

for id in ids:
    group=groups.get_group(id)
    history=[]
    for index, row in group.iterrows():
        s=list(history)
        len_state.append(length if len(s)>=length else 1 if len(s)==0 else len(s))
        s=pad_history(s,length,pad_item)
        a=row['cate_id']
        is_c=row['clk']
        tmp_true_item = row['adgroup_id']
        state.append(s)
        action.append(a)
        true_item.append(tmp_true_item)
        is_click.append(is_c)
        history.append(row['adgroup_id'])
        next_s=list(history)
        len_next_state.append(length if len(next_s)>=length else 1 if len(next_s)==0 else len(next_s))
        next_s=pad_history(next_s,length,pad_item)
        next_state.append(next_s)
        is_done.append(False)
    is_done[-1]=True

# save final ret

dic={'state':state,'len_state':len_state,'action':action,'is_click':is_click,'next_state':next_state,'len_next_states':len_next_state,'is_done':is_done, 'true_item':true_item}

reply_buffer=pd.DataFrame(data=dic)
to_pickled_df(data_dir, replay_buffer=reply_buffer)

dic={'state_size':[length],'item_num':[pad_item], 'category_num':[pad_category]}
data_statis=pd.DataFrame(data=dic)
to_pickled_df(data_dir,data_statis=data_statis)

to_pickled_df(data_dir, sampled_train=train_sessions)
to_pickled_df(data_dir, sampled_val=val_sessions)
to_pickled_df(data_dir,sampled_test=test_sessions)

# check shape
print("reply_buffer.shape:",reply_buffer.shape)

train_ids = train_sessions.session_id.unique()
print("len(train_ids):",len(train_ids))

eval_ids = val_sessions.session_id.unique()
print("len(eval_ids):",len(eval_ids))

test_ids = test_sessions.session_id.unique()
print("len(test_ids):",len(test_ids))
