# coding=utf8
import os
import sys
import collections
import re
import time
import argparse
from functools import partial as partial
from tensorflow import feature_column as fc
from sklearn.metrics import roc_auc_score
import yaml
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

os.system("unzip /var/dl/runtime/script/dnn.zip -d /var/dl/runtime/script/")


def tf_global_config(intra_threads, inter_threads):
    import tensorflow as tf

    config = tf.ConfigProto(
        allow_soft_placement=True,
        intra_op_parallelism_threads=intra_threads,
        inter_op_parallelism_threads=inter_threads,
    )
    config.gpu_options.allow_growth = True
    tf.Session(config=config).__enter__()


tf_global_config(intra_threads=8, inter_threads=8)
script_path = sys.argv[0][:sys.argv[0].rfind("/")]
configs = yaml.safe_load(open(script_path + '/config.yaml'))


# 路径地址
user_group_path = configs['user_ceph_path']
user_dnn = configs['user_dnn']
user_embedding = configs['user_embedding']
recall_ceph_path = configs['recall_ceph_path']
user_dnn_test = configs['user_dnn_test']
record_path = configs['save_fea']


LABEL_SIZE = 21
FIRST_CATE_SIZE = 32  # 上面的一级类目映射到[0,31],0代表一级类目未知
FIRST_CATE_EMB_SIZE = 16  # 一级类目embedding size
ITEM_DICT_DIR = 'item_dict'
UID_DICT_DIR = 'uid_dict'
TRAIN_DATA_DIR = 'train_data'
TEST_DATA_DIR = 'test_data'
MODEL_DIR = configs['save_model_path']
LOG_DIR = 'log'
PREDICT_DIR = 'predict'
PREDICT_OFFLINE_DIR = 'predict_offline'
PREDICT_DATA = 'train_data'
DUR_BOUNDARIES = [10, 15, 20, 25, 30, 35, 40,
                  50, 60, 70, 85, 100, 120, 150, 200, 240, 360]
EMBEDDING_NUM = 64


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', nargs='?', default='20190319')
    parser.add_argument('--data_dir', nargs='?', default='.')
    parser.add_argument('--mode', nargs='?', default='predict_offline',
                        help='train,predict,predict_offline')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--embed_size', type=int, default=32,
                        help='Embedding size.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--topk', type=int, default=20,
                        help='Evaluate the top k items.')
    parser.add_argument('--layers', nargs='?', default='[2048,1024,512]',
                        help="Size of each layer of mlp.")
    parser.add_argument('--step_num', type=int, default=10,
                        help='step for print loss and save model')
    return parser.parse_args()


def get_filenames(data_dir, pattern='part*'):
    filenames = []

    if os.path.isdir(data_dir) is False:
        return filenames

    for fi in os.listdir(data_dir):
        file_path = os.path.join(data_dir, fi)
        if os.path.isfile(file_path) and re.match(pattern, fi):
            filenames.append(file_path)

    return filenames


def get_feature_columns():
    srcItem_cate1 = tf.feature_column.categorical_column_with_hash_bucket(
        'FEA_SrcItemFirstCat', hash_bucket_size=128)
    item_cate1 = tf.feature_column.categorical_column_with_hash_bucket(
        'FEA_ItemFirstCat', hash_bucket_size=128)
    srcItem_cate2 = tf.feature_column.categorical_column_with_hash_bucket(
        'FEA_SrcItemSecondCat', hash_bucket_size=256)
    item_cate2 = tf.feature_column.categorical_column_with_hash_bucket(
        'FEA_ItemSecondCat', hash_bucket_size=256)
    cate_columns = tf.feature_column.shared_embedding_columns(
        [srcItem_cate1, item_cate1], dimension=4,
        initializer=tf.contrib.layers.xavier_initializer(), combiner='sum')
    cate2_columns = tf.feature_column.shared_embedding_columns(
        [srcItem_cate2, item_cate2], dimension=4,
        initializer=tf.contrib.layers.xavier_initializer(), combiner='sum')

    srcItem_cp = tf.feature_column.categorical_column_with_hash_bucket(
        'FEA_SrcItemCp', hash_bucket_size=256)
    item_cp = tf.feature_column.categorical_column_with_hash_bucket(
        'FEA_ItemCp', hash_bucket_size=256)
    cp_columns = tf.feature_column.shared_embedding_columns(
        [srcItem_cp, item_cp], dimension=4,
        initializer=tf.contrib.layers.xavier_initializer(), combiner='sum')

    srcItem_word = tf.feature_column.categorical_column_with_hash_bucket(
        'FEA_SrcItemKeywords', hash_bucket_size=10000)
    item_word = tf.feature_column.categorical_column_with_hash_bucket(
        'FEA_ItemKeywords', hash_bucket_size=10000)
    word_columns = tf.feature_column.shared_embedding_columns(
        [srcItem_word, item_word], dimension=16,
        initializer=tf.contrib.layers.xavier_initializer(), combiner='sum')

    # omgid = tf.feature_column.categorical_column_with_hash_bucket(
    #         'FEA_CtxUid', hash_bucket_size=10000)
    # omgid_column = tf.feature_column.embedding_column(omgid, dimension=16)

    # 用户主要属性
    # profession = tf.feature_column.indicator_column(
    #     tf.feature_column.categorical_column_with_vocabulary_list(
    #         'FEA_UserProfession', ['0', '1', '2', '3', '4', '5', '6', '7'], default_value=0))
    # residence = tf.feature_column.indicator_column(
    #     tf.feature_column.categorical_column_with_vocabulary_list(
    #         'FEA_UserResidenceType', ['-1', '32001002', '32001001'], default_value=0))
    # network = tf.feature_column.indicator_column(
    #     tf.feature_column.categorical_column_with_vocabulary_list(
    #         'FEA_UserNetType', ['0', '1', '2', '3', '4', '5'], default_value=0))
    # city = tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_hash_bucket(
    #     'FEA_UserCity', hash_bucket_size=512), dimension=4,combiner='mean')
    age = fc.indicator_column(fc.categorical_column_with_vocabulary_list(
        "FEA_UserAge", vocabulary_list=[-1]))
    # dev_mode = tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_hash_bucket(
    #     'FEA_UserDevMode', hash_bucket_size=8192), dimension=8)
    # parenting = tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_hash_bucket(
    #     'FEA_UserParenting', hash_bucket_size=40), dimension=2)
    gender = fc.indicator_column(
        fc.categorical_column_with_vocabulary_list("FEA_UserSex", vocabulary_list=['-1', '1', '2']))
    # ctxfeed = fc.indicator_column(
    # fc.categorical_column_with_vocabulary_list("FEA_CtxFeed",vocabulary_list=voc['FEA_CtxFeed']))
    education = fc.indicator_column(fc.categorical_column_with_vocabulary_list(
        "FEA_UserEducation", vocabulary_list=['10001', '10002', '10004', '10007', '10009', '10008', '10010', '-1']))
    province = fc.indicator_column(
        fc.categorical_column_with_vocabulary_list("FEA_UserIndustry", vocabulary_list=['-1']))
    # profession = fc.indicator_column(
    # fc.categorical_column_with_vocabulary_list("FEA_UserProfession",vocabulary_list=['-1']))
    grade = fc.indicator_column(fc.categorical_column_with_vocabulary_list(
        "FEA_UserCityGrade", vocabulary_list=['-1', '1', '2', '3', '4', '5']))
    cold = fc.indicator_column(
        fc.categorical_column_with_vocabulary_list("FEA_UserCold", vocabulary_list=['-1', '1', '0', '2']))
    # group = fc.indicator_column(
    # fc.categorical_column_with_vocabulary_list("FEA_UserGroup",vocabulary_list=voc['FEA_UserGroup']))
    status = fc.indicator_column(fc.categorical_column_with_vocabulary_list(
        "FEA_UserStatus", vocabulary_list=['-1']))
    # os_type = tf.feature_column.indicator_column(
    #     tf.feature_column.categorical_column_with_vocabulary_list(
    #         'FEA_CtxPlatform', ['android', 'ios'], default_value=0))
    group = fc.embedding_column(
        fc.categorical_column_with_hash_bucket("FEA_UserGroup", 5000, dtype=tf.string), 4, combiner='mean')
    app = fc.embedding_column(
        fc.categorical_column_with_hash_bucket("FEA_VIDAGE", 2000, dtype=tf.string), 4, combiner='mean')
    # qq = fc.embedding_column(
    # fc.categorical_column_with_hash_bucket("FEA_UserQQLiveTag",5000,dtype=tf.string),4,combiner='mean')
    # off = fc.embedding_column(
    # fc.categorical_column_with_hash_bucket("FEA_UserOffTagOut",5000,dtype=tf.string),4,combiner='mean')

    # UserCp_columns= fc.categorical_column_with_hash_bucket("FEA_UserCpTag",100000,dtype=tf.string)
    # ItemCp_columns= fc.categorical_column_with_hash_bucket("FEA_ItemCpTags",100000,dtype=tf.string)
    # CpTag_columns = tf.feature_column.shared_embedding_columns(
    #         [ItemCp_columns, UserCp_columns], dimension=32,
    #         initializer=tf.contrib.layers.xavier_initializer(), combiner='sum')

    srcItemTag_column = fc.categorical_column_with_hash_bucket(
        "FEA_SrcItemTag", 100000, dtype=tf.string)
    itemXfTag_column = fc.categorical_column_with_hash_bucket(
        "FEA_ItemXfTag", 100000, dtype=tf.string)
    Tag_column = tf.feature_column.shared_embedding_columns(
        [srcItemTag_column, itemXfTag_column], dimension=32,
        initializer=tf.contrib.layers.xavier_initializer(), combiner='sum')

    algid = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_hash_bucket(
            'FEA_ALGID', hash_bucket_size=64),
        dimension=4, combiner='sum')
    item_duration = tf.feature_column.numeric_column('FEA_ItemDuration')
    item_duration_buckets = tf.feature_column.bucketized_column(
        item_duration, boundaries=DUR_BOUNDARIES)
    srcItem_duration = tf.feature_column.numeric_column('FEA_SrcItemDuration')
    srcItem_duration_buckets = tf.feature_column.bucketized_column(
        srcItem_duration, boundaries=DUR_BOUNDARIES)

    # catWeight= fc.weighted_categorical_column(
    # fc.categorical_column_with_hash_bucket("FEA_UserCat",256,dtype=tf.string),"FEA_UserCatWeight")
    # catEmbed = fc.embedding_column(catWeight,4,combiner='mean')
    order_weight = fc.categorical_column_with_hash_bucket(
        "FEA_Order", 100, dtype=tf.string)
    order_embed = fc.embedding_column(order_weight, 4, combiner='mean')
    uid_weight = fc.categorical_column_with_hash_bucket(
        "FEA_Uid", 1000, dtype=tf.string)
    uid_embed = fc.embedding_column(uid_weight, 4, combiner='mean')
    gen_weight = fc.categorical_column_with_hash_bucket(
        "FEA_Cquality", 100, dtype=tf.int64)
    gen_embed = fc.embedding_column(gen_weight, 4, combiner='mean')

    tag_weight = fc.categorical_column_with_hash_bucket(
        "FEA_UserTag", 100000, dtype=tf.string)
    tagp_weight = fc.categorical_column_with_hash_bucket(
        "FEA_UserRTagPos", 100000, dtype=tf.string)
    tagn_weight = fc.categorical_column_with_hash_bucket(
        "FEA_UserRTagNeg", 100000, dtype=tf.string)
    tag_three_columns = tf.feature_column.shared_embedding_columns(
        [tag_weight, tagp_weight, tagn_weight], dimension=4,
        initializer=tf.contrib.layers.xavier_initializer(), combiner='sum')

    vid = tf.feature_column.categorical_column_with_hash_bucket(
        'FEA_SrcItemId', hash_bucket_size=20000)

    session_click_userOff = tf.feature_column.categorical_column_with_hash_bucket(
        'FEA_UserPvOff', hash_bucket_size=20000)
    session_click_userReal = tf.feature_column.categorical_column_with_hash_bucket(
        'FEA_UserPvReal', hash_bucket_size=20000)
    session_click_userMon = tf.feature_column.categorical_column_with_hash_bucket(
        'FEA_UserPvMonth', hash_bucket_size=20000)

    isPv = tf.feature_column.categorical_column_with_hash_bucket(
        'FEA_ItemId', hash_bucket_size=20000)
    vid_columns = tf.feature_column.shared_embedding_columns(
        [vid, isPv, session_click_userOff, session_click_userReal,
            session_click_userMon], dimension=16,
        initializer=tf.contrib.layers.xavier_initializer(), combiner='sum')

    # 三个视图
    srcitem = [vid_columns[0], cate2_columns[0], cate_columns[0], Tag_column[0],
               cp_columns[0], word_columns[0], algid, srcItem_duration_buckets, order_embed, gen_embed]
    item = [vid_columns[1], cate2_columns[1], cate_columns[1], Tag_column[1],
            cp_columns[1], word_columns[1], item_duration_buckets]
    user = [age, gender, education, province, grade, cold, status, group, app, uid_embed, tag_three_columns[0],
            tag_three_columns[1], tag_three_columns[2], vid_columns[2], vid_columns[3], vid_columns[4]]
    params = {'item': item, 'user': user, 'src': srcitem}
    return params


# def change_vocab():
#     vocab_path = './train_data/202004270000'
#     path = './train_data/voc.txt'
#     cmd = 'cat %s/*.gz > %s' % (vocab_path, path)
#     ret = os.system(cmd)
#     print("done")


def decode_train(example):
    features = tf.parse_single_example(example,
                                       features={
                                           "label": tf.FixedLenFeature([], tf.int64),
                                           "FEA_SrcItemId": tf.FixedLenFeature([], tf.string),
                                           "FEA_SrcItemCp": tf.FixedLenFeature([], tf.string),
                                           "FEA_SrcItemFirstCat": tf.FixedLenFeature([], tf.string),
                                           "FEA_SrcItemSecondCat": tf.FixedLenFeature([], tf.string),
                                           "FEA_ItemId": tf.FixedLenFeature([], tf.string),
                                           "FEA_ItemCp": tf.FixedLenFeature([], tf.string),
                                           "FEA_ItemFirstCat": tf.FixedLenFeature([], tf.string),
                                           "FEA_ItemSecondCat": tf.FixedLenFeature([], tf.string),
                                           "FEA_ItemDuration": tf.FixedLenFeature([], tf.int64),
                                           "FEA_VIDAGE": tf.FixedLenFeature([], tf.string),
                                           "FEA_ALGID": tf.FixedLenFeature([], tf.string),
                                           "FEA_SrcItemAgeDay": tf.FixedLenFeature([], tf.string),
                                           "FEA_SrcItemDuration": tf.FixedLenFeature([], tf.int64),
                                           "FEA_ItemAgeDay": tf.FixedLenFeature([], tf.string),
                                           "FEA_UserGroup": tf.FixedLenFeature([], tf.string),
                                           "FEA_UserAge": tf.FixedLenFeature([], tf.int64),
                                           "FEA_UserCold": tf.FixedLenFeature([], tf.string),
                                           "FEA_UserEducation": tf.FixedLenFeature([], tf.string),
                                           "FEA_UserSex": tf.FixedLenFeature([], tf.string),
                                           "FEA_UserIndustry": tf.FixedLenFeature([], tf.string),
                                           "FEA_UserStatus": tf.FixedLenFeature([], tf.string),
                                           "FEA_UserCityGrade": tf.FixedLenFeature([], tf.string),
                                           "FEA_SrcItemKeywords": tf.VarLenFeature(tf.string),
                                           "FEA_SrcItemTag": tf.VarLenFeature(tf.string),
                                           "FEA_ItemXfTag": tf.VarLenFeature(tf.string),
                                           "FEA_ItemKeywords": tf.VarLenFeature(tf.string),
                                           "FEA_UserTag": tf.VarLenFeature(tf.string),
                                           "FEA_UserRTagPos": tf.VarLenFeature(tf.string),
                                           "FEA_UserRTagNeg": tf.VarLenFeature(tf.string),
                                           "FEA_UserPvReal": tf.VarLenFeature(tf.string),
                                           "FEA_UserPvOff": tf.VarLenFeature(tf.string),
                                           "FEA_UserPvMonth": tf.VarLenFeature(tf.string),
                                           "FEA_Order": tf.FixedLenFeature([], tf.string),
                                           "FEA_Uid": tf.FixedLenFeature([], tf.string),
                                           "FEA_Cquality": tf.FixedLenFeature([], tf.int64),
                                       })

    other_features = {}
    other_features["FEA_Order"] = features['FEA_Order']
    other_features["FEA_Uid"] = features['FEA_Uid']
    other_features["FEA_Cquality"] = features['FEA_Cquality']
    other_features["FEA_UserGroup"] = features['FEA_UserGroup']
    other_features["FEA_UserPvOff"] = features['FEA_UserPvOff']
    other_features["FEA_UserPvReal"] = features['FEA_UserPvReal']
    other_features["FEA_UserPvMonth"] = features['FEA_UserPvMonth']
    other_features["FEA_SrcItemCp"] = features['FEA_SrcItemCp']
    other_features["FEA_SrcItemId"] = features['FEA_SrcItemId']
    other_features["FEA_SrcItemFirstCat"] = features['FEA_SrcItemFirstCat']
    other_features["FEA_SrcItemSecondCat"] = features['FEA_SrcItemSecondCat']
    other_features["FEA_ItemId"] = features['FEA_ItemId']
    other_features["FEA_ItemCp"] = features['FEA_ItemCp']
    other_features["FEA_VIDAGE"] = features['FEA_VIDAGE']
    other_features["FEA_ALGID"] = features['FEA_ALGID']
    other_features["FEA_SrcItemAgeDay"] = features['FEA_SrcItemAgeDay']
    other_features["FEA_SrcItemDuration"] = features['FEA_SrcItemDuration']
    other_features["FEA_ItemAgeDay"] = features['FEA_ItemAgeDay']
    other_features["FEA_ItemKeywords"] = features['FEA_ItemKeywords']
    other_features["FEA_SrcItemKeywords"] = features['FEA_SrcItemKeywords']
    other_features["FEA_SrcItemTag"] = features['FEA_SrcItemTag']
    other_features["FEA_UserTag"] = features['FEA_UserTag']
    other_features["FEA_UserRTagPos"] = features['FEA_UserRTagPos']
    other_features["FEA_UserRTagNeg"] = features['FEA_UserRTagNeg']
    other_features["FEA_UserCold"] = features['FEA_UserCold']
    other_features["FEA_UserCityGrade"] = features['FEA_UserCityGrade']
    other_features["FEA_UserEducation"] = features['FEA_UserEducation']
    other_features["FEA_UserIndustry"] = features["FEA_UserIndustry"]
    other_features["FEA_UserStatus"] = features["FEA_UserStatus"]
    other_features["FEA_UserSex"] = features["FEA_UserSex"]
    other_features["FEA_UserAge"] = features["FEA_UserAge"]
    other_features["FEA_ItemFirstCat"] = features["FEA_ItemFirstCat"]
    other_features["FEA_ItemSecondCat"] = features["FEA_ItemSecondCat"]
    other_features["FEA_ItemXfTag"] = features["FEA_ItemXfTag"]
    other_features["FEA_ItemDuration"] = features["FEA_ItemDuration"]
    labels = features['label']
    return other_features, labels, features['FEA_Uid']


def decode_predict(example):
    features = tf.parse_single_example(example,
                                       features={
                                           "label": tf.FixedLenFeature([], tf.int64),
                                           "FEA_SrcItemId": tf.FixedLenFeature([], tf.string),
                                           "FEA_SrcItemCp": tf.FixedLenFeature([], tf.string),
                                           "FEA_SrcItemFirstCat": tf.FixedLenFeature([], tf.string),
                                           "FEA_SrcItemSecondCat": tf.FixedLenFeature([], tf.string),
                                           "FEA_ItemId": tf.FixedLenFeature([], tf.string),
                                           "FEA_ItemCp": tf.FixedLenFeature([], tf.string),
                                           "FEA_ItemFirstCat": tf.FixedLenFeature([], tf.string),
                                           "FEA_ItemSecondCat": tf.FixedLenFeature([], tf.string),
                                           "FEA_ItemDuration": tf.FixedLenFeature([], tf.int64),
                                           "FEA_VIDAGE": tf.FixedLenFeature([], tf.string),
                                           "FEA_ALGID": tf.FixedLenFeature([], tf.string),
                                           "FEA_SrcItemAgeDay": tf.FixedLenFeature([], tf.string),
                                           "FEA_SrcItemDuration": tf.FixedLenFeature([], tf.int64),
                                           "FEA_ItemAgeDay": tf.FixedLenFeature([], tf.string),
                                           "FEA_UserGroup": tf.FixedLenFeature([], tf.string),
                                           "FEA_UserAge": tf.FixedLenFeature([], tf.int64),
                                           "FEA_UserCold": tf.FixedLenFeature([], tf.string),
                                           "FEA_UserEducation": tf.FixedLenFeature([], tf.string),
                                           "FEA_UserSex": tf.FixedLenFeature([], tf.string),
                                           "FEA_UserIndustry": tf.FixedLenFeature([], tf.string),
                                           "FEA_UserStatus": tf.FixedLenFeature([], tf.string),
                                           "FEA_UserCityGrade": tf.FixedLenFeature([], tf.string),
                                           "FEA_SrcItemKeywords": tf.VarLenFeature(tf.string),
                                           "FEA_SrcItemTag": tf.VarLenFeature(tf.string),
                                           "FEA_ItemXfTag": tf.VarLenFeature(tf.string),
                                           "FEA_ItemKeywords": tf.VarLenFeature(tf.string),
                                           "FEA_UserTag": tf.VarLenFeature(tf.string),
                                           "FEA_UserRTagPos": tf.VarLenFeature(tf.string),
                                           "FEA_UserRTagNeg": tf.VarLenFeature(tf.string),
                                           "FEA_UserPvReal": tf.VarLenFeature(tf.string),
                                           "FEA_UserPvOff": tf.VarLenFeature(tf.string),
                                           "FEA_UserPvMonth": tf.VarLenFeature(tf.string),
                                           "FEA_Order": tf.FixedLenFeature([], tf.string),
                                           "FEA_Uid": tf.FixedLenFeature([], tf.string),
                                           "FEA_Cquality": tf.FixedLenFeature([], tf.int64),
                                       })

    other_features = {}
    other_features["FEA_Order"] = features['FEA_Order']
    other_features["FEA_Uid"] = features['FEA_Uid']
    other_features["FEA_Cquality"] = features['FEA_Cquality']
    other_features["FEA_UserGroup"] = features['FEA_UserGroup']
    other_features["FEA_UserPvOff"] = features['FEA_UserPvOff']
    other_features["FEA_UserPvReal"] = features['FEA_UserPvReal']
    other_features["FEA_UserPvMonth"] = features['FEA_UserPvMonth']
    other_features["FEA_SrcItemCp"] = features['FEA_SrcItemCp']
    other_features["FEA_SrcItemId"] = features['FEA_SrcItemId']
    other_features["FEA_SrcItemFirstCat"] = features['FEA_SrcItemFirstCat']
    other_features["FEA_SrcItemSecondCat"] = features['FEA_SrcItemSecondCat']
    other_features["FEA_ItemId"] = features['FEA_ItemId']
    other_features["FEA_ItemCp"] = features['FEA_ItemCp']
    other_features["FEA_VIDAGE"] = features['FEA_VIDAGE']
    other_features["FEA_ALGID"] = features['FEA_ALGID']
    other_features["FEA_SrcItemAgeDay"] = features['FEA_SrcItemAgeDay']
    other_features["FEA_SrcItemDuration"] = features['FEA_SrcItemDuration']
    other_features["FEA_ItemAgeDay"] = features['FEA_ItemAgeDay']
    other_features["FEA_ItemKeywords"] = features['FEA_ItemKeywords']
    other_features["FEA_SrcItemKeywords"] = features['FEA_SrcItemKeywords']
    other_features["FEA_SrcItemTag"] = features['FEA_SrcItemTag']
    other_features["FEA_UserTag"] = features['FEA_UserTag']
    other_features["FEA_UserRTagPos"] = features['FEA_UserRTagPos']
    other_features["FEA_UserRTagNeg"] = features['FEA_UserRTagNeg']
    other_features["FEA_UserCold"] = features['FEA_UserCold']
    other_features["FEA_UserCityGrade"] = features['FEA_UserCityGrade']
    other_features["FEA_UserEducation"] = features['FEA_UserEducation']
    other_features["FEA_UserIndustry"] = features["FEA_UserIndustry"]
    other_features["FEA_UserStatus"] = features["FEA_UserStatus"]
    other_features["FEA_UserSex"] = features["FEA_UserSex"]
    other_features["FEA_UserAge"] = features["FEA_UserAge"]
    other_features["FEA_ItemFirstCat"] = features["FEA_ItemFirstCat"]
    other_features["FEA_ItemSecondCat"] = features["FEA_ItemSecondCat"]
    other_features["FEA_ItemXfTag"] = features["FEA_ItemXfTag"]
    other_features["FEA_ItemDuration"] = features["FEA_ItemDuration"]
    labels = features['label']
    return other_features, labels, features['FEA_Uid']


class Model(object):
    def __init__(self, args, labels, other_features, total_user_id, b_uid, item_emb_=None):
        self.batch_size = args.batch_size
        self.layers_unit = eval(args.layers)
        self.embed_size = args.embed_size
        self.neg_samples = 4
        self.lr = args.lr
        self.labels = tf.cast(labels, tf.float32)
        self.other_features = other_features
        self.user_id = tf.constant(total_user_id)
        self.train_user = b_uid
        self.user_feature = tf.placeholder(tf.float32, shape=(None, 72))
        self.table1 = tf.contrib.lookup.index_table_from_tensor(mapping=self.user_id,
                                                                num_oov_buckets=10, default_value=-1)

    def cus_nn(self, net, mode, deep_net, is_training):
        # regularizer = tf.contrib.layers.l2_regularizer(scale=args.l2)
        for para in deep_net:
            net = tf.layers.dense(
                inputs=net,
                units=para,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                activation=partial(tf.nn.leaky_relu, alpha=0.2),
                use_bias=True,
                # kernel_regularizer=regularizer
            )
            net = tf.layers.dropout(net, rate=0.2, training=is_training)
            # net = tf.layers.batch_normalization(net)
        return net

    def build_model(self, is_training=True):
        columns = get_feature_columns()
        item_columns = fc.input_layer(self.other_features, columns['item'])
        user_columns = fc.input_layer(self.other_features, columns['user'])
        # 查找同ID的用户聚类特征
        self.user_findid = self.table1.lookup(
            self.other_features['FEA_CtxUid'])
        self.user_embedding = tf.gather(self.user_feature, self.user_findid)

        # 用户四种关系分离并进行attention
        user_split = tf.split(self.user_embedding, 4, 1)
        rela = tf.stack(user_split, 1)
        rela_conv = tf.layers.conv1d(rela, 1, 1)
        coef = tf.expand_dims(user_columns, 1)
        coef = tf.layers.conv1d(coef, 1, 1)
        out = tf.multiply(coef, rela_conv)
        coefs = tf.nn.softmax(tf.nn.tanh(out), 1)
        res = tf.multiply(coefs, rela)
        res = tf.reduce_sum(res, 1)

        item = self.cus_nn(item_columns, None, [
                           EMBEDDING_NUM * 2, EMBEDDING_NUM], is_training)
        norm_item = tf.sqrt(tf.reduce_sum(tf.square(item), 1, True))
        item_emb = tf.truediv(item, norm_item)
        # for column in sorted(params['user_columns'], key=lambda x: x.name):
        #     print(column.name)

        # 产出item embedding
        if args.mode == 'sample':
            self.prediction = {
                'vid': self.other_features['FEA_SrcItemId'],
                'item': item_emb,
            }
        else:
            user = self.cus_nn(user_columns, None, [
                               EMBEDDING_NUM * 4, EMBEDDING_NUM * 2, EMBEDDING_NUM], is_training)

            # attenton embedding拼接后送入全连接层
            user_columns_out = tf.concat([user_columns, res], axis=1)
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
            user = tf.layers.dense(
                inputs=user_columns_out,
                units=EMBEDDING_NUM,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                activation=partial(tf.nn.leaky_relu, alpha=0.2),
                use_bias=True,
                kernel_regularizer=regularizer,
            )
            # 残差连接
            # user = user + net
            # user =  tf.concat([res, user],axis=1)
            # user = self.cus_nn(user, None, [EMBEDDING_NUM], is_training)

            norm_user = tf.sqrt(tf.reduce_sum(tf.square(user), 1, True))
            user_emb = tf.truediv(user, norm_user)
            self.cos_sim_raw = tf.reduce_sum(
                tf.multiply(user_emb, item_emb), 1, True)

            self.prob = tf.nn.sigmoid(self.cos_sim_raw)
            # self.hit_prob = tf.slice(self.prob, [0, 0], [-1, 1])

    def train(self):
        self.loss_total = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.cos_sim_raw, labels=tf.expand_dims(self.labels, 1)
        )
        self.loss_total = tf.expand_dims(self.labels, 1) * self.loss_total + 0.5 * (
            1-tf.expand_dims(self.labels, 1)) * self.loss_total

        # AUC loss
        y_row = tf.reshape(self.labels, (1, -1))
        y_scores = tf.reshape(self.prob, (1, -1))
        y_mask = tf.tile(y_row, [y_row.shape[1], 1])
        y_column = tf.reshape(y_row, (-1, 1))
        y_mask = y_mask + y_column
        M = tf.reduce_sum(y_row)
        N = tf.cast(tf.shape(y_row)[1], tf.float32) - M
        y_mask = tf.cast(tf.equal(y_mask, 1.0), tf.float64)
        y_scores_log = tf.log(y_scores/(1.0-y_scores))
        y_column = self.labels*y_scores_log
        y_row = (self.labels-1) * y_scores_log
        y_row = tf.tile(y_row, [y_row.shape[1], 1])
        y_column = tf.reshape(y_column, (-1, 1))
        y_log = y_column + y_row
        y_log = tf.cast(y_log, tf.float64)
        y_final = tf.multiply(y_log, y_mask)
        y_final = tf.clip_by_value(y_final, -100, 0)
        y_final = tf.square(y_final)
        y_final = y_final / tf.cast(M*N, tf.float64)
        loss = tf.reduce_sum(y_final)

        # 分配AUC loss的全局权重
        self.loss_merge = tf.cast(self.loss_total, tf.float64) * (1+loss)
        self.loss = tf.reduce_mean(self.loss_merge)
        tf.summary.scalar('loss', self.loss)

        correct_prediction = tf.equal(
            tf.cast(self.prob > 0.5, tf.float32), tf.expand_dims(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', self.loss)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.opt = tf.train.AdamOptimizer(
                learning_rate=self.lr).minimize(loss=self.loss)

    def predict(self):
        # self.loss = -tf.reduce_mean(tf.log(tf.clip_by_value(self.hit_prob, 1e-8, 1)))  # / args.batch_size
        # tf.summary.scalar('loss', self.loss)
        # one = tf.constant(1, dtype=tf.int64)
        # self.labels = one - self.labels
        # correct_prediction = tf.equal(tf.argmax(self.prob, 1), self.labels)
        # self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # tf.summary.scalar('loss', self.loss)

        correct_prediction = tf.equal(
            tf.cast(self.prob > 0.5, tf.float32), tf.expand_dims(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def get_features(data_dir, batch_size, mode='train'):
    decode = decode_train if mode == 'train' else decode_predict
    sloppy1 = True if mode == 'train' else False
    drop_remainder = True if mode == 'train' else False
    with tf.name_scope('input'):
        files = tf.data.Dataset.list_files(data_dir)
        ds = files.apply(tf.data.experimental.parallel_interleave(lambda f: tf.data.TFRecordDataset(
            f, "GZIP"
        ), cycle_length=8, sloppy=sloppy1))
        ds = ds.shuffle(batch_size)
        ds = ds.map(decode, num_parallel_calls=8).batch(
            batch_size, drop_remainder).prefetch(1)
        iterator = ds.make_initializable_iterator()
    return iterator


def _parse_record(example_photo):
    features = {
        'omgid': tf.FixedLenFeature([], tf.string),
        'raw_feature': tf.FixedLenFeature([], tf.string),
    }
    parsed_features = tf.parse_single_example(example_photo, features=features)

    return parsed_features['omgid'], tf.decode_raw(parsed_features['raw_feature'], tf.float32)


def get_features_user(mode='train'):
    with tf.name_scope('input'):
        files = tf.data.Dataset.list_files(user_embedding)
        ds = files.apply(tf.contrib.data.parallel_interleave(
            tf.data.TFRecordDataset, cycle_length=8))
        ds = ds.map(_parse_record, num_parallel_calls=8).batch(
            100000).prefetch(1)
        iterator = ds.make_initializable_iterator()
    return iterator


def _parse_record_(example_photo):
    features = {
        'vid': tf.FixedLenFeature([], tf.string),
        'raw_feature': tf.FixedLenFeature([], tf.string),
    }
    parsed_features = tf.parse_single_example(example_photo, features=features)

    return parsed_features['vid'], tf.decode_raw(parsed_features['raw_feature'], tf.float32)


def get_features_item(mode='train'):
    with tf.name_scope('input'):
        files = tf.data.Dataset.list_files(record_path)
        ds = files.apply(tf.contrib.data.parallel_interleave(
            tf.data.TFRecordDataset, cycle_length=8))
        ds = ds.map(_parse_record_, num_parallel_calls=8).batch(
            100000).prefetch(1)
        iterator = ds.make_initializable_iterator()
    return iterator


def find_key_by_value(val2key_dict, values):
    key = [val2key_dict[x] for x in values]
    return key


def get_yesterday(now_str):
    from datetime import datetime, timedelta
    a = datetime.strptime(now_str, "%Y%M%d")
    y = a - timedelta(2)
    return datetime.strftime(y, "%Y%M%d")


def del_old_mk_new(dir_name, args):
    predict_item = "%s/%s/%s" % (args.data_dir, dir_name, args.date)
    os.system("rm -r %s" % predict_item)
    os.system("mkdir -p %s" % predict_item)
    yesterday_predict_dir = "%s/%s/%s" % (args.data_dir,
                                          dir_name, get_yesterday(args.date))
    os.system("rm -r %s" % yesterday_predict_dir)
    predict_path = "%s/predict.txt" % (predict_item)
    os.system("rm %s" % predict_path)
    return predict_path


def train(args):
    iters = get_features_user()
    step_num = args.step_num
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    config = tf.ConfigProto(gpu_options=gpu_options)

    total_user_id = []
    total_user_feature = []
    with tf.Session(config=config) as sess:
        sess.run(tf.group(tf.global_variables_initializer(),
                          tf.local_variables_initializer()))
        sess.run(iters.initializer)

        while True:
            try:
                user_data = sess.run([iters.get_next()])
                user_id, user_feature = user_data[0][0], user_data[0][1]
                if len(total_user_id) == 0:
                    total_user_id = user_id
                    total_user_feature = user_feature
                else:
                    total_user_feature = np.concatenate(
                        (total_user_feature, user_feature), 0)
                    total_user_id = np.concatenate((total_user_id, user_id), 0)

            except tf.errors.OutOfRangeError:
                print('finish user embedding load')
                break
        # 获取用户聚类的特征
        print(total_user_id.shape)
        print(total_user_feature.shape)

        model_path = "%s/%s/youtubednn" % (MODEL_DIR, args.date)
        data_iter = get_features(user_dnn, args.batch_size, args.mode)
        data = data_iter.get_next()
        other_features, b_labels, b_uid = data
        model = Model(args, b_labels, other_features, total_user_id, b_uid)
        model.build_model(is_training=True)
        model.train()
        saver = tf.train.Saver()
        sess.run(tf.tables_initializer())
        sess.run(tf.group(tf.global_variables_initializer(),
                          tf.local_variables_initializer()))

        for e in range(args.epochs):
            sess.run(data_iter.initializer)  # 重复多个epoch,重置数据迭代器
            epoch_loss = 0.0
            step = 0
            epoch_acc = 0
            total_auc = 0
            start_time = time.time()
            tf.logging.info("start time: %d" % start_time)
            while True:
                try:
                    loss, acc, _, prob, labels = sess.run([model.loss, model.accuracy,
                                                          model.opt, model.prob, model.labels], feed_dict={
                                                          model.user_feature: total_user_feature})
                    prob = np.reshape(np.array(prob), [1, -1])
                    labels = np.reshape(np.array(labels), [1, -1])
                    try:
                        auc = roc_auc_score(labels[0], prob[0])
                    except ValueError:
                        pass
                    total_auc += auc
                    epoch_acc += acc
                    if step % step_num == 0:
                        cost_time = time.time() - start_time
                        tf.logging.info("step %d loss:%.5f, cost time: %d s, acc:%.5f,auc:%.5f " % (
                            step, loss, cost_time, epoch_acc/(step+1), total_auc/(step+1)))
                        start_time = time.time()
                        saver.save(sess, model_path)
                        tf.train.write_graph(sess.graph.as_graph_def(
                        ), model_path, 'youtubednn.pbtxt', as_text=True)
                    step += 1
                    epoch_loss += loss

                except tf.errors.OutOfRangeError:
                    cost_time = time.time()-start_time
                    tf.logging.info("Epoch %d loss: %.5f, cost time: %d s, step:%d, acc:%.5f " % (
                        e, epoch_loss/step, cost_time, step, epoch_acc/step))
                    saver.save(sess, model_path)
                    tf.train.write_graph(sess.graph.as_graph_def(
                    ), model_path, 'youtubednn.pbtxt', as_text=True)
                    break


def get_item_dict_recall(data_dir):
    item_dict = collections.defaultdict(list)
    if os.path.isfile(data_dir):
        with open(data_dir, 'r') as fp:
            for line in fp:
                arr = line.strip().split(',')
                if arr[1] == '':
                    continue
                item_dict[arr[0]].append(arr[1:])
    return item_dict


def predict(args):
    iters = get_features_user()
    total_user_id = []
    total_user_feature = []
    with tf.Session() as sess:
        sess.run(tf.group(tf.global_variables_initializer(),
                          tf.local_variables_initializer()))
        sess.run(iters.initializer)

        while True:
            try:
                user_data = sess.run([iters.get_next()])
                user_id, user_feature = user_data[0][0], user_data[0][1]
                if len(total_user_id) == 0:
                    total_user_id = user_id
                    total_user_feature = user_feature
                else:
                    total_user_feature = np.concatenate(
                        (total_user_feature, user_feature), 0)
                    total_user_id = np.concatenate((total_user_id, user_id), 0)

            except tf.errors.OutOfRangeError:
                print('finish user embedding load')
                break

        print(total_user_id.shape)
        print(total_user_feature.shape)

        data_iter = get_features(user_dnn_test, args.batch_size, args.mode)
        data = data_iter.get_next()
        other_features, b_labels, b_uid = data
        model = Model(args, b_labels, other_features, total_user_id, b_uid)
        model.build_model(is_training=False)  # 预测不加batch_normalize
        model.predict()
        saver = tf.train.Saver()
        sess.run(tf.tables_initializer())
        sess.run(tf.group(tf.global_variables_initializer(),
                          tf.local_variables_initializer()))

        model_path = "%s/%s/youtubednn" % (MODEL_DIR, args.date)
        mode = args.mode
        saver.restore(sess, model_path)
        sess.run(data_iter.initializer)  # 重复多个epoch,重置数据迭代器
        step = 0
        total_acc = 0
        total_auc = 0
        start_time = time.time()
        step_num = args.step_num
        if mode == 'predict':
            while True:
                try:
                    prob, labels, acc = sess.run([model.prob, model.labels, model.accuracy], feed_dict={
                                                 model.user_feature: total_user_feature})
                    auc = roc_auc_score(y_true=labels, y_score=prob)
                    total_acc += acc
                    total_auc += auc
                    tf.logging.info("batch %f , %f" %
                                    (total_acc/(step+1), total_auc/(step+1)))
                    step += 1
                except tf.errors.OutOfRangeError:
                    break
        else:
            while True:
                try:
                    uid, output_value, output_index = sess.run(
                        [b_uid, top_value, top_index])
                    output_cid = [find_key_by_value(
                        id2CidDict, x) for x in output_index]
                    with open(predict_path, 'a', encoding='utf8') as f:
                        for i, cid_list in enumerate(output_cid):
                            uid_one = str(uid[i], encoding='utf8')
                            val_list = output_value[i]
                            output_str = "\2".join(
                                ["%s%s%.4f" % (k, "\3", v) for k, v in zip(cid_list, val_list)])
                            print_str = "%s%s%s" % (uid_one, "\1", output_str)
                            if step % step_num == 0:
                                tf.logging.info(print_str)
                            f.write("%s\n" % (print_str))
                            step += 1
                except tf.errors.OutOfRangeError:
                    print(step)
                    cost_time = time.time()-start_time
                    print("predict cost time: %d s" % (cost_time))
                    break


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(
        float_list=tf.train.FloatList(value=[value])
    )


def main(args):

    mode = args.mode
    if mode == 'train':
        train(args)
    else:
        predict(args)


if __name__ == '__main__':
    args = parse_args()
    main(args)
