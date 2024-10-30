import tensorflow as tf
import numpy as np
import pandas as pd
import os
import argparse
import trfl
from utility import *
from NextItNetModules import *

def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")

    parser.add_argument('--epoch', type=int, default=30,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='data',
                        help='data directory')
    # parser.add_argument('--pretrain', type=int, default=1,
    #                     help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save the model to pretrain file')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--r_click', type=float, default=0.2,
                        help='reward for the click behavior.')
    parser.add_argument('--r_buy', type=float, default=1.0,
                        help='reward for the purchase behavior.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')

    return parser.parse_args()


class NextItNet:
    def __init__(self, hidden_size,learning_rate,item_num,state_size):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.hidden_size=hidden_size
        self.item_num=int(item_num)
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.all_embeddings=self.initialize_embeddings()

        self.inputs = tf.placeholder(tf.int32, [None, state_size],name='inputs')
        self.len_state=tf.placeholder(tf.int32, [None],name='len_state')
        self.target= tf.placeholder(tf.int32, [None],name='target') # target item, to calculate ce loss
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.inputs, item_num)), -1)

        # self.input_emb=tf.nn.embedding_lookup(all_embeddings['state_embeddings'],self.inputs)
        self.model_para = {
            'dilated_channels': 64,  # larger is better until 512 or 1024
            'dilations': [1, 2, 1, 2, 1, 2, ],  # YOU should tune this hyper-parameter, refer to the paper.
            'kernel_size': 3,
        }

        context_embedding = tf.nn.embedding_lookup(self.all_embeddings['state_embeddings'],
                                                   self.inputs)
        context_embedding *= mask

        dilate_output = context_embedding
        for layer_id, dilation in enumerate(self.model_para['dilations']):
            dilate_output = nextitnet_residual_block(dilate_output, dilation,
                                                    layer_id, self.model_para['dilated_channels'],
                                                    self.model_para['kernel_size'], causal=True, train=self.is_training)
            dilate_output *= mask

        self.state_hidden = extract_axis_1(dilate_output, self.len_state - 1)

        self.output = tf.contrib.layers.fully_connected(self.state_hidden,self.item_num,activation_fn=None,scope='fc')

        self.loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target,logits=self.output)
        self.loss = tf.reduce_mean(self.loss)
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)



    def initialize_embeddings(self):
        all_embeddings = dict()
        state_embeddings= tf.Variable(tf.random_normal([self.item_num+1, self.hidden_size], 0.0, 0.01),
            name='state_embeddings')
        all_embeddings['state_embeddings']=state_embeddings
        return all_embeddings

def evaluate(sess):
    eval_sessions=pd.read_pickle(os.path.join(data_directory, 'sampled_val.df'))
    eval_ids = eval_sessions.session_id.unique()
    groups = eval_sessions.groupby('session_id')
    batch = 100
    evaluated=0
    total_clicks=0.0
    total_purchase = 0.0
    total_reward = [0, 0, 0, 0]
    hit_clicks=[0,0,0,0]
    ndcg_clicks=[0,0,0,0]
    hit_purchase=[0,0,0,0]
    ndcg_purchase=[0,0,0,0]
    while evaluated<len(eval_ids):
        states, len_states, actions, rewards = [], [], [], []
        for i in range(batch):
            if evaluated==len(eval_ids):
                break
            id=eval_ids[evaluated]
            group=groups.get_group(id)
            history=[]
            for index, row in group.iterrows():
                state=list(history)
                len_states.append(state_size if len(state)>=state_size else 1 if len(state)==0 else len(state))
                state=pad_history(state,state_size,item_num)
                states.append(state)
                action=row['item_id']
                is_buy=row['is_buy']
                reward = reward_buy if is_buy == 1 else reward_click
                if is_buy==1:
                    total_purchase+=1.0
                else:
                    total_clicks+=1.0
                actions.append(action)
                rewards.append(reward)
                history.append(row['item_id'])
            evaluated+=1
        prediction=sess.run(NextRec.output, feed_dict={NextRec.inputs: states,NextRec.len_state:len_states,NextRec.is_training:False})
        sorted_list=np.argsort(prediction)
        calculate_hit(sorted_list,topk,actions,rewards,reward_click,total_reward,hit_clicks,ndcg_clicks,hit_purchase,ndcg_purchase)
    print('#############################################################')
    print('total clicks: %d, total purchase:%d' % (total_clicks, total_purchase))
    for i in range(len(topk)):
        hr_click=hit_clicks[i]/total_clicks
        hr_purchase=hit_purchase[i]/total_purchase
        ng_click=ndcg_clicks[i]/total_clicks
        ng_purchase=ndcg_purchase[i]/total_purchase
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('cumulative reward @ %d: %f' % (topk[i],total_reward[i]))
        print('clicks hr ndcg @ %d : %f, %f' % (topk[i],hr_click,ng_click))
        print('purchase hr and ndcg @%d : %f, %f' % (topk[i], hr_purchase, ng_purchase))
    print('#############################################################')



if __name__ == '__main__':
    # Network parameters
    args = parse_args()

    data_directory = args.data
    data_statis = pd.read_pickle(
        os.path.join(data_directory, 'data_statis.df'))  # read data statistics, includeing state_size and item_num
    state_size = data_statis['state_size'][0]  # the length of history to define the state
    item_num = data_statis['item_num'][0]  # total number of items
    reward_click = args.r_click
    reward_buy = args.r_buy
    topk=[5,10,15,20]
    # save_file = 'pretrain-GRU/%d' % (hidden_size)

    tf.reset_default_graph()

    NextRec = NextItNet(hidden_size=args.hidden_factor, learning_rate=args.lr,item_num=item_num,state_size=state_size)

    replay_buffer = pd.read_pickle(os.path.join(data_directory, 'replay_buffer.df'))
    saver = tf.train.Saver()

    total_step=0
    with tf.Session() as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        # evaluate(sess)
        num_rows=replay_buffer.shape[0]
        num_batches=int(num_rows/args.batch_size)
        for i in range(args.epoch):
            for j in range(num_batches):
                batch = replay_buffer.sample(n=args.batch_size).to_dict()
                state = list(batch['state'].values())
                len_state = list(batch['len_state'].values())
                target=list(batch['action'].values())
                loss, _ = sess.run([NextRec.loss, NextRec.opt],
                                   feed_dict={NextRec.inputs: state,
                                              NextRec.len_state: len_state,
                                              NextRec.target: target,
                                              NextRec.is_training:True})
                total_step+=1
                if total_step % 200 == 0:
                    print("the loss in %dth batch is: %f" % (total_step, loss))
                if total_step % 2000 == 0:
                    evaluate(sess)
        # saver.save(sess, save_file)









