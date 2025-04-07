import tensorflow as tf
import numpy as np
import pandas as pd
import os
import argparse
import trfl
import pdb
from tqdm import tqdm
import ast
from utility import *
from NextItNetModules import *

def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")

    parser.add_argument('--epoch', type=int, default=30,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='./data',
                        help='data directory')
    # parser.add_argument('--pretrain', type=int, default=1,
    #                     help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save the model to pretrain file')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    
    # parser.add_argument('--r_click', type=float, default=0.2,help='reward for the click behavior.')
    # parser.add_argument('--r_buy', type=float, default=1.0,help='reward for the purchase behavior.')
    parser.add_argument('--r_view', type=float, default=0.2,help='reward for the click behavior.')
    parser.add_argument('--r_click', type=float, default=1.0,help='reward for the purchase behavior.')

    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--discount', type=float, default=0.5,
                        help='Discount factor for RL.')

    return parser.parse_args()


class NextItNet:
    def __init__(self, hidden_size,learning_rate,item_num,category_num,state_size,name='NextRec'):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.hidden_size=hidden_size
        self.item_num=int(item_num)
        self.category_num = int(category_num)
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.name = name
        with tf.variable_scope(self.name):

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

            context_embedding = tf.nn.embedding_lookup(self.all_embeddings['state_embeddings'], self.inputs)
            context_embedding *= mask

            dilate_output = context_embedding
            for layer_id, dilation in enumerate(self.model_para['dilations']):
                dilate_output = nextitnet_residual_block(dilate_output, dilation,
                                                        layer_id, self.model_para['dilated_channels'],
                                                        self.model_para['kernel_size'], causal=True, train=self.is_training)
                dilate_output *= mask

            self.state_hidden = extract_axis_1(dilate_output, self.len_state - 1)

            self.output1 = tf.contrib.layers.fully_connected(self.state_hidden, self.category_num, activation_fn=None, scope="q-value")  # all q-values
            self.output2 = tf.contrib.layers.fully_connected(self.state_hidden, self.item_num, activation_fn=None, scope="ce-logits")  # all ce logits

            # TRFL way
            self.actions = tf.placeholder(tf.int32, [None])
            self.true_items = tf.placeholder(tf.int32, [None])
            self.targetQs_ = tf.placeholder(tf.float32, [None, self.category_num])
            self.targetQs_selector = tf.placeholder(tf.float32, [None, self.category_num])  # used for select best action for double q learning
            self.reward = tf.placeholder(tf.float32, [None])
            self.discount = tf.placeholder(tf.float32, [None])

            # TRFL double qlearning
            qloss, q_learning = trfl.double_qlearning(self.output1, self.actions, self.reward, self.discount,self.targetQs_, self.targetQs_selector)

            celoss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.true_items, logits=self.output2)
            
            self.loss = tf.reduce_mean(qloss + celoss)
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)



    def initialize_embeddings(self):
        all_embeddings = dict()
        state_embeddings= tf.Variable(tf.random_normal([self.item_num+1, self.hidden_size], 0.0, 0.01), name='state_embeddings')
        all_embeddings['state_embeddings']=state_embeddings
        return all_embeddings

def evaluate(sess):
    # eval_sessions=pd.read_pickle(os.path.join(data_directory, 'sampled_val.df'))
    eval_sessions=pd.read_json(os.path.join(data_directory, 'sampled_val.json'))
    eval_ids = eval_sessions.session_id.unique()
    groups = eval_sessions.groupby('session_id')
    batch = 200
    # print("len(eval_ids):",len(eval_ids), " eval nums:", (len(eval_ids) // batch) * batch)
    evaluated=0
    total_clicks=0.0
    total_views=0.0
    # total_purchase = 0.0
    total_reward = [0, 0, 0, 0]
    hit_clicks=[0,0,0,0]
    ndcg_clicks=[0,0,0,0]
    
    # hit_purchase=[0,0,0,0]
    # ndcg_purchase=[0,0,0,0]
    
    # pbar = tqdm(total=len(eval_ids), desc="Evaluate-ing")
    # while evaluated< len(eval_ids):
    while evaluated< len(eval_ids):
        # print("evaluated:",evaluated,"len(eval_ids):",len(eval_ids))
        states, len_states, actions, true_items, rewards = [], [], [], [], []
        for i in range(batch):
            id=eval_ids[evaluated]
            group=groups.get_group(id)
            history=[]
            for index, row in group.iterrows():
                state=list(history)
                len_states.append(state_size if len(state)>=state_size else 1 if len(state)==0 else len(state))
                state=pad_history(state,state_size,item_num)
                states.append(state)
                # action=row['item_id']
                true_item = row['adgroup_id']
                is_click=row['clk']
                reward = reward_click if is_click == 1 else reward_view

                if is_click==1:
                    total_clicks+=1.0
                else:
                    total_views+=1.0

                true_items.append(true_item)
                actions.append(action)
                rewards.append(reward)
                history.append(row['adgroup_id'])
            evaluated+=1
            # pbar.update(1)
        prediction=sess.run(NextRec1.output2, feed_dict={NextRec1.inputs: states,NextRec1.len_state:len_states,NextRec1.is_training:False})
        sorted_list=np.argsort(prediction)

        # calculate_hit(sorted_list,topk,actions,rewards,reward_click,total_reward,hit_clicks,ndcg_clicks,hit_purchase,ndcg_purchase)

        calculate_hit_only_click(sorted_list,topk,true_items,rewards,reward_click,total_reward,hit_clicks,ndcg_clicks)

    print('#############################################################')
    # print('total clicks: %d, total purchase:%d' % (total_clicks, total_purchase))
    print('total clicks: %d ' % (total_clicks))
    for i in range(len(topk)):
        hr_click=hit_clicks[i]/total_clicks
        # hr_purchase=hit_purchase[i]/total_purchase
        ng_click=ndcg_clicks[i]/total_clicks
        # ng_purchase=ndcg_purchase[i]/total_purchase
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('cumulative reward @ %d: %f' % (topk[i],total_reward[i]))
        print('clicks hr ndcg @ %d : %f, %f' % (topk[i],hr_click,ng_click))
        # print('purchase hr and ndcg @%d : %f, %f' % (topk[i], hr_purchase, ng_purchase))
    print('#############################################################')



if __name__ == '__main__':
    # Network parameters
    args = parse_args()

    data_directory = args.data
    # data_statis = pd.read_pickle(os.path.join(data_directory, 'data_statis.df'))  # read data statistics, includeing state_size and item_num
    data_statis = pd.read_json(os.path.join(data_directory, 'data_statis.json'))  # read data statistics, includeing state_size and item_num
    state_size = data_statis['state_size'][0]  # the length of history to define the state
    item_num = data_statis['item_num'][0]  # total number of items
    category_num = data_statis['category_num'][0]

    reward_click = args.r_click
    reward_view = args.r_view

    topk=[5,10,15,20]
    # save_file = 'pretrain-GRU/%d' % (hidden_size)

    tf.reset_default_graph()

    NextRec1 = NextItNet(hidden_size=args.hidden_factor, learning_rate=args.lr,item_num=item_num,category_num=category_num,state_size=state_size,name='NextRec1')
    NextRec2 = NextItNet(hidden_size=args.hidden_factor, learning_rate=args.lr, item_num=item_num,category_num=category_num,state_size=state_size,name='NextRec2')

    # replay_buffer = pd.read_pickle(os.path.join(data_directory, 'replay_buffer.df'))
    replay_buffer = pd.read_json(os.path.join(data_directory, 'replay_buffer.json'))
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
                
                next_state = list(batch['next_state'].values())
                # next_state = [ast.literal_eval(item) for item in t_next_state]
                len_next_state = list(batch['len_next_states'].values())
                # double q learning, pointer is for selecting which network  is target and which is main
                pointer = np.random.randint(0, 2)
                if pointer == 0:
                    mainQN = NextRec1
                    target_QN = NextRec2
                else:
                    mainQN = NextRec2
                    target_QN = NextRec1
                target_Qs = sess.run(target_QN.output1,
                                     feed_dict={target_QN.inputs: next_state,
                                                target_QN.len_state: len_next_state,
                                                target_QN.is_training: True})
                target_Qs_selector = sess.run(mainQN.output1,
                                              feed_dict={mainQN.inputs: next_state,
                                                         mainQN.len_state: len_next_state,
                                                         mainQN.is_training: True})

                # Set target_Qs to 0 for states where episode ends
                is_done = list(batch['is_done'].values())
                for index in range(target_Qs.shape[0]):
                    if is_done[index]:
                        target_Qs[index] = np.zeros([category_num])

                state = list(batch['state'].values())
                len_state = list(batch['len_state'].values())
                action = list(batch['action'].values())
                is_click = list(batch['is_click'].values())
                true_item = list(batch['true_item'].values())

                reward = []
                for k in range(len(is_click)):
                    reward.append(reward_click if is_click[k] == 1 else reward_view)
                discount = [args.discount] * len(action)

                loss, _ = sess.run([mainQN.loss, mainQN.opt],
                                   feed_dict={mainQN.inputs: state,
                                              mainQN.len_state: len_state,
                                              mainQN.targetQs_: target_Qs,
                                              mainQN.reward: reward,
                                              mainQN.discount: discount,
                                              mainQN.actions: action,
                                              mainQN.true_items: true_item,
                                              mainQN.targetQs_selector: target_Qs_selector,
                                              mainQN.is_training: True})
                total_step += 1
                if total_step % 200 == 0:
                    print("the loss in %dth batch is: %f" % (total_step, loss))
                if total_step % 2000 == 0:
                    evaluate(sess)









