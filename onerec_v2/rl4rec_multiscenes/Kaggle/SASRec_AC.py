import tensorflow as tf
import numpy as np
import pandas as pd
import os
import argparse
import trfl
from utility import *
from SASRecModules import *
from trfl import indexing_ops

def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")

    parser.add_argument('--epoch', type=int, default=40,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='data',
                        help='data directory')
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
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--discount', type=float, default=0.5,
                        help='Discount factor for RL.')
    parser.add_argument('--dropout_rate', default=0.1, type=float)
    return parser.parse_args()


class SASRecnetwork:
    def __init__(self, hidden_size,learning_rate,item_num,state_size, name='DQNetwork'):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.hidden_size=hidden_size
        self.item_num=int(item_num)
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.name=name
        with tf.variable_scope(self.name):

            self.all_embeddings=self.initialize_embeddings()

            self.inputs = tf.placeholder(tf.int32, [None, state_size],name='inputs')
            self.len_state=tf.placeholder(tf.int32, [None],name='len_state')
            self.target= tf.placeholder(tf.int32, [None],name='target') # target item, to calculate ce loss

            self.input_emb=tf.nn.embedding_lookup(self.all_embeddings['state_embeddings'],self.inputs)
            # Positional Encoding
            pos_emb=tf.nn.embedding_lookup(self.all_embeddings['pos_embeddings'],tf.tile(tf.expand_dims(tf.range(tf.shape(self.inputs)[1]), 0), [tf.shape(self.inputs)[0], 1]))
            self.seq=self.input_emb+pos_emb

            mask = tf.expand_dims(tf.to_float(tf.not_equal(self.inputs, item_num)), -1)
            # Dropout
            self.seq = tf.layers.dropout(self.seq,
                                          rate=args.dropout_rate,
                                          training=tf.convert_to_tensor(self.is_training))
            self.seq *= mask

            # Build blocks

            for i in range(args.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):
                    # Self-attention
                    self.seq = multihead_attention(queries=normalize(self.seq),
                                                   keys=self.seq,
                                                   num_units=self.hidden_size,
                                                   num_heads=args.num_heads,
                                                   dropout_rate=args.dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope="self_attention")

                    # Feed forward
                    self.seq = feedforward(normalize(self.seq), num_units=[self.hidden_size, self.hidden_size],
                                           dropout_rate=args.dropout_rate,
                                           is_training=self.is_training)
                    self.seq *= mask

            self.seq = normalize(self.seq)
            self.state_hidden=extract_axis_1(self.seq, self.len_state - 1)

            self.output1 = tf.contrib.layers.fully_connected(self.state_hidden, self.item_num,
                                                             activation_fn=None, scope="q-value")  # all q-values
            self.output2 = tf.contrib.layers.fully_connected(self.state_hidden, self.item_num,
                                                             activation_fn=None, scope="ce-logits")  # all logits

            # TRFL way
            self.actions = tf.placeholder(tf.int32, [None])
            self.targetQs_ = tf.placeholder(tf.float32, [None, item_num])
            self.targetQs_selector = tf.placeholder(tf.float32, [None,
                                                                 item_num])  # used for select best action for double q learning
            self.reward = tf.placeholder(tf.float32, [None])
            self.discount = tf.placeholder(tf.float32, [None])

            # TRFL double qlearning
            qloss, q_learning = trfl.double_qlearning(self.output1, self.actions, self.reward, self.discount,
                                                      self.targetQs_, self.targetQs_selector)
            q_indexed = tf.stop_gradient(indexing_ops.batched_index(self.output1, self.actions))

            celoss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.actions, logits=self.output2)

            celoss2 = tf.multiply(q_indexed, tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.actions,
                                                                                           logits=self.output2))
            self.loss1 = tf.reduce_mean(celoss1+qloss)
            self.loss2 =tf.reduce_mean(celoss2+qloss)
            self.opt1 = tf.train.AdamOptimizer(learning_rate).minimize(self.loss1)
            self.opt2 = tf.train.AdamOptimizer(learning_rate).minimize(self.loss2)


    def initialize_embeddings(self):
        all_embeddings = dict()
        state_embeddings= tf.Variable(tf.random_normal([self.item_num+1, self.hidden_size], 0.0, 0.01),
            name='state_embeddings')
        pos_embeddings=tf.Variable(tf.random_normal([self.state_size, self.hidden_size], 0.0, 0.01),
            name='pos_embeddings')
        all_embeddings['state_embeddings']=state_embeddings
        all_embeddings['pos_embeddings']=pos_embeddings
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
        prediction=sess.run(SASRec1.output2, feed_dict={SASRec1.inputs: states,SASRec1.len_state:len_states,SASRec1.is_training:False})
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

    SASRec1 = SASRecnetwork(hidden_size=args.hidden_factor, learning_rate=args.lr,item_num=item_num,state_size=state_size,name='SASRec1')
    SASRec2 = SASRecnetwork(hidden_size=args.hidden_factor, learning_rate=args.lr, item_num=item_num,
                            state_size=state_size, name='SASRec2')

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

                next_state = list(batch['next_state'].values())
                len_next_state = list(batch['len_next_states'].values())
                # double q learning, pointer is for selecting which network  is target and which is main
                pointer = np.random.randint(0, 2)
                if pointer == 0:
                    mainQN = SASRec1
                    target_QN = SASRec2
                else:
                    mainQN = SASRec2
                    target_QN = SASRec1
                target_Qs = sess.run(target_QN.output1,
                                     feed_dict={target_QN.inputs: next_state,
                                                target_QN.len_state: len_next_state,
                                                target_QN.is_training:True})
                target_Qs_selector = sess.run(mainQN.output1,
                                              feed_dict={mainQN.inputs: next_state,
                                                         mainQN.len_state: len_next_state,
                                                         mainQN.is_training:True})

                # Set target_Qs to 0 for states where episode ends
                is_done = list(batch['is_done'].values())
                for index in range(target_Qs.shape[0]):
                    if is_done[index]:
                        target_Qs[index] = np.zeros([item_num])

                state = list(batch['state'].values())
                len_state = list(batch['len_state'].values())
                action = list(batch['action'].values())
                is_buy = list(batch['is_buy'].values())
                reward = []
                for k in range(len(is_buy)):
                    reward.append(reward_buy if is_buy[k] == 1 else reward_click)
                discount = [args.discount] * len(action)

                if total_step<30000:

                    loss, _ = sess.run([mainQN.loss1, mainQN.opt1],
                                       feed_dict={mainQN.inputs: state,
                                                  mainQN.len_state: len_state,
                                                  mainQN.targetQs_: target_Qs,
                                                  mainQN.reward: reward,
                                                  mainQN.discount: discount,
                                                  mainQN.actions: action,
                                                  mainQN.targetQs_selector: target_Qs_selector,
                                                  mainQN.is_training:True})
                    total_step += 1
                    if total_step % 200 == 0:
                        print("the loss in %dth batch is: %f" % (total_step, loss))
                    if total_step % 2000== 0:
                        evaluate(sess)
                else:
                    loss, _ = sess.run([mainQN.loss2, mainQN.opt2],
                                       feed_dict={mainQN.inputs: state,
                                                  mainQN.len_state: len_state,
                                                  mainQN.targetQs_: target_Qs,
                                                  mainQN.reward: reward,
                                                  mainQN.discount: discount,
                                                  mainQN.actions: action,
                                                  mainQN.targetQs_selector: target_Qs_selector,
                                                  mainQN.is_training: True})
                    total_step += 1
                    if total_step % 200 == 0:
                        print("the loss in %dth batch is: %f" % (total_step, loss))
                    if total_step % 2000 == 0:
                        evaluate(sess)






