import tensorflow as tf
import numpy as np
import pandas as pd
import os
import argparse
from collections import deque
from utility import pad_history,calculate_hit

import trfl
from trfl import indexing_ops

def parse_args():
    parser = argparse.ArgumentParser(description="Run nive double q learning.")

    parser.add_argument('--epoch', type=int, default=300,
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
    parser.add_argument('--r_negative', type=float, default=-1.0,
                        help='reward for the negative behavior.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--discount', type=float, default=0.5,
                        help='Discount factor for RL.')
    return parser.parse_args()


class QNetwork:
    def __init__(self, hidden_size, learning_rate, item_num, state_size, pretrain, name='DQNetwork'):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.pretrain = pretrain
        # self.save_file = save_file
        self.name = name
        with tf.variable_scope(self.name):
            self.all_embeddings=self.initialize_embeddings()
            self.inputs = tf.placeholder(tf.int32, [None, state_size])  # sequence of history, [batchsize,state_size]
            self.len_state = tf.placeholder(tf.int32, [
                None])  # the length of valid positions, because short sesssions need to be padded

            # one_hot_input = tf.one_hot(self.inputs, self.item_num+1)
            self.input_emb = tf.nn.embedding_lookup(self.all_embeddings['state_embeddings'], self.inputs)

            gru_out, self.states_hidden = tf.nn.dynamic_rnn(
                tf.contrib.rnn.GRUCell(self.hidden_size),
                self.input_emb,
                dtype=tf.float32,
                sequence_length=self.len_state,
            )

            self.output = tf.contrib.layers.fully_connected(self.states_hidden, self.item_num,
                                                            activation_fn=None)  # all q-values

            # TRFL way
            self.actions = tf.placeholder(tf.int32, [None])

            self.negative0=tf.placeholder(tf.int32,[None])
            self.negative1 = tf.placeholder(tf.int32, [None])
            self.negative2= tf.placeholder(tf.int32, [None])
            self.negative3= tf.placeholder(tf.int32, [None])
            self.negative4= tf.placeholder(tf.int32, [None])
            self.negative5= tf.placeholder(tf.int32, [None])
            self.negative6= tf.placeholder(tf.int32, [None])
            self.negative7= tf.placeholder(tf.int32, [None])
            self.negative8= tf.placeholder(tf.int32, [None])
            self.negative9= tf.placeholder(tf.int32, [None])

            self.targetQs_ = tf.placeholder(tf.float32, [None, item_num])
            self.targetQs_selector = tf.placeholder(tf.float32, [None,
                                                                 item_num])  # used for select best action for double q learning
            self.reward = tf.placeholder(tf.float32, [None])
            self.discount = tf.placeholder(tf.float32, [None])

            self.targetQ_current_ = tf.placeholder(tf.float32, [None, item_num])
            self.targetQ_current_selector = tf.placeholder(tf.float32, [None,
                                                                 item_num])  # used for select best action for double q learning


            # TRFL double qlearning
            qloss_positive, q_learning_positive = trfl.double_qlearning(self.output, self.actions, self.reward, self.discount,self.targetQs_, self.targetQs_selector)

            neg_reward=tf.constant(reward_negative,dtype=tf.float32, shape=(256,))

            qloss_negative0,q_learning_negative0 = trfl.double_qlearning(self.output,self.negative0,neg_reward,self.discount,self.targetQ_current_,self.targetQ_current_selector)

            qloss_negative1, q_learning_negative1 = trfl.double_qlearning(self.output, self.negative1, neg_reward,self.discount, self.targetQ_current_,self.targetQ_current_selector)

            qloss_negative2, q_learning_negative2 = trfl.double_qlearning(self.output, self.negative2, neg_reward,self.discount, self.targetQ_current_,self.targetQ_current_selector)

            qloss_negative3, q_learning_negative3 = trfl.double_qlearning(self.output, self.negative3, neg_reward,self.discount, self.targetQ_current_,self.targetQ_current_selector)

            qloss_negative4, q_learning_negative4 = trfl.double_qlearning(self.output, self.negative4, neg_reward,self.discount, self.targetQ_current_,self.targetQ_current_selector)

            qloss_negative5, q_learning_negative5 = trfl.double_qlearning(self.output, self.negative5, neg_reward,self.discount, self.targetQ_current_,self.targetQ_current_selector)

            qloss_negative6, q_learning_negative6 = trfl.double_qlearning(self.output, self.negative6, neg_reward,self.discount, self.targetQ_current_,self.targetQ_current_selector)

            qloss_negative7, q_learning_negative7 = trfl.double_qlearning(self.output, self.negative7, neg_reward,self.discount, self.targetQ_current_,self.targetQ_current_selector)

            qloss_negative8, q_learning_negative8 = trfl.double_qlearning(self.output, self.negative8, neg_reward,self.discount, self.targetQ_current_,self.targetQ_current_selector)

            qloss_negative9, q_learning_negative9 = trfl.double_qlearning(self.output, self.negative9, neg_reward,self.discount, self.targetQ_current_,self.targetQ_current_selector)


            self.loss = tf.reduce_mean(qloss_positive + qloss_negative0 + qloss_negative1 + qloss_negative2 + qloss_negative3 + qloss_negative4 + qloss_negative5 + qloss_negative6 + qloss_negative7 + qloss_negative8 + qloss_negative9)
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def initialize_embeddings(self):
        all_embeddings = dict()
        if self.pretrain == False:
            with tf.variable_scope(self.name):
                state_embeddings = tf.Variable(tf.random_normal([self.item_num + 1, self.hidden_size], 0.0, 0.01),
                                           name='state_embeddings')
                all_embeddings['state_embeddings'] = state_embeddings
        # else:
        #     weight_saver = tf.train.import_meta_graph(self.save_file + '.meta')
        #     pretrain_graph = tf.get_default_graph()
        #     state_embeddings = pretrain_graph.get_tensor_by_name('state_embeddings:0')
        #     with tf.Session() as sess:
        #         weight_saver.restore(sess, self.save_file)
        #         se = sess.run([state_embeddings])[0]
        #     with tf.variable_scope(self.name):
        #         all_embeddings['state_embeddings'] = tf.Variable(se, dtype=tf.float32)
        #     print("load!")
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
        prediction=sess.run(QN_1.output, feed_dict={QN_1.inputs: states,QN_1.len_state:len_states})
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
    reward_negative=args.r_negative
    topk=[5,10,15,20]
    # save_file = 'pretrain-GRU/%d' % (hidden_size)

    tf.reset_default_graph()

    QN_1 = QNetwork(name='QN_1', hidden_size=args.hidden_factor, learning_rate=args.lr, item_num=item_num,
                    state_size=state_size, pretrain=False)
    QN_2 = QNetwork(name='QN_2', hidden_size=args.hidden_factor, learning_rate=args.lr, item_num=item_num,
                    state_size=state_size, pretrain=False)

    replay_buffer = pd.read_pickle(os.path.join(data_directory, 'replay_buffer.df'))
    # saver = tf.train.Saver()

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

                #state = list(batch['state'].values())

                next_state = list(batch['next_state'].values())
                len_next_state = list(batch['len_next_states'].values())
                # double q learning, pointer is for selecting which network  is target and which is main
                pointer = np.random.randint(0, 2)
                if pointer == 0:
                    mainQN = QN_1
                    target_QN = QN_2
                else:
                    mainQN = QN_2
                    target_QN = QN_1
                target_Qs = sess.run(target_QN.output,
                                     feed_dict={target_QN.inputs: next_state,
                                                target_QN.len_state: len_next_state})
                target_Qs_selector = sess.run(mainQN.output,
                                              feed_dict={mainQN.inputs: next_state,
                                                         mainQN.len_state: len_next_state})
                # Set target_Qs to 0 for states where episode ends
                is_done = list(batch['is_done'].values())
                for index in range(target_Qs.shape[0]):
                    if is_done[index]:
                        target_Qs[index] = np.zeros([item_num])

                state = list(batch['state'].values())
                len_state = list(batch['len_state'].values())
                target_Q_current = sess.run(target_QN.output,
                                            feed_dict={target_QN.inputs: state,
                                                       target_QN.len_state: len_state})
                target_Q__current_selector = sess.run(mainQN.output,
                                                      feed_dict={mainQN.inputs: state,
                                                                 mainQN.len_state: len_state})
                action = list(batch['action'].values())
                negative_list1=[]
                negative_list2 = []
                negative_list3 = []
                negative_list4 = []
                negative_list5 = []
                negative_list6 = []
                negative_list7 = []
                negative_list8 = []
                negative_list9 = []
                negative_list0 = []

                for index in range(target_Qs.shape[0]):
                    neg=np.random.randint(item_num)
                    while neg==action[index]:
                        neg = np.random.randint(item_num)
                    negative_list0.append(neg)
                for index in range(target_Qs.shape[0]):
                    neg=np.random.randint(item_num)
                    while neg==action[index]:
                        neg = np.random.randint(item_num)
                    negative_list1.append(neg)
                for index in range(target_Qs.shape[0]):
                    neg=np.random.randint(item_num)
                    while neg==action[index]:
                        neg = np.random.randint(item_num)
                    negative_list2.append(neg)
                for index in range(target_Qs.shape[0]):
                    neg=np.random.randint(item_num)
                    while neg==action[index]:
                        neg = np.random.randint(item_num)
                    negative_list3.append(neg)
                for index in range(target_Qs.shape[0]):
                    neg=np.random.randint(item_num)
                    while neg==action[index]:
                        neg = np.random.randint(item_num)
                    negative_list4.append(neg)
                for index in range(target_Qs.shape[0]):
                    neg=np.random.randint(item_num)
                    while neg==action[index]:
                        neg = np.random.randint(item_num)
                    negative_list5.append(neg)
                for index in range(target_Qs.shape[0]):
                    neg=np.random.randint(item_num)
                    while neg==action[index]:
                        neg = np.random.randint(item_num)
                    negative_list6.append(neg)
                for index in range(target_Qs.shape[0]):
                    neg=np.random.randint(item_num)
                    while neg==action[index]:
                        neg = np.random.randint(item_num)
                    negative_list7.append(neg)
                for index in range(target_Qs.shape[0]):
                    neg=np.random.randint(item_num)
                    while neg==action[index]:
                        neg = np.random.randint(item_num)
                    negative_list8.append(neg)
                for index in range(target_Qs.shape[0]):
                    neg=np.random.randint(item_num)
                    while neg==action[index]:
                        neg = np.random.randint(item_num)
                    negative_list9.append(neg)

                is_buy=list(batch['is_buy'].values())
                reward=[]
                for k in range(len(is_buy)):
                    reward.append(reward_buy if is_buy[k] == 1 else reward_click)
                discount = [args.discount] * len(action)

                loss, _ = sess.run([mainQN.loss, mainQN.opt],
                                   feed_dict={mainQN.inputs: state,
                                              mainQN.len_state: len_state,
                                              mainQN.targetQs_: target_Qs,
                                              mainQN.reward: reward,
                                              mainQN.discount: discount,
                                              mainQN.actions: action,
                                              mainQN.targetQs_selector: target_Qs_selector,
                                              mainQN.negative0:negative_list0,
                                              mainQN.negative1: negative_list1,
                                              mainQN.negative2: negative_list2,
                                              mainQN.negative3: negative_list3,
                                              mainQN.negative4: negative_list4,
                                              mainQN.negative5: negative_list5,
                                              mainQN.negative6: negative_list6,
                                              mainQN.negative7: negative_list7,
                                              mainQN.negative8: negative_list8,
                                              mainQN.negative9: negative_list9,
                                              mainQN.targetQ_current_:target_Q_current,
                                              mainQN.targetQ_current_selector:target_Q__current_selector
                                              })
                total_step += 1
                if total_step % 200 == 0:
                    print("the loss in %dth batch is: %f" % (total_step, loss))
                if total_step % 10000 == 0:
                    evaluate(sess)



