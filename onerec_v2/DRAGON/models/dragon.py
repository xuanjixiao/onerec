# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
FREEDOM: A Tale of Two Graphs: Freezing and Denoising Graph Structures for Multimodal Recommendation
# Update: 01/08/2022
"""


import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss
from utils.utils import build_sim, compute_normalized_laplacian


class DRAGON(GeneralRecommender):
    def __init__(self, config, dataset):
        super(DRAGON, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.knn_k = config['knn_k']
        self.lambda_coeff = config['lambda_coeff']
        self.cf_model = config['cf_model']
        self.n_layers = config['n_mm_layers']
        self.n_ui_layers = config['n_ui_layers']
        self.reg_weight = config['reg_weight']
        self.build_item_graph = True
        self.mm_image_weight = config['mm_image_weight']
        self.dropout = config['dropout']
        self.degree_ratio = config['degree_ratio']
        dataset_path = config['dataset_path']
        # dragon特有参数
        self.aggr_mode = config['aggr_mode']
        self.user_aggr_mode = 'softmax'
        self.num_blocks = config['res_block_num']  # 残差块的数量
        self.v_weight = config['v_weight']
        self.dragon_weight = config['dragon_weight']
        self.t_weight = 1 - self.v_weight
        self.k = 40

        self.n_nodes = self.n_users + self.n_items

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        self.user_graph_dict = np.load(os.path.join(dataset_path, config['user_graph_dict_file']),
                                       allow_pickle=True).item()
       
        
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.masked_adj, self.mm_adj = None, None
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices, self.edge_values = self.edge_indices.to(self.device), self.edge_values.to(self.device)
        # self.edge_full_indices = torch.arange(self.edge_values.size(0)).to(self.device)

        edge_index = self.pack_edge_index(self.interaction_matrix)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)
    
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        self.user_t_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.user_v_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        
        self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                np.random.randn(self.n_users, self.feat_embed_dim), dtype=torch.float32, requires_grad=True),
                gain=1).to(self.device))

        # sota v1
        self.weight_u = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.n_users, 2, 1), dtype=torch.float32, requires_grad=True)))
        self.weight_u.data = F.softmax(self.weight_u, dim=1)
        self.weight_i = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.n_items, 2, 1), dtype=torch.float32, requires_grad=True)))
        self.weight_i.data = F.softmax(self.weight_i, dim=1)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        mm_adj_file = os.path.join(dataset_path, 'mm_adj_freedomdsp_{}_{}.pt'.format(self.knn_k, int(10*self.mm_image_weight)))

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file)
        else:
            if self.v_feat is not None:
                indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                self.mm_adj = image_adj
            if self.t_feat is not None:
                indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                del text_adj
                del image_adj
            torch.save(self.mm_adj, mm_adj_file)

        self.user_mlp_layer = nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim)
        if self.v_feat is not None:            
            self.v_gcn = GCN(self.preference, self.n_users, self.n_items, self.aggr_mode, feat_embed_dim=self.feat_embed_dim,
                             device=self.device, use_mlp=False)  # 256)
        if self.t_feat is not None:
            self.t_gcn = GCN(self.preference, self.n_users, self.n_items, self.aggr_mode, feat_embed_dim=self.feat_embed_dim,
                             device=self.device, use_mlp=False)
        
        # sota_v1
        self.user_graph = UserGraphSample(self.n_users, 'add', self.feat_embed_dim)

        # 创建多层映射网络 - 为同质和多样性信息分别定义残差块
        self.v_homo_res_blocks = nn.ModuleList()  # 视觉模态同质信息的残差块
        self.v_diver_res_blocks = nn.ModuleList()  # 视觉模态多样性信息的残差块
        self.t_homo_res_blocks = nn.ModuleList()  # 文本模态同质信息的残差块
        self.t_diver_res_blocks = nn.ModuleList()  # 文本模态多样性信息的残差块

        # sota_v1
        self.ffn = FFN(hidden_size=self.feat_embed_dim, inner_hidden_size=self.feat_embed_dim*4, bias=True)
        self.v_mlp = FFN(hidden_size=self.feat_embed_dim, inner_hidden_size=self.feat_embed_dim*4, bias=True)
        self.t_mlp = FFN(hidden_size=self.feat_embed_dim, inner_hidden_size=self.feat_embed_dim*4, bias=True)

        for i in range(self.num_blocks):
            # 视觉模态的同质信息残差块
            self.v_homo_res_blocks.append(
                FFN(hidden_size=self.feat_embed_dim, inner_hidden_size=self.feat_embed_dim*4, bias=True, activation=nn.ReLU)
            )
            # 视觉模态的多样性信息残差块
            self.v_diver_res_blocks.append(
                FFN(hidden_size=self.feat_embed_dim, inner_hidden_size=self.feat_embed_dim*4, bias=True, activation=nn.ReLU)
            )
            # 文本模态的同质信息残差块
            self.t_homo_res_blocks.append(
                FFN(hidden_size=self.feat_embed_dim, inner_hidden_size=self.feat_embed_dim*4, bias=True, activation=nn.ReLU)
            )
            # 文本模态的多样性信息残差块
            self.t_diver_res_blocks.append(
                FFN(hidden_size=self.feat_embed_dim, inner_hidden_size=self.feat_embed_dim*4, bias=True, activation=nn.ReLU)
            )
        # 添加初始的同质/多样性分离层
        self.v_homo_map = FFN(hidden_size=self.feat_embed_dim, inner_hidden_size=self.feat_embed_dim*4, 
                            bias=True, activation=nn.ReLU)
        self.v_diver_map = FFN(hidden_size=self.feat_embed_dim, inner_hidden_size=self.feat_embed_dim*4, 
                            bias=True, activation=nn.ReLU)

        self.t_homo_map = FFN(hidden_size=self.feat_embed_dim, inner_hidden_size=self.feat_embed_dim*4, 
                            bias=True, activation=nn.ReLU)
        self.t_diver_map = FFN(hidden_size=self.feat_embed_dim, inner_hidden_size=self.feat_embed_dim*4, 
                            bias=True, activation=nn.ReLU)

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def pre_epoch_processing(self):
        self.epoch_user_graph, self.user_weight_matrix = self.topk_sample(self.k)
        self.user_weight_matrix = self.user_weight_matrix.to(self.device)
    
        if self.dropout <= .0:
            self.masked_adj = self.norm_adj
            return
        # degree-sensitive edge pruning
        degree_len = int(self.edge_values.size(0) * (1. - self.dropout))
        degree_idx = torch.multinomial(self.edge_values, degree_len)
        # random sample
        keep_indices = self.edge_indices[:, degree_idx]
        # norm values
        keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.n_users, self.n_items)))
        all_values = torch.cat((keep_values, keep_values))
        # update keep_indices to users/items+self.n_users
        keep_indices[1] += self.n_users
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
        self.masked_adj = torch.sparse.FloatTensor(all_indices, all_values, self.norm_adj.shape).to(self.device)

    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def pre_epoch_processing_dragon_v1(self):
        # sota_v1
        self.epoch_user_graph, self.user_weight_matrix = self.topk_sample(self.k)
        self.user_weight_matrix = self.user_weight_matrix.to(self.device)

    def topk_sample(self, k):
        # sota_v1
        user_graph_index = []
        count_num = 0
        user_weight_matrix = torch.zeros(len(self.user_graph_dict), k)
        tasike = []
        for i in range(k):
            tasike.append(0)
        for i in range(len(self.user_graph_dict)):
            if len(self.user_graph_dict[i][0]) < k:
                count_num += 1
                if len(self.user_graph_dict[i][0]) == 0:
                    # pdb.set_trace()
                    user_graph_index.append(tasike)
                    continue
                user_graph_sample = self.user_graph_dict[i][0][:k]
                user_graph_weight = self.user_graph_dict[i][1][:k]
                while len(user_graph_sample) < k:
                    rand_index = np.random.randint(0, len(user_graph_sample))
                    user_graph_sample.append(user_graph_sample[rand_index])
                    user_graph_weight.append(user_graph_weight[rand_index])
                user_graph_index.append(user_graph_sample)

                if self.user_aggr_mode == 'softmax':
                    user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)  # softmax
                if self.user_aggr_mode == 'mean':
                    user_weight_matrix[i] = torch.ones(k) / k  # mean
                continue
            user_graph_sample = self.user_graph_dict[i][0][:k]
            user_graph_weight = self.user_graph_dict[i][1][:k]

            if self.user_aggr_mode == 'softmax':
                user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)  # softmax
            if self.user_aggr_mode == 'mean':
                user_weight_matrix[i] = torch.ones(k) / k  # mean
            user_graph_index.append(user_graph_sample)

        # pdb.set_trace()
        return user_graph_index, user_weight_matrix


    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        # ndarray([598918, 2]) for ml-imdb
        return np.column_stack((rows, cols))

    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        # edge normalized values
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values

    def forward(self, adj):
        h = self.item_id_embedding.weight
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)

        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings + h

    def forward_new(self, interaction):
        # dragon
        user_nodes, raw_pos_item_nodes, raw_neg_item_nodes = interaction[0], interaction[1], interaction[2]
        pos_item_nodes = raw_pos_item_nodes + self.n_users
        neg_item_nodes = raw_neg_item_nodes + self.n_users

        if self.v_feat is not None:
            self.v_rep, _ = self.v_gcn(self.edge_index, self.image_trs(self.v_feat))
        if self.t_feat is not None:
            self.t_rep, _ = self.t_gcn(self.edge_index, self.text_trs(self.t_feat))

        v_rep = self.v_rep[self.num_user:]
        t_rep = self.t_rep[self.num_user:]

        ############################################ multi-modal information aggregation
        # ****************** 同质信息的处理， 实验中可以放开这部分 ******************
        # h = (homogen_v_rep + homogen_t_rep) / 2
        # # print('device:', self.mm_adj.device, h.device)  # 输出：cpu
        v_h = v_rep
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, v_h)
        v_item_rep = v_h + h
        h_u1 = self.user_graph(self.user_v_embedding, self.epoch_user_graph, self.user_weight_matrix)
        v_user_rep = self.user_v_embedding + h_u1

        t_h = t_rep
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, t_h)
        t_item_rep = t_h + h
        h_u1 = self.user_graph(self.user_t_embedding, self.epoch_user_graph, self.user_weight_matrix)
        t_user_rep = self.user_t_embedding + h_u1

        t_user_rep = torch.unsqueeze(t_user_rep, 2)
        v_user_rep = torch.unsqueeze(v_user_rep, 2)
        user_rep = torch.matmul(torch.cat((t_user_rep, v_user_rep), dim=2), self.weight_u)
        user_rep = torch.squeeze(user_rep)
       
        # 1. 初始分离同质和多样性信息
        v_homo = self.v_homo_map(v_rep)
        v_diver = v_rep - v_homo
        v_diver = self.v_diver_map(v_diver)
        v_diver = self.diversity_constraint(v_diver)

        t_homo = self.t_homo_map(t_rep)
        t_diver = t_rep - t_homo
        t_diver = self.t_diver_map(t_diver)
        t_diver = self.diversity_constraint(t_diver)
        
        # 2. 对同质和多样性信息分别应用残差块
        # 应用多层残差块进行特征融合
        v_homo = self.apply_resnet_blocks(v_homo, self.v_homo_res_blocks)
        v_diver = self.apply_resnet_blocks(v_diver, self.v_diver_res_blocks)
        
        t_homo = self.apply_resnet_blocks(t_homo, self.t_homo_res_blocks)
        t_diver = self.apply_resnet_blocks(t_diver, self.t_diver_res_blocks)

        hidden_v_embed_pos = torch.cat((user_rep, v_item_rep), dim=0)[pos_item_nodes]
        hidden_t_embed_pos = torch.cat((user_rep, t_item_rep), dim=0)[pos_item_nodes]
        hidden_v_embed_neg = torch.cat((user_rep, v_item_rep), dim=0)[neg_item_nodes]
        hidden_t_embed_neg = torch.cat((user_rep, t_item_rep), dim=0)[neg_item_nodes]

        homo_v_embed_pos = torch.cat((user_rep, v_homo), dim=0)[pos_item_nodes]
        homo_t_embed_pos = torch.cat((user_rep, t_homo), dim=0)[pos_item_nodes]
        homo_v_embed_neg = torch.cat((user_rep, v_homo), dim=0)[neg_item_nodes]
        homo_t_embed_neg = torch.cat((user_rep, t_homo), dim=0)[neg_item_nodes]

        diversity_v_embed_pos = torch.cat((user_rep, v_diver), dim=0)[pos_item_nodes]
        diversity_t_embed_pos = torch.cat((user_rep, v_diver), dim=0)[pos_item_nodes]
        diversity_v_embed_neg = torch.cat((user_rep, v_diver), dim=0)[neg_item_nodes]
        diversity_t_embed_neg = torch.cat((user_rep, t_item_rep), dim=0)[neg_item_nodes]


        def QKV(user_tensor, k_list):
            k_values_tensor = torch.stack(k_list)
            # print("k_values_tensor:", k_values_tensor.shape)

            # 步骤1：计算相似度得分（原始注意力分数）
            sim_scores = torch.einsum('nd,knd->kn', user_tensor, k_values_tensor)  # 形状 [K, N]
            # 步骤2：沿K维度对分数做Softmax归一化
            weights = torch.softmax(sim_scores, dim=0)     # 形状 [K, N]
            # 步骤3：对k向量加权并沿K维度求和
            weighted_k = k_values_tensor * weights.unsqueeze(-1)         # 形状 [K, N, D]
            final_item_rep = torch.sum(weighted_k, dim=0)          # 形状 [N, D]
            return final_item_rep

        user_tensor = user_rep[user_nodes]
        k_list = [hidden_v_embed_pos, homo_v_embed_pos, diversity_v_embed_pos, hidden_t_embed_pos, homo_t_embed_pos, diversity_t_embed_pos]
        pos_item_rep = QKV(user_tensor, k_list)

        k_list = [hidden_v_embed_neg, homo_v_embed_neg, diversity_v_embed_neg, hidden_t_embed_neg, homo_t_embed_neg, diversity_t_embed_neg] 
        neg_item_rep = QKV(user_tensor, k_list)
    
        pos_scores = torch.sum(user_tensor * pos_item_rep, dim=1)
        neg_scores = torch.sum(user_tensor * neg_item_rep, dim=1)

        # return pos_scores, neg_scores, t_homo, v_homo, t_diver, v_diver
        return user_tensor, pos_item_rep, neg_item_rep, t_homo, v_homo, t_diver, v_diver


    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings = self.forward(self.masked_adj)
        self.build_item_graph = False

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]
        batch_mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        mf_v_loss, mf_t_loss = 0.0, 0.0
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
            mf_t_loss = self.bpr_loss(ua_embeddings[users], text_feats[pos_items], text_feats[neg_items])
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
            mf_v_loss = self.bpr_loss(ua_embeddings[users], image_feats[pos_items], image_feats[neg_items])
        # return batch_mf_loss + self.reg_weight * (mf_t_loss + mf_v_loss)

        # dragon
        # user_tensor, pos_item_rep, neg_item_rep, t_rep, v_rep, v_inter, t_inter = self.forward_v1(interaction)
        user_tensor, pos_item_rep, neg_item_rep, t_rep, v_rep = self.forward_v1(interaction)
        dragon_v1_loss = self.calculate_dragon_loss_v1(users, user_tensor, pos_item_rep, neg_item_rep, t_rep, v_rep)
        # user_tensor, pos_item_rep, neg_item_rep, t_homo, v_homo, t_diver, v_diver = self.forward_new(interaction)
        # return batch_mf_loss + self.reg_weight * (mf_t_loss + mf_v_loss)
        return batch_mf_loss + self.reg_weight * (mf_t_loss + mf_v_loss) + self.dragon_weight * dragon_v1_loss


    def calculate_dragon_loss_v1(self, users, user_tensor, pos_item_rep, neg_item_rep, t_rep, v_rep, v_inter=None, t_inter=None):
        loss_value = self.bpr_loss(user_tensor, pos_item_rep, neg_item_rep)
        reg_embedding_loss_v = (self.v_preference[users] ** 2).mean() if self.v_preference is not None else 0.0
        reg_embedding_loss_t = (self.t_preference[users] ** 2).mean() if self.t_preference is not None else 0.0
        reg_loss = self.reg_weight * (reg_embedding_loss_v + reg_embedding_loss_t)
        reg_loss += self.reg_weight * (self.weight_u ** 2).mean()
        align_loss1 = self.v_t_align_loss(v_rep,t_rep)
        return loss_value + 0.01 * reg_loss + 0.01 * align_loss1

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

    def v_t_align_loss(self,v_rep,t_rep):
        def mmd_linear(t_rep, v_rep):
            """线性时间MMD近似"""
            mean_t = t_rep.mean(0)
            mean_v = v_rep.mean(0)
            mean_diff = (mean_t - mean_v).pow(2).sum()
            return mean_diff
        return mmd_linear(v_rep,t_rep)
    
    def v_t_diver_loss(self,v_rep,t_rep):
        # 这里是PAMD原始论文的做法
        dot_prod = torch.sum(torch.mul(v_rep, t_rep), dim=1)
        L_ort = torch.mean(dot_prod ** 2, dim=0)

        # 我突然想到这里也可以考虑同样的类似于MMD的方法，可以先取均值，再进行计算,如有必要,可以开放下面的部分:
        # # 计算两个模态的均值向量
        # mean_t = t_rep.mean(dim=0)  # [dim_latent]
        # mean_v = v_rep.mean(dim=0)  # [dim_latent]
        # mean_t = mean_t.squeeze()
        # mean_v = mean_v.squeeze()
        # # 计算内积的平方
        # dot_prod = torch.dot(mean_t, mean_v)  # 标量值
        # L_ort = dot_prod ** 2  # 内积的平方
        # # pdb.set_trace()
        return L_ort

    def diversity_constraint(self, diver_rep):
        """
        对多样性信息进行数值约束，防止过大/过小
        使用自适应梯度缩放技术，保持梯度流
        """
        # 1. 计算当前多样性信息的统计量
        rep_mean = diver_rep.mean()
        rep_std = diver_rep.std()
        
        # 2. 设置约束范围（可配置为超参数）
        clip_range = 2.0  # 标准差倍数范围
        
        # 3. 计算裁剪边界
        lower_bound = rep_mean - clip_range * rep_std
        upper_bound = rep_mean + clip_range * rep_std
        
        # 4. 使用torch.clamp进行裁剪
        clipped_rep = torch.clamp(diver_rep, min=lower_bound, max=upper_bound)
        
        # 5. 保持原始梯度（梯度重定向）
        return diver_rep + (clipped_rep - diver_rep).detach()

    def apply_resnet_blocks(self, rep, res_blocks):
        """应用多层残差块进行特征融合"""
        residual = rep
        for block in res_blocks:
            # 特征变换
            transformed = block(residual)
            
            # 残差连接
            residual = residual + transformed
        
        return residual
    
    def forward_v1(self, interaction):
        user_nodes, raw_pos_item_nodes, raw_neg_item_nodes = interaction[0], interaction[1], interaction[2]
        pos_item_nodes = raw_pos_item_nodes + self.n_users
        neg_item_nodes = raw_neg_item_nodes + self.n_users


        if self.v_feat is not None:
            self.v_rep, self.v_preference = self.v_gcn(self.edge_index, self.image_trs(self.v_feat))
        if self.t_feat is not None:
            self.t_rep, self.t_preference = self.t_gcn(self.edge_index, self.text_trs(self.t_feat))

        v_rep = self.v_rep[self.n_users:]
        t_rep = self.t_rep[self.n_users:]

        # ****************** 同质信息的处理， 实验中可以放开这部分 ******************
        t_h = t_rep
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, t_h)
        t_item_rep = t_h + h

        t_user_rep = self.t_rep[:self.n_users]
        h_u1 = self.user_graph(t_user_rep, self.epoch_user_graph, self.user_weight_matrix)
        t_user_rep = t_user_rep + h_u1


        v_h = v_rep
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, v_h)
        v_item_rep = v_h + h

        # user_matrix.shape torch.Size([19445, 1, 40]), u_features.shape. torch.Size([19445, 40, 64, 1]) 
        # ok: user_matrix: torch.Size([19445, 1, 40]) u_features.shape. torch.Size([19445, 40, 128]) 
        v_user_rep = self.v_rep[:self.n_users]
        h_u1 = self.user_graph(v_user_rep, self.epoch_user_graph, self.user_weight_matrix)
        v_user_rep = v_user_rep + h_u1

        t_user_rep = torch.unsqueeze(t_user_rep, 2)
        v_user_rep = torch.unsqueeze(v_user_rep, 2)
        user_rep = torch.matmul(torch.cat((t_user_rep, v_user_rep), dim=2), self.weight_u)
        user_rep = torch.squeeze(user_rep)

        v_rep_mlp = self.v_mlp(v_rep)
        t_rep_mlp = self.t_mlp(t_rep)
        v_rep = v_rep + v_rep_mlp
        t_rep = t_rep + t_rep_mlp

        item_rep = self.v_weight * (v_rep) + self.t_weight * (t_rep)

        self.result_embed = torch.cat((user_rep, item_rep), dim=0)

        user_tensor = self.result_embed[user_nodes]
        pos_item_tensor = self.result_embed[pos_item_nodes]
        neg_item_tensor = self.result_embed[neg_item_nodes]

        # v_inter_embed = torch.cat((user_rep, v_rep), dim=0)
        # t_inter_embed = torch.cat((user_rep, t_rep), dim=0)
        # v_inter = torch.cat([v_inter_embed[pos_item_nodes],v_inter_embed[neg_item_nodes]],dim=0)
        # t_inter = torch.cat([t_inter_embed[pos_item_nodes],t_inter_embed[neg_item_nodes]],dim=0)

        diversity_v_embed_pos = torch.cat((user_rep, v_item_rep), dim=0)[pos_item_nodes]
        diversity_t_embed_pos = torch.cat((user_rep, t_item_rep), dim=0)[pos_item_nodes]

        diversity_v_embed_neg = torch.cat((user_rep, v_item_rep), dim=0)[neg_item_nodes]
        diversity_t_embed_neg = torch.cat((user_rep, t_item_rep), dim=0)[neg_item_nodes]

        def QKV(user_tensor, k_list):
            k_values_tensor = torch.stack(k_list)
            # 步骤1：计算相似度得分（原始注意力分数）
            sim_scores = torch.einsum('nd,knd->kn', user_tensor, k_values_tensor)  # 形状 [K, N]
            # 步骤2：沿K维度对分数做Softmax归一化
            weights = torch.softmax(sim_scores, dim=0)     # 形状 [K, N]
            # 步骤3：对k向量加权并沿K维度求和
            weighted_k = k_values_tensor * weights.unsqueeze(-1)         # 形状 [K, N, D]
            final_item_rep = torch.sum(weighted_k, dim=0)          # 形状 [N, D]
            return final_item_rep

        k_list = [pos_item_tensor, diversity_v_embed_pos, diversity_t_embed_pos]
        pos_item_rep = QKV(user_tensor, k_list)

        k_list = [neg_item_tensor, diversity_v_embed_neg, diversity_t_embed_neg]
        neg_item_rep = QKV(user_tensor, k_list)

        pos_scores = torch.sum(user_tensor * pos_item_rep, dim=1)
        neg_scores = torch.sum(user_tensor * neg_item_rep, dim=1)
        # return pos_scores, neg_scores, t_rep, v_rep, v_inter, t_inter
        return user_tensor, pos_item_rep, neg_item_rep, t_rep, v_rep
        # return user_tensor, pos_item_rep, neg_item_rep, t_rep, v_rep, v_inter, t_inter


class GCN(torch.nn.Module):
    def __init__(self, user_embedding, n_users, num_item, aggr_mode,
                 feat_embed_dim=None, device=None, use_mlp=False):
        super(GCN, self).__init__()
        self.user_embedding = user_embedding
        self.n_users = n_users
        self.num_item = num_item
        self.aggr_mode = aggr_mode
        self.feat_embed_dim = feat_embed_dim
        self.device = device
        self.use_mlp = use_mlp

        self.user_embedding.to(self.device)
        if use_mlp:
            self.linear_1 = nn.Linear(self.feat_embed_dim, 4 * self.feat_embed_dim)
            self.linear_2 = nn.Linear(4 * self.feat_embed_dim, self.feat_embed_dim)
        self.conv_embed_1 = BaseGcn(self.feat_embed_dim, self.feat_embed_dim, aggr=self.aggr_mode)

    def forward(self, edge_index, features):
        # features: item multimodal feature
        if self.use_mlp:
            temp_features = F.leaky_relu(self.linear_1(features))
            temp_features = F.leaky_relu(self.linear_2(temp_features))

        else:
            temp_features = features
        # print("user.embedding.weight.shape: ", self.user_embedding.weight.shape)
        # print("tempp_features.shape", temp_features.shape)
        x = torch.cat((self.user_embedding, temp_features), dim=0).to(self.device)
        # x: user + item
        x = F.normalize(x).to(self.device)
        h = self.conv_embed_1(x, edge_index)  # equation 1
        h_1 = self.conv_embed_1(h, edge_index)
        x_hat = h + x + h_1
        return x_hat, self.user_embedding


class BaseGcn(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
        super(BaseGcn, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, size=None):
        if size is None:
            edge_index, _ = remove_self_loops(edge_index)
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        if self.aggr == 'add':
            row, col = edge_index
            deg = degree(row, size[0], dtype=x_j.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return norm.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class FFN(torch.nn.Module):
    def __init__(self, hidden_size, inner_hidden_size=None,
                 bias=True, activation=nn.ReLU):
        super(FFN, self).__init__()
        self.hidden_size = hidden_size
        if inner_hidden_size is None:
            inner_hidden_size = 4 * hidden_size
        else:
            self.inner_hidden_size = inner_hidden_size
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.inner_hidden_size, bias=bias),
            activation(),
            torch.nn.Linear(self.inner_hidden_size, self.hidden_size, bias=bias),
        )

    def forward(self, hidden_states):
        """
        hidden_states: [item_num,hidden_size]
        """
        output = self.layers(hidden_states)
        return output


class UserGraphSample(torch.nn.Module):
    """ sota_v1 """
    def __init__(self, num_user, aggr_mode, feat_embed_dim):
        super(UserGraphSample, self).__init__()
        self.num_user = num_user
        self.feat_embed_dim = feat_embed_dim
        self.aggr_mode = aggr_mode

    def forward(self, features, user_graph, user_matrix):
        index = user_graph
        u_features = features[index]
        user_matrix = user_matrix.unsqueeze(1)
        u_pre = torch.matmul(user_matrix, u_features)
        u_pre = u_pre.squeeze()
        return u_pre
