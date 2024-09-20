# coding: utf-8

#
# updated: Mar. 25, 2022
# Filled non-existing raw features with non-zero after encoded from encoders

"""
Data pre-processing
##########################
"""
from logging import getLogger
from collections import Counter
import os
import pandas as pd
import numpy as np
import torch
from utils.data_utils import (ImageResize, ImagePad, image_to_tensor, load_decompress_img_from_lmdb_value)
import lmdb


class RecDataset(object):
    def __init__(self, config, df=None):
        self.config = config
        self.logger = getLogger()

        # data path & files
        self.dataset_name = config['dataset']
        self.dataset_path = os.path.abspath(config['data_path']+self.dataset_name)

        # dataframe
        self.uid_field = self.config['USER_ID_FIELD']
        self.iid_field = self.config['ITEM_ID_FIELD']
        self.splitting_label = self.config['inter_splitting_label']

        # Multi-modal
        self.txt_features = None
        self.aud_features = None
        self.img_features = None

        if df is not None:
            self.df = df
            return
        # if all files exists
        check_file_list = [self.config['inter_file_name']]
        if self.config['use_raw_features']:
            check_file_list.extend([self.config['text_file_name'], self.config['img_dir_name']+'/data.mdb'])
        for i in check_file_list:
            file_path = os.path.join(self.dataset_path, i)
            if not os.path.isfile(file_path):
                raise ValueError('File {} not exist'.format(file_path))

        # load rating file from data path?
        self.load_inter_graph(config['inter_file_name'])
        self.item_num = int(max(self.df[self.iid_field].values)) + 1
        self.user_num = int(max(self.df[self.uid_field].values)) + 1

    def load_inter_graph(self, file_name):
        inter_file = os.path.join(self.dataset_path, file_name)
        cols = [self.uid_field, self.iid_field, self.splitting_label]
        self.df = pd.read_csv(inter_file, usecols=cols, sep=self.config['field_separator'])
        if not self.df.columns.isin(cols).all():
            raise ValueError('File {} lost some required columns.'.format(inter_file))

    def split(self):
        dfs = []
        # splitting into training/validation/test
        for i in range(3):
            temp_df = self.df[self.df[self.splitting_label] == i].copy()
            temp_df.drop(self.splitting_label, inplace=True, axis=1)        # no use again
            dfs.append(temp_df)
        if self.config['filter_out_cod_start_users']:
            # filtering out new users in val/test sets
            train_u = set(dfs[0][self.uid_field].values)
            for i in [1, 2]:
                dropped_inter = pd.Series(True, index=dfs[i].index)
                dropped_inter ^= dfs[i][self.uid_field].isin(train_u)
                dfs[i].drop(dfs[i].index[dropped_inter], inplace=True)

        if self.config['use_raw_features']:
            self._setup_features()

        # wrap as RecDataset
        full_ds = [self.copy(_) for _ in dfs]
        return full_ds

    def _setup_features(self):
        file_path = os.path.join(self.dataset_path, self.config['text_file_name'])
        # text中没有feature的补空
        _cnt_idx, com_sentences = 1, ['']       # item 0
        id_field, txt_featrue_field = self.config['ITEM_ID_FIELD'], self.config['TEXT_ID_FIELD']
        if self.dataset_name == 'ml-imdb':
            # [movieID,title,imdbID,genre,desc]
            self.txt_df = pd.read_csv(file_path, usecols=['movieID', 'desc', 'genre'])
            #self.txt_df['desc'].fillna('', inplace=True)
        elif self.dataset_name == 'games' or self.dataset_name == 'cds':
            self.txt_df = pd.read_csv(file_path, usecols=['itemID', 'description', 'title', 'category'])
            self.txt_df['description'] = self.txt_df['title'] + ' ' + self.txt_df['description']
        # For test
        for _, row in self.txt_df.iterrows():
            while _cnt_idx < row[id_field]:
                com_sentences.append('')
                _cnt_idx += 1
            if pd.isnull(row[txt_featrue_field]):
                com_sentences.append('')
            else:
                com_sentences.append(row[txt_featrue_field].strip())
            _cnt_idx += 1
        assert len(com_sentences) == self.item_num, 'text features not equal # of items.'

        self.txt_features, self.txt_masks = self.load_txt_features(com_sentences)    # len(ids) * hidden_size
        self.img_features, self.img_masks = self.load_img_features()                                    # len(ids) * channel * H * W
        ### useing dummy features
        #self.txt_features = torch.randint(30522, (self.item_num, self.config['max_txt_len'])).type(torch.IntTensor)
        #self.img_features = torch.rand((self.item_num, 3, self.config['max_img_size'], self.config['max_img_size']))

    def load_txt_features(self, sentences):
        from transformers import BertTokenizer
        # Load the BERT tokenizer.
        self.logger.info('Loading BERT tokenizer...')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        # Tokenize all the sentences and map the tokens to their word IDs.
        txt_max_len = self.config['max_txt_len']
        input_ids, attention_masks = [], []

        for sent in sentences:
            if sent == '':
                input_ids.append(torch.zeros((1, txt_max_len)))         # 0
                attention_masks.append(torch.zeros((1, txt_max_len)))   # 0
                continue
            encoded_dict = tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
                max_length=txt_max_len,  # Pad & truncate all sentences.
                # pad_to_max_length=True,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
            )
            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict['input_ids'])
            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])

        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0).type(torch.IntTensor)  # len(ids) * hidden_size
        attention_masks = torch.cat(attention_masks, dim=0).type(torch.IntTensor)
        # Print sentence 0, now as a list of IDs.
        print('Original: ', sentences[1])
        print('Token IDs:', input_ids[1])
        return input_ids, attention_masks

    def load_img_features(self):
        ############## Image features
        max_img_size = self.config['max_img_size']
        img_lmdb_dir = os.path.join(self.dataset_path, self.config['img_dir_name'])
        self.img_resize = ImageResize(max_img_size, "bilinear")  # longer side will be resized to max_img_size
        self.img_pad = ImagePad(max_img_size, max_img_size)  # pad to max_img_size * max_img_size
        self.env = lmdb.open(img_lmdb_dir, readonly=True, create=False)  # readahead=not _check_distributed()
        self.txn = self.env.begin(buffers=True)
        # load images
        img_arr, img_masks = [], []
        for i in range(self.item_num):
            i_arr, i_masks = self._load_img(i)
            img_arr.append(i_arr)
            img_masks.append(i_masks)
        return torch.vstack(img_arr), torch.tensor(img_masks)  # len(ids)*chanel*h*w

    def _load_img(self, img_id):
        """Load and apply transformation to image
        Returns:
            torch.float, in [0, 255], (n_frm=1, c, h, w)
        """
        tmp_v = self.txn.get(str(img_id).encode("utf-8"))
        if not tmp_v:
            return torch.zeros(1, 3, self.config['max_img_size'], self.config['max_img_size']), 0
        raw_img = load_decompress_img_from_lmdb_value(tmp_v)
        image_np = np.array(raw_img, dtype=np.uint8)  # (h, w, c)
        raw_img_tensor = image_to_tensor(image_np, keepdim=False).float()  # (c, h, w) [0, 255]
        resized_img = self.img_resize(raw_img_tensor)
        transformed_img = self.img_pad(resized_img)  # (n_frm=1, c, h, w)
        return transformed_img, 1

    def _extract_feat_remap_disk(self, u_id_map, i_id_map):
        files = [os.path.join(self.dataset_path, '{}_feat_sample.npy'.format(i)) for i in 'atv']
        self.atv_feats = [np.load(i, allow_pickle=True) if os.path.isfile(i) else None for i in files]
        print(len(self.atv_feats))
        # mapping

    def copy(self, new_df):
        """Given a new interaction feature, return a new :class:`Dataset` object,
                whose interaction feature is updated with ``new_df``, and all the other attributes the same.

                Args:
                    new_df (pandas.DataFrame): The new interaction feature need to be updated.

                Returns:
                    :class:`~Dataset`: the new :class:`~Dataset` object, whose interaction feature has been updated.
                """
        nxt = RecDataset(self.config, new_df)
        if self.config['use_raw_features']:
            nxt.txt_features = self.txt_features
            nxt.txt_masks = self.txt_masks
            nxt.img_features = self.img_features
            nxt.img_masks = self.img_masks

        nxt.item_num = self.item_num
        nxt.user_num = self.user_num
        return nxt

    def get_user_num(self):
        return self.user_num

    def get_item_num(self):
        return self.item_num

    def shuffle(self):
        """Shuffle the interaction records inplace.
        """
        self.df = self.df.sample(frac=1, replace=False).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Series result
        return self.df.iloc[idx]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        info = [self.dataset_name]
        self.inter_num = len(self.df)
        uni_u = pd.unique(self.df[self.uid_field])
        uni_i = pd.unique(self.df[self.iid_field])
        tmp_user_num, tmp_item_num = 0, 0
        if self.uid_field:
            tmp_user_num = len(uni_u)
            avg_actions_of_users = self.inter_num/tmp_user_num
            info.extend(['The number of users: {}'.format(tmp_user_num),
                         'Average actions of users: {}'.format(avg_actions_of_users)])
        if self.iid_field:
            tmp_item_num = len(uni_i)
            avg_actions_of_items = self.inter_num/tmp_item_num
            info.extend(['The number of items: {}'.format(tmp_item_num),
                         'Average actions of items: {}'.format(avg_actions_of_items)])
        info.append('The number of inters: {}'.format(self.inter_num))
        if self.uid_field and self.iid_field:
            sparsity = 1 - self.inter_num / tmp_user_num / tmp_item_num
            info.append('The sparsity of the dataset: {}%'.format(sparsity * 100))
        return '\n'.join(info)
