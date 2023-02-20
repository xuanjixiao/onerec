# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from components.transforms.base_transform import BaseTransform


class CategoricalTransform(BaseTransform):

    def __init__(self,
                 feature_names,
                 embed_size=128,
                 default_num_oov_buckets=1,
                 map_num_oov_buckets={},
                 map_top_k_to_select={},
                 map_shared_embedding={},
                 scope_name="categorical_transform"):
        self._feature_names = feature_names
        self._default_num_oov_buckets = default_num_oov_buckets
        self._map_num_oov_buckets = map_num_oov_buckets
        self._map_top_k_to_select = map_top_k_to_select
        self._map_shared_embedding = map_shared_embedding
        self._embed_size = embed_size
        self._scope_name = scope_name

        self._hash_tables, hash_sizes = self._create_hash_tables()
        self._embedding_tables = self._create_embedding_tables(hash_sizes)

    def _transform_fn(self, example):
        #example["duration"]=example["FEA_DURATION"]
        example = self._hash_lookup(example)
        example = self._embedding_lookup(example)
        return example

    def _hash_lookup(self, example):
        for fea_name in self._feature_names:
            if fea_name not in self._map_shared_embedding: 
                example[fea_name] = self._hash_tables[fea_name].lookup(
                    example[fea_name]
                )
            else:
                shared_fea_name = self._map_shared_embedding[fea_name]
                example[fea_name] = self._hash_tables[shared_fea_name].lookup(
                    example[fea_name]
                )
        return example

    def _embedding_lookup(self, example):
        """
        
        """
        for fea_name in self._feature_names:
            if fea_name not in self._map_shared_embedding:
                example[fea_name] = tf.nn.embedding_lookup(
                    self._embedding_tables[fea_name], example[fea_name]
                )
            else:
                shared_fea_name = self._map_shared_embedding[fea_name]
                example[fea_name] = tf.nn.embedding_lookup(
                    self._embedding_tables[shared_fea_name], example[fea_name]
                )
        return example

    def _create_hash_tables(self):
        hash_tables = {}
        hash_sizes = {}
        for fea_name in self._feature_names:
            if fea_name in self._map_shared_embedding:
                assert self._map_shared_embedding[fea_name] in self._feature_names
            else:
                num_oov_buckets = (
                    self._map_num_oov_buckets[fea_name]
                    if fea_name in self._map_num_oov_buckets
                    else self._default_num_oov_buckets
                )
                top_k = (
                    self._map_top_k_to_select[fea_name]
                    if fea_name in self._map_top_k_to_select
                    else None
                )
                vocab = ['-1']
                
                
                hash_tables[fea_name] = tf.contrib.lookup.index_table_from_tensor(
                    mapping=tf.constant(vocab), num_oov_buckets=num_oov_buckets
                )
                hash_sizes[fea_name] = len(vocab) + num_oov_buckets
        return hash_tables, hash_sizes

    def _create_embedding_tables(self, hash_sizes):
        embedding_tables = {}
        with tf.variable_scope(self._scope_name, reuse=tf.AUTO_REUSE):
            for fea_name in self._feature_names:
                if fea_name in self._map_shared_embedding:
                    assert self._map_shared_embedding[fea_name] in self._feature_names
                else:
                    embedding_tables[fea_name] = tf.get_variable(
                        fea_name + "_embed", [hash_sizes[fea_name], self._embed_size]
                    )
        return embedding_tables
