# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from components.networks.base_network import BaseNetwork


class DlrmSparseNetwork(BaseNetwork):

    def __init__(self,
                 categorical_features,
                 numerical_features,
                 multivalue_features,
                 attention_features,
                 loss_ctr,
                 loss_cvr,
                 ptype="mean",
                 hidden_sizes=[512, 512, 512],
                 scope_name="dlrm2_network",
                 save_model_mode='placeholder',
                 save_task_id=0,
                 masks_dir=None
                 ):
        self._categorical_features = categorical_features
        self._numerical_features = numerical_features
        self._multivalue_features = multivalue_features
        self._attention_features = attention_features
        self._loss_ctr = loss_ctr
        self._loss_cvr = loss_cvr
        self._hidden_sizes = hidden_sizes
        self._scope_name = scope_name
        self._ptype = ptype
        self._mode = save_model_mode
        self._weights = {}
        self.masks ={}
        self.masks_dir=masks_dir
        self._save_task_id=save_task_id
    def _dense_layer(self,
                  name,
                  inputs,
                  units,
                  mask,
                  activation=None,
                  use_bias=True,
                  kernel_initializer=tf.contrib.layers.xavier_initializer(
              uniform=False)):
    #Mimics tf.dense_layer but masks weights and uses presets as necessary.
    # If there is a preset for this layer, use it.

        # Create the weights.
        weights = tf.get_variable(
            name=name + '_w',
            shape=[inputs.shape[1], units],
            initializer=kernel_initializer,trainable=True)

        weights = tf.multiply(weights, mask)

        self._weights[name] = weights

        # Compute the output.
        output = tf.matmul(inputs, weights)

        # Add bias if applicable.
        if use_bias:
            bias = tf.get_variable(
            name=name + '_biases', shape=[units], initializer=tf.zeros_initializer())
            output += bias
        
        # Activate.
        if activation:
            return activation(output)
        else:
            return output


    def _train_fn(self, example):
        with tf.variable_scope(self._scope_name, reuse=tf.AUTO_REUSE):
            logits_ctr,logits_cvr = self._build_graph(example)
            #logits = self._build_graph(example)
            loss_ctr = self._loss_ctr.loss_fn(logits_ctr, example)
            loss_cvr = self._loss_cvr.loss_fn(logits_cvr,example)
            return loss_ctr,loss_cvr

    def _eval_fn(self, example):
        with tf.variable_scope(self._scope_name, reuse=tf.AUTO_REUSE):
            logits_ctr,logits_cvr = self._build_graph(example)
            #logits = self._build_graph(example)
            outputs_ctr = tf.sigmoid(logits_ctr)
            #outputs_cvr = tf.nn.relu(logits)
            outputs_cvr = tf.sigmoid(logits_cvr)
            return outputs_ctr,outputs_cvr

    def _get_serve_inputs(self):
        # inputs = self._numerical_features + self._categorical_features
        inputs = self._categorical_features.copy()
        for feature in self._multivalue_features:
           inputs.append(feature)
        return inputs
    
    def _serve_fn_opt(self, example):
        batch_size = self._get_batch_size(example)
        with tf.variable_scope(self._scope_name, reuse=tf.AUTO_REUSE):
            categorical_part = self._tile_tensors_with_batch_size(
                self._build_categorical_part(example),
                batch_size)
            # numerical_part = self._tile_tensors_with_batch_size(
            #    self._build_numerical_part(example),
            #    batch_size)
            multivalue_part = self._tile_tensors_with_batch_size(
               self._build_multivalue_part(example),
               batch_size)
            # hiddens = tf.stack(categorical_part + numerical_part + multivalue_part,
            hiddens = tf.stack(categorical_part + multivalue_part,
            # hiddens = tf.stack(categorical_part,
                               axis=1)
            #hiddens_ctr=self._build_cross_interaction(hiddens)
            #hiddens_cvr=self._build_cross_interaction(hiddens)
            hiddens = self._build_cross_interaction(hiddens)
            #logits = self._build_upper_part_graph(hiddens)
            logits = self._build_upper_part_graph_saved_pb(hiddens)
            outputs_ctr = tf.sigmoid(logits)
            outputs_cvr = tf.nn.relu(logits)
            outputs=[outputs_ctr,outputs_cvr]
            print("saved pb task_id {}".format(self._save_task_id))
        return outputs[self._save_task_id]#,outputs_cvr

    def _serve_fn(self, example):
        if self._mode == 'example':
            outputs = self._eval_fn(example)
        else:
            outputs = self._serve_fn_opt(example)
        return outputs

    def _build_graph(self, inputs):
        hiddens = self._build_lower_part_graph(inputs)
        #hiddens_ctr=self._build_cross_interaction(hiddens)
        #hiddens_cvr=self._build_cross_interaction(hiddens)
        hiddens = self._build_cross_interaction(hiddens)
        outputs_ctr,outputs_cvr = self._build_upper_part_graph(hiddens)

        #outputs_ctr = self._build_upper_part_graph(hiddens_ctr)
        #outputs_cvr = self._build_upper_part_graph(hiddens_cvr)
        return outputs_ctr,outputs_cvr
        #return outputs

    def _build_lower_part_graph(self, inputs):
        categorical_part = self._build_categorical_part(inputs)
        # numerical_part = self._build_numerical_part(inputs)
        multivalue_part = self._build_multivalue_part(inputs)
        # attention_part = self._build_attention_part(inputs)
        # return tf.stack(categorical_part + numerical_part + multivalue_part, axis=1)
        # return tf.stack(categorical_part + multivalue_part + attention_part, axis=1)
        return tf.stack(categorical_part + multivalue_part, axis=1)
        # return tf.stack(categorical_part, axis=1)

    def _build_upper_part_graph(self, inputs):
        hidden = inputs
        for i, size in enumerate(self._hidden_sizes):
            mask_name="fc_"+str(i)+"_m"
            if not mask_name in self.masks:
                mask=tf.placeholder(tf.float32,shape=(hidden.shape[1],size),name="fc_"+str(i)+"_m")
                self.masks[mask_name]=mask  #prevent define mask repeatedly
            hidden = self._dense_layer(name="fc_"+str(i),inputs=hidden,units=size,mask=self.masks[mask_name],activation=tf.nn.relu)
         
        #for i, size in enumerate(self._hidden_sizes):
        #   hidden = slim.fully_connected(hidden, size, scope="fc_" + str(i))
        logit_ctr=slim.fully_connected(hidden, 1, activation_fn=None, scope="logit_ctr")
        logit_cvr=slim.fully_connected(hidden, 1, activation_fn=None, scope="logit_cvr")
        #return slim.fully_connected(hidden, 1, activation_fn=None, scope="logit")
        return logit_ctr,logit_cvr

    def _build_upper_part_graph_v2(self, inputs):
        hidden = inputs
        for i, size in enumerate(self._hidden_sizes):
            mask_name="fc_"+str(i)+"_m"
            if not mask_name in self.masks:
                mask=tf.placeholder(tf.float32,shape=(hidden.shape[1],size),name="fc_"+str(i)+"_m")
                self.masks[mask_name]=mask  #prevent define mask repeatedly
            hidden = self._dense_layer(name="fc_"+str(i),inputs=hidden,units=size,mask=self.masks[mask_name],activation=tf.nn.relu)
         
        #for i, size in enumerate(self._hidden_sizes):
        hidden_ctr = slim.fully_connected(inputs, 512, scope="hidden_ctr_" + str(0))
        hidden_ctr = tf.concat([hidden,hidden_ctr],axis=1)
        hidden_ctr = slim.fully_connected(hidden_ctr, 256, scope="hidden_ctr_" + str(1))
        hidden_ctr = slim.fully_connected(hidden_ctr, 128, scope="hidden_ctr_" + str(2))

        hidden_cvr = slim.fully_connected(inputs, 512, scope="hidden_cvr_" + str(0))
        hidden_cvr = tf.concat([hidden,hidden_cvr],axis=1)
        hidden_cvr = slim.fully_connected(hidden_cvr, 256, scope="hidden_cvr_" + str(1))
        hidden_cvr = slim.fully_connected(hidden_cvr, 128, scope="hidden_cvr_" + str(2))

        logit_ctr=slim.fully_connected(hidden_ctr, 1, activation_fn=None, scope="logit_ctr")
        logit_cvr=slim.fully_connected(hidden_cvr, 1, activation_fn=None, scope="logit_cvr")
        #return slim.fully_connected(hidden, 1, activation_fn=None, scope="logit")
        return logit_ctr,logit_cvr
    def _build_upper_part_graph_saved_pb(self, inputs):
        hidden = inputs
        mask_dic=self._load_masks(self._save_task_id,self.masks_dir)##dlrm2_network/fc_1_w:0
        for i, size in enumerate(self._hidden_sizes):
            mask_name="fc_"+str(i)+"_m"
            key=self._scope_name+"/fc_"+str(i)+"_w:0"
            print(key)
            m=mask_dic[key]
            #b=np.ones((hidden.shape[1],size))
            mask=tf.constant(m,dtype=tf.float32,shape=(hidden.shape[1],size),name=mask_name)
            
            #mask=tf.placeholder(tf.float32,shape=(hidden.shape[1],size),name=mask_name)
            #self.masks[mask_name]=mask  #prevent define mask repeatedly
            hidden = self._dense_layer(name="fc_"+str(i),inputs=hidden,units=size,mask=mask,activation=tf.nn.relu)
         
        #for i, size in enumerate(self._hidden_sizes):
        #   hidden = slim.fully_connected(hidden, size, scope="fc_" + str(i))
        return slim.fully_connected(hidden, 1, activation_fn=None, scope="logit")

    def _build_cross_interaction(self, inputs):
        cross = tf.matmul(inputs, tf.transpose(inputs, perm=[0, 2, 1]))
        cross = tf.reshape(cross, [-1, cross.get_shape()[1] * cross.get_shape()[2]])
        flatten = tf.reshape(inputs,
                             [-1, inputs.get_shape()[1] * inputs.get_shape()[2]])
        outputs = tf.concat([cross, flatten], axis=1)
        return outputs

    def _build_categorical_part(self, inputs):
        return [tf.squeeze(inputs[name], axis=1) for name in self._categorical_features]

    def _build_numerical_part(self, inputs):
        outputs = []
        if len(self._numerical_features) != 0:
            h0 = tf.concat([inputs[name] for name in self._numerical_features], axis=1)
            h1 = slim.fully_connected(h0, 128, scope="numerical_fc1")
            h2 = slim.fully_connected(h1, 64, scope="numerical_fc2")
            outputs = slim.fully_connected(h2, 32, scope="numerical_fc3")
        return [outputs]

    def _build_multivalue_part(self, inputs):
        def pooling(vals, ptype, fea_name):
            if ptype == "mean":
                return tf.reduce_mean(vals, axis=1)
            elif ptype == "sum":
                return tf.reduce_sum(vals, axis=1)
            elif ptype == "fc":
                return tf.squeeze(slim.fully_connected(
                    tf.transpose(vals, [0, 2, 1]),
                    1,
                    scope="multivalue_fc_%s" % fea_name))

        outputs = [pooling(inputs[name], self._ptype, name) 
            for name in self._multivalue_features]
        return outputs

    def _build_attention_part(self, inputs):
        # key: [batch, dim]
        # vals: [batch, vlen, dim]
        # out: [batch, dim], weighted sum of vals
        def attention(key, vals):
            #key = tf.expand_dims(key, axis=1)
            weight = tf.reduce_sum(tf.multiply(key, vals), axis=2)
            sum_weight = tf.reduce_sum(weight, axis=1, keepdims=True)
            norm_weight = weight / sum_weight
            return tf.reduce_sum(tf.multiply(tf.expand_dims(norm_weight, axis=2), vals),
                    axis=1)
        outputs = [attention(inputs[names[0]], inputs[names[1]])
            for names in self._attention_features]
        return outputs
    
    def _load_masks(self,task_id,masks_dir):
        """mlt/mask/task_id_mask.pkl
        """
        import os
        import pickle
        
        masks_dir=masks_dir
        masks_path = [
            os.path.join(masks_dir, f)
            for f in os.listdir(masks_dir)
            if not f.startswith("init")
        ]
        masks_path = list(
            sorted(
                filter(lambda f: os.path.isfile(f), masks_path),
                key=lambda s: int(os.path.basename(s).split(".pkl")[0]),
            )
        )
        masks = []
        print("loading masks")

        for path in masks_path:
            with open(path, "rb") as f:
                dump = pickle.load(f)
            assert "mask" in dump and "pruning_time" in dump
            print(
                "loading pruning_time {}".format(dump["pruning_time"])
            )
            masks.append(dump["mask"])
        # sanity check
        assert len(masks) == len(masks_path)

        mask=masks[task_id]
        #for name, p in mask.items():
        #   print(name,p)#dlrm2_network/fc_1_w:0,
        return mask

