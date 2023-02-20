# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf
from absl import app
from absl import flags
import yaml

sys.path.append("..")
from components.statistics_gens.dataset_statistics_gen import DatasetStatisticsGen
from components.datasets.tfrecord_dataset import TFRecordDataset
from components.evaluators.evaluator import Evaluator
from components.networks.dlrm_sparse_network import DlrmSparseNetwork
from components.trainers.trainer_mtl import TrainerMTL
from components.trainers.trainer_single import TrainerSingle
from components.transforms.select_transform import FeatureSelector
from components.transforms.categorical_transform import CategoricalTransform
from pipelines.utils.config_parser import parse_feature_configs
from pipelines.utils.config_parser import parse_metric_configs
from pipelines.utils.config_parser import parse_loss_configs
from pipelines.utils.util import tf_global_config



FLAGS = flags.FLAGS
flags.DEFINE_string("config_filepath", "./drank_tf.yaml", "Yaml config filepath.")                 
flags.DEFINE_string("cut_names", "fc", "need cut layer")
flags.DEFINE_string("training_mode", "single", "single or mtl")
flags.DEFINE_float("prune_final_rate",0.2,"the final rate of left weights")
flags.DEFINE_integer("prune_pruning_iter",4,"the iterator time")
flags.DEFINE_integer("task_id",0,"sparse sharing single task id")
flags.DEFINE_string("test_hour", "202003281900", "training data partition")
flags.DEFINE_string("train_hour", "202003281900", "training data partition")
flags.DEFINE_string("validate_hour", "202003282000", "validation data partition")

def get_complete_data_path(conf):
    tp = [conf["train_tfrecord_path"]]
    print("Train path:%s" % tp)

    testp = []
    if len(conf["test_tfrecord_path"]) > 0:
        testp = [conf["test_tfrecord_path"]]
    print("test path:%s" % testp)

    vp = []
    if len(conf["val_tfrecord_path"]) > 0:
        vp = [conf["val_tfrecord_path"]]
    print("Val path:%s" % vp)
    return tp, testp, vp


def main_pipeline():
    
    config_filepath = FLAGS.config_filepath
    configs = yaml.safe_load(open(config_filepath))

    print("=============================%s===================================" % "start")
    train_path, test_path, val_path = get_complete_data_path(configs)
     

    print("Phase 1. Create Feature Selector and Datasets ---------------------")
    feature_configs = parse_feature_configs(configs["feature_configs"])
   
    selector = FeatureSelector(feature_configs=feature_configs)
    dataset_train = TFRecordDataset(
        filepath=train_path,
        batch_size=configs["batch_size"],
        file_shuffle=configs["train_file_shuffle"],
        shuffle_buffer_size=configs["train_shuffle_buffer_size"],
        map_functions=[selector.transform_fn],
        drop_remainder=False,
        sloppy=True,
    )
    dataset_train = TFRecordDataset(
        filepath=train_path,
        batch_size=configs["batch_size"],
        file_shuffle=configs["train_file_shuffle"],
        shuffle_buffer_size=configs["train_shuffle_buffer_size"],
        map_functions=[selector.transform_fn],
        drop_remainder=False,
        sloppy=True,
    )

    dataset_test = TFRecordDataset(
        filepath=test_path,
        batch_size=configs["batch_size"],
        file_shuffle=False,
        shuffle_buffer_size=1,
        map_functions=[selector.transform_fn],
        drop_remainder=False,
        sloppy=False,
    )
    dataset_val = TFRecordDataset(
        filepath=val_path,
        batch_size=configs["batch_size"],
        file_shuffle=False,
        shuffle_buffer_size=1,
        map_functions=[selector.transform_fn],
        drop_remainder=False,
        sloppy=False,
    )
    
    gen =DatasetStatisticsGen(dataset_train)
    s=gen.run()
    print(s)
    print("Phase 2. Create Transforms and Network ----------------------------")
    transform1 = CategoricalTransform(
        feature_names=configs["transform"]["categorical_features"],
        default_num_oov_buckets=configs["default_num_oov_buckets"],
        embed_size=configs["embed_size"],
        map_num_oov_buckets=configs["map_num_oov_buckets"],
        map_top_k_to_select=configs["map_top_k_to_select"],
        map_shared_embedding=configs["map_shared_embedding"],
    )

    loss_ctr = parse_loss_configs(configs["ctrloss"])
    loss_cvr = parse_loss_configs(configs["cvrloss"])
    network = DlrmSparseNetwork(
        categorical_features=configs["network"]["categorical_features"],
        numerical_features=configs["network"]["numerical_features"],
        multivalue_features=configs["network"]["multivalue_features"],
        attention_features=configs["network"]["attention_features"],
        ptype=configs["network"]["pooling_type"],
        loss_ctr=loss_ctr,
        loss_cvr= loss_cvr,
        save_model_mode="placehodler",
        hidden_sizes=configs["hidden_size"]
    )
    
    print("Phase 3. Create Evaluator and Trainer, and Run --------------------")
    ctr_metrics = parse_metric_configs(configs["ctrmetrics"])
    cvr_metrics = parse_metric_configs(configs["cvrmetrics"])
    tester = Evaluator(
        dataset=dataset_test,
        transform_functions=[transform1.transform_fn],
        eval_fn=network.eval_fn,
        ctr_metrics=ctr_metrics,
        cvr_metrics=cvr_metrics,
        restore_checkpoint_path=configs["restore_checkpoint_path"]
    )
    evaluator = Evaluator(
        dataset=dataset_val,
        transform_functions=[transform1.transform_fn],
        eval_fn=network.eval_fn,
        ctr_metrics=ctr_metrics,
        cvr_metrics=cvr_metrics,
        restore_checkpoint_path=configs["restore_checkpoint_path"]
    )
    if FLAGS.training_mode == "mtl":
        trainer = TrainerMTL(
            datasets=dataset_train,
            transform_functions=[transform1.transform_fn],
            train_fn=network.train_fn,
            validate_steps=configs["validate_steps"],
            log_steps=configs["log_steps"],
            learning_rate=configs["learning_rate"],
            train_epochs=configs["train_epochs"],
            evaluator=evaluator,
            tester=tester,
            save_checkpoints_dir=configs["save_mtl_checkpoints_dir"],
            restore_checkpoint_path=configs["restore_mtl_checkpoint_path"],
            tensorboard_logdir=configs["tensorboard_logdir"],
            train_hour=FLAGS.train_hour,
            test_hour=FLAGS.test_hour,
            validate_hour=FLAGS.validate_hour,
            validate_duaring_training=configs["validate_duaring_training"],
            cut_name=FLAGS.cut_names
        )
    else:
        trainer = TrainerSingle(
        dataset=dataset_train,
        transform_functions=[transform1.transform_fn],
        train_fn=network.train_fn,
        validate_steps=configs["validate_steps"],
        log_steps=configs["log_steps"],
        learning_rate=configs["learning_rate"],
        train_epochs=configs["train_epochs"],
        evaluator=evaluator,
        tester=tester,
        save_checkpoints_dir=configs["save_checkpoints_dir"],
        restore_checkpoint_path=configs["restore_checkpoint_path"],
        tensorboard_logdir=configs["tensorboard_logdir"],
        train_hour=FLAGS.train_hour,
        test_hour=FLAGS.test_hour,
        validate_hour=FLAGS.validate_hour,
        validate_duaring_training=configs["validate_duaring_training"],
        cut_name=FLAGS.cut_names,
        final_rate=FLAGS.prune_final_rate,
        pruning_iter=FLAGS.prune_pruning_iter,
        task_id=FLAGS.task_id
    )
    trainer.run()

    print("All done\n\n")

def main(argv):
    tf_global_config(intra_threads=8, inter_threads=8)
    main_pipeline()
    


if __name__ == "__main__":
    app.run(main)