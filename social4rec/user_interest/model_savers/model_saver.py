# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.lib.io import file_io


class ModelSaver(object):

    def __init__(self,
                 transform_functions,
                 serve_fn,
                 restore_checkpoint_path,
                 save_model_dir):
        self._transform_functions = transform_functions
        self._serve_fn = serve_fn
        self._restore_checkpoint_path = restore_checkpoint_path
        self._save_model_dir = save_model_dir

        self._predict, self._input_example = self._build_serve_graph()

    def run(self):
        self._sess = self._create_and_init_session()
        self._restore_checkpoint()
        self._save_model()
        self._sess.close()

    def _save_model(self):
        signature = tf.saved_model.signature_def_utils.predict_signature_def(
            inputs={"example": self._input_example}, outputs={"predict": self._predict}
        )
        if file_io.file_exists(self._save_model_dir):
            file_io.delete_recursively(self._save_model_dir)
        builder = tf.saved_model.builder.SavedModelBuilder(self._save_model_dir)
        import tensorflow.saved_model.signature_constants as s_const

        builder.add_meta_graph_and_variables(
            sess=self._sess,
            tags=[tf.saved_model.tag_constants.SERVING],
            signature_def_map={s_const.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature},
        )
        builder.save()

    def _create_and_init_session(self):
        sess = tf.Session()
        tf.global_variables_initializer().run(session=sess)
        tf.tables_initializer(name="init_all_tables").run(session=sess)
        return sess

    def _restore_checkpoint(self):
        if self._restore_checkpoint_path:
            checkpoint_saver = tf.train.Saver(max_to_keep=None)
            checkpoint_saver.restore(self._sess, self._restore_checkpoint_path)

    def _build_serve_graph(self):
        example = tf.placeholder(tf.string, shape=(None,), name="example")
        transform_fn = self._join_pipeline(self._transform_functions)
        predict = self._serve_fn(transform_fn(example))
        return predict, example

    def _join_pipeline(self, map_functions):

        def joined_map_fn(example):
            for map_fn in map_functions:
                example = map_fn(example)
            return example

        return joined_map_fn
