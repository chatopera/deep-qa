# coding: utf8
# Copyright 2017 Hai Liang Wang. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
'''
Train Model with TensorFlow
Author: Hai Liang Wang <hailiang.hl.wang@gmail.com>
'''

import os
import tensorflow as tf
import shutil
from config import Config
from tqdm import tqdm
from utils import log
from munch import munchify
from models.rnn import Model
from dataset.textdata import TextData
from time import localtime, strftime

config = Config()
logger = log.getLogger(__name__)

def main(unused_argv):
    batch_data = TextData(munchify({
        'rootDir': config.root_dir,
        'corpus': config.corpus_name,
        'maxLength': config.train_max_length,
        'maxLengthEnco': config.train_max_length_enco,
        'maxLengthDeco': config.train_max_length_deco,
        'datasetTag': '',
        'test': False,
        'watsonMode': False,
        'batchSize': config.train_num_batch_size
    }))
    with tf.device(None):
        tf_model = Model(config, config.dataset)
    tf_writer = tf.train.SummaryWriter(config.model_save_dir)
    tf_saver = tf.train.Saver(max_to_keep=200)  # Arbitrary limit
    tf_sess = tf.Session()
    tf_sess.run(tf.initialize_all_variables())

    def save_session():
        """
        Save the model parameters and the variables
        """
        tqdm.write('Checkpoint reached: saving model (don\'t stop the run)...')
        # TODO Save config as parameters.
        # Save the model parameters and the variables
        logger.info('Save tf session ... %s' % config.model_save_ckpt)
        tf_saver.save(tf_sess, config.model_save_ckpt)

        logger.info('Copy source code ... %s' % config.model_save_dir)
        sourcecode_path = os.path.join(config.model_save_dir, 'deepqa2')
        if not os.path.exists(sourcecode_path):
            shutil.copytree(os.path.join(config.root_dir,
                                         'deepqa2'), sourcecode_path)

        logger.info('Create logs dir ...')
        if not os.path.exists(os.path.join(config.model_save_dir, 'logs')):
            os.makedirs(os.path.join(config.model_save_dir, 'logs'))

        dataset_path = os.path.join(config.model_save_dir, 'data')
        logger.info('Copy dataset ... %s' % dataset_path)
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        shutil.copy(config.dataset_pkl_path, dataset_path)

        logger.info('Save config.ini ... %s' % config.model_save_dir)
        shutil.copy(config.config_ini_path, config.model_save_dir)
        logger.info('Done.')

    # Define the summary operator (Warning: Won't appear on the tensorboard
    # graph)
    mergedSummaries = tf.merge_all_summaries()
    tf_writer.add_graph(tf_sess.graph)  # Add a Graph into the eventfile.
    train_glob_step = 0
    try:  # If the user exit while training, we still try to save the model
        for e in range(config.train_num_epoch):
            print()
            print("----- Epoch {}/{} ; (lr={}) -----".format(e + 1,
                                                             config.train_num_epoch, config.train_learning_rate))

            batches = batch_data.getBatches()

            tic = strftime("%Y-%m-%d %H:%M:%S", localtime())
            for next_batch in tqdm(batches, desc='Training'):
                ops, feedDict = tf_model.step(next_batch)
                assert len(ops) == 2  # training, loss
                _, loss, summary = tf_sess.run(
                    ops + (mergedSummaries,), feedDict)
                tf_writer.add_summary(summary, train_glob_step)
                train_glob_step += 1  # number of batch iterations.

                # Checkpoint
                if train_glob_step % config.train_save_every == 0:
                    save_session()
            toc = strftime("%Y-%m-%d %H:%M:%S", localtime())

            logger.info("Epoch running from %s to %s" % (tic, toc))

    except (KeyboardInterrupt, SystemExit):
        logger.warn('Interruption detected, exiting the program...')
    save_session()

if __name__ == '__main__':
    tf.app.run()
