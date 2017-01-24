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
from config import config
from tqdm import tqdm
from utils import log
from models.rnn import Model
from utils.helper import BatchData
from time import localtime, strftime

logger = log.getLogger(__name__)

def main(unused_argv):
    batch_data = BatchData(config)
    with tf.device(None):
        tf_model = Model(config, config.dataset)
    tf_writer = tf.train.SummaryWriter(config.model_save_dir)
    tf_saver = tf.train.Saver(max_to_keep=200) # Arbitrary limit
    tf_sess = tf.Session()
    tf_sess.run(tf.initialize_all_variables())

    def save_session():
        """ 
        Save the model parameters and the variables
        """
        tqdm.write('Checkpoint reached: saving model (don\'t stop the run)...')
        # TODO Save config as parameters.
        # Save the model parameters and the variables
        tf_saver.save(tf_sess, config.model_save_ckpt)
        logger.info('Copy dataset ...')
        shutil.copy(config.dataset_pkl_path, config.model_save_dir)
        logger.info('Save config.ini ...')
        shutil.copy(config.config_ini_path, config.model_save_dir)
        logger.info('Done.')

    mergedSummaries = tf.merge_all_summaries() # Define the summary operator (Warning: Won't appear on the tensorboard graph)
    tf_writer.add_graph(tf_sess.graph) # Add a Graph into the eventfile.
    train_glob_step = 0
    try: # If the user exit while training, we still try to save the model
        for e in range(config.train_num_epoch):
            print()
            print("----- Epoch {}/{} ; (lr={}) -----".format(e+1, config.train_num_epoch, config.train_learning_rate))

            batches = batch_data.next()

            tic = strftime("%Y-%m-%d %H:%M:%S", localtime())
            for next_batch in tqdm(batches, desc='Training'):
                ops, feedDict = tf_model.step(next_batch)
                assert len(ops) == 2 # training, loss
                _, loss, summary = tf_sess.run(ops + (mergedSummaries,), feedDict)
                tf_writer.add_summary(summary, train_glob_step)
                train_glob_step += 1 # number of batch iterations.

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