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
config module to load configurations
'''

import os
import socket
import pickle
import configparser
from time import localtime, strftime
from utils.helper import singleton

CONF_DIR = os.path.dirname(os.path.abspath(__file__))

def get_cfg_dir():
    '''
    Get cfg dir
    '''
    if not os.path.exists(CONF_DIR):
        os.mkdir(CONF_DIR)
    return CONF_DIR


def get_cfg_path():
    '''
    Get cfg path
    '''
    return os.path.join(get_cfg_dir(), 'config.ini')


def load_config():
    '''
    Load configurations
    '''
    cf = get_cfg_path()
    if not os.path.exists(cf):
        f = open(cf, 'w')
        f.close()

    config = configparser.ConfigParser()
    config.read(cf)
    return config


def read_properties():
    '''
    Read Properties from Config File.
    '''
    config = load_config()
    secs = config.sections()
    conf = {}
    for x in secs:
        conf[x] = {y: config.get(x, y) for y in config.options(x)}
    conf['data'] = {'save': CONF_DIR + '/save',
                    'dataset': CONF_DIR + '/data/dataset.pkl'}
    conf['log']['log_path'] = CONF_DIR + '/logs'
    return conf

@singleton
class Config:
    '''
    Load All Parameters in one place.
    '''
    def __init__(self):
        self.ini = read_properties()
        '''
        Define project params
        '''
        self.config_ini_path = get_cfg_path()
        self.project_dir = os.getcwd()
        self.model_dir = os.path.join(self.project_dir, 'model')
        self.model_save_tag = "deeplearning.cobra.%s.%s" % (socket.gethostname(), strftime("%Y%m%d.%H%M%S", localtime()))
        self.model_save_dir = os.path.join(self.project_dir, 'save/' + self.model_save_tag)
        self.model_save_ckpt = os.path.join(self.model_save_dir, 'model.ckpt')

        '''
        Define Dataset
        '''
        if not os.path.exists(self.ini['data']['dataset']):
            raise 'Corpus Data not exists.'
        print('Start to load corpus ...')
        with open(self.ini['data']['dataset'], 'rb') as handle:
            self.dataset = pickle.load(handle)

        self.dataset_pkl_path = self.ini['data']['dataset']
        self.dataset_word2id = self.dataset["word2id"]
        self.dataset_id2word = self.dataset["id2word"]
        self.dataset_trainingSamples = self.dataset["trainingSamples"]

        self.dataset_padToken = self.dataset_word2id["<pad>"]
        self.dataset_goToken = self.dataset_word2id["<go>"]
        self.dataset_eosToken = self.dataset_word2id["<eos>"]
        self.dataset_unknownToken = self.dataset_word2id["<unknown>"]  # Restore special words

        print('>> dataset word2id size: %d' % len(self.dataset_word2id.keys()))
        print('>> dataset id2word size: %d' % len(self.dataset_id2word.keys()))
        print('>> dataset training samples size: %d' % len(self.dataset_trainingSamples))
        print('>> dataset training max length: %d' % self.dataset["maxLength"])

        '''
        Define hyper parameters for model training.
        '''
        # Epoch training runs    
        self.train_num_epoch = int(self.ini['hyparams']['train_num_epoch'])
        # number of rnn layers
        self.train_num_layers = int(self.ini['hyparams']['train_num_layers']) 
        # batch size
        self.train_num_batch_size = int(self.ini['hyparams']['train_num_batch_size'])
        # embedding size
        self.train_num_embedding = int(self.ini['hyparams']['train_num_embedding'])
        # number of hidden units of RNN Cell
        self.train_hidden_size = int(self.ini['hyparams']['train_hidden_size'])
        # softmax samples
        self.train_softmax_samples = int(self.ini['hyparams']['train_softmax_samples'])
        # TODO is watson mode, what is it, config from config.ini 
        self.train_is_watson_mode = False
        # Save every N steps 
        self.train_save_every = int(self.ini['hyparams']['train_save_every'])
        # Trained Max Length
        self.train_max_length = self.dataset["maxLength"]
        # For now, not arbitrary  independent maxLength between encoder and decoder
        self.train_max_length_enco = self.dataset["maxLength"]
        self.train_max_length_deco = self.dataset["maxLength"] + 2
        self.train_learning_rate = float(self.ini['hyparams']['train_learning_rate'])

config = Config()

if __name__ == "__main__":
    conf = read_properties()
    # for x in conf['rule']['blacklist']:
    #     print x
    print(conf)