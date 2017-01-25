# coding: utf8
# Copyright 2017 Hai Liang Wang. All Rights Reserved.
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
import os
import sys
sys.path.append(os.path.join(os.path.dirname(
    os.path.realpath(__file__)), os.pardir))
import configparser
from dataset.textdata import TextData
from munch import munchify

CONF_DIR = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), os.pardir)


def get_cfg_path():
    '''
    Get cfg path
    '''
    return os.path.join(CONF_DIR, 'config.ini')


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


def main():
    config = load_config()
    print('Corpus Name %s' % config.get('corpus', 'corpus_name'))
    train_max_length = config.getint('corpus', 'corpus_max_length')
    train_max_length_enco = train_max_length
    train_max_length_deco = train_max_length + 2
    print('Max Length %d' % train_max_length)
    td = TextData(munchify({'rootDir': config.get('corpus', 'corpus_path'),
                            'corpus': config.get('corpus', 'corpus_name'),
                            'maxLength': train_max_length,
                            'maxLengthEnco': train_max_length_enco,
                            'maxLengthDeco': train_max_length_deco,
                            'datasetTag': ''}))

if __name__ == '__main__':
    main()
