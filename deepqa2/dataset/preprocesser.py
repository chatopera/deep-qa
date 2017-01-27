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
from config import read_properties
from utils import log
logger = log.getLogger(__name__)
config = read_properties()


def main():
    logger.info('Corpus Name %s' % config['corpus']['corpus_name'])
    train_max_length = int(config['corpus']['corpus_max_length'])
    train_max_length_enco = train_max_length
    train_max_length_deco = train_max_length + 2
    print('Max Length %d' % train_max_length)
    td = TextData(munchify({'rootDir': config['rootDir'],
                            'corpus': config['corpus']['corpus_name'],
                            'maxLength': train_max_length,
                            'maxLengthEnco': train_max_length_enco,
                            'maxLengthDeco': train_max_length_deco,
                            'datasetTag': ''}))

if __name__ == '__main__':
    main()
