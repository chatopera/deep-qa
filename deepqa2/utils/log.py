# coding:utf8
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
Simple Logging Service
'''
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.realpath(__file__))))
from config import read_properties
import logging

config = read_properties()

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_file = '%s/root.log' % config['log']['log_path']
print('Saving logs into', log_file)

fh = logging.FileHandler(log_file)
fh.setFormatter(formatter)
fh.setLevel(config['log']['log_level'])
ch = logging.StreamHandler()
ch.setFormatter(formatter)
ch.setLevel(config['log']['log_level'])


def getLogger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)
    return logger

if __name__ == "__main__":
    logger = getLogger('foo')
    logger.info('bar')
