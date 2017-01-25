#! /bin/bash 
###########################################
# Start Training
# http://eng.snaplingo.net/how-to-use-tensorboard/
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

# constants
baseDir=$(cd `dirname "$0"`;pwd)
# functions
function killOtherTrains(){
    echo "Kill other trains ..."
    cd /tmp
    for x in `ps -ef|grep train.py|awk '{ print $2 }'`; do
        ps -p $x 2>&1 >>/dev/null
        if [ $? == 0 ];then
            sudo kill -9 $x 2>&1 >>/dev/null ;
        fi
    done
    echo "Start to train ..."
    cd $baseDir/..
    nohup python deepqa2/train.py &
}

function confirm(){
    ps -ef|grep deepqa2/train.py
    while true; do
        read -p "Kill other trains? " yn
        case $yn in
            [Yy]* ) killOtherTrains; break;;
            [Nn]* ) exit;;
            * ) echo "Please answer yes/y/Y or no/n/N.";;
        esac
    done
}
# main 
[ -z "${BASH_SOURCE[0]}" -o "${BASH_SOURCE[0]}" = "$0" ] || return
confirm