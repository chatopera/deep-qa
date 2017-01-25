#! /bin/bash 
###########################################
# Start TensorFlow Board
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
tf_board_port=6006
logdir_root=$baseDir/../save
run_prefix=chatbot
logdir=''

# functions
function generate_logdir(){
    cd $logdir_root
    for x in `ls`;do 
        logdir=$run_prefix$x:$logdir_root/$x,$logdir
    done;
    logdir=${logdir::-1}
    echo "watching " $logdir
}

function start_tensorboard(){
    tensorboard --logdir=$logdir --port $tf_board_port
}

# main 
[ -z "${BASH_SOURCE[0]}" -o "${BASH_SOURCE[0]}" = "$0" ] || return
generate_logdir
sleep 1
start_tensorboard
