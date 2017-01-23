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

import sys, os
sys.path.append(os.path.join(os.path.dirname(
    os.path.realpath(__file__)), os.pardir))

import random

def singleton(class_):
  instances = {}
  def getinstance(*args, **kwargs):
    if class_ not in instances:
        instances[class_] = class_(*args, **kwargs)
    return instances[class_]
  return getinstance

class Batch:
    """Struct containing batches info
    """
    def __init__(self):
        self.encoderSeqs = []
        self.decoderSeqs = []
        self.targetSeqs = []
        self.weights = []

class BatchData:
    def __init__(self, args):
        self.args = args
        self.dataset = args.dataset
        self.batch_size = args.train_num_batch_size
        self.training_samples = self.dataset['trainingSamples']
        self.training_samples_size = len(self.training_samples)
        self.padToken = self.dataset['word2id']["<pad>"]
        self.goToken = self.dataset['word2id']["<go>"]
        self.eosToken = self.dataset['word2id']["<eos>"]
        self.unknownToken = self.dataset['word2id']["<unknown>"]  # Restore special words

    def _create_batch(self, samples):
        """Create a single batch from the list of sample. The batch size is automatically defined by the number of
        samples given.
        The inputs should already be inverted. The target should already have <go> and <eos>
        Warning: This function should not make direct calls to args.batchSize !!!
        Args:
            samples (list<Obj>): a list of samples, each sample being on the form [input, target]
        Return:
            Batch: a batch object en
        """

        batch = Batch()
        batchSize = len(samples)

        # Create the batch tensor
        for i in range(batchSize):
            # Unpack the sample
            sample = samples[i]
            batch.encoderSeqs.append(list(reversed(sample[0])))  # Reverse inputs (and not outputs), little trick as defined on the original seq2seq paper
            batch.decoderSeqs.append([self.goToken] + sample[1] + [self.eosToken])  # Add the <go> and <eos> tokens
            batch.targetSeqs.append(batch.decoderSeqs[-1][1:])  # Same as decoder, but shifted to the left (ignore the <go>)

            # Long sentences should have been filtered during the dataset creation
            assert len(batch.encoderSeqs[i]) <= self.args.train_max_length_enco
            assert len(batch.decoderSeqs[i]) <= self.args.train_max_length_deco

            # Add padding & define weight
            batch.encoderSeqs[i]   = [self.padToken] * (self.args.train_max_length_enco  - len(batch.encoderSeqs[i])) + batch.encoderSeqs[i]  # Left padding for the input
            batch.weights.append([1.0] * len(batch.targetSeqs[i]) + [0.0] * (self.args.train_max_length_deco - len(batch.targetSeqs[i])))
            batch.decoderSeqs[i] = batch.decoderSeqs[i] + [self.padToken] * (self.args.train_max_length_deco - len(batch.decoderSeqs[i]))
            batch.targetSeqs[i]  = batch.targetSeqs[i]  + [self.padToken] * (self.args.train_max_length_deco - len(batch.targetSeqs[i]))

        # Simple hack to reshape the batch
        encoderSeqsT = []  # Corrected orientation
        for i in range(self.args.train_max_length_enco):
            encoderSeqT = []
            for j in range(batchSize):
                encoderSeqT.append(batch.encoderSeqs[j][i])
            encoderSeqsT.append(encoderSeqT)
        batch.encoderSeqs = encoderSeqsT

        decoderSeqsT = []
        targetSeqsT = []
        weightsT = []
        for i in range(self.args.train_max_length_deco):
            decoderSeqT = []
            targetSeqT = []
            weightT = []
            for j in range(batchSize):
                decoderSeqT.append(batch.decoderSeqs[j][i])
                targetSeqT.append(batch.targetSeqs[j][i])
                weightT.append(batch.weights[j][i])
            decoderSeqsT.append(decoderSeqT)
            targetSeqsT.append(targetSeqT)
            weightsT.append(weightT)
        batch.decoderSeqs = decoderSeqsT
        batch.targetSeqs = targetSeqsT
        batch.weights = weightsT

        # # Debug
        # self.printBatch(batch)  # Input inverted, padding should be correct
        # print(self.sequence2str(samples[0][0]))
        # print(self.sequence2str(samples[0][1]))  # Check we did not modified the original sample

        return batch
    
    def next(self):
        print('Shuffling the dataset')
        random.shuffle(self.training_samples)
        batches = []

        def gen_next_samples():
            """ Generator over the mini-batch training samples
            """
            for i in range(0, self.training_samples_size, self.batch_size):
                yield self.training_samples[i:min(i + self.batch_size, self.training_samples_size)]

        for samples in gen_next_samples():
            batch = self._create_batch(samples)
            batches.append(batch)
            
        return batches

def main():
    pass

if __name__ == '__main__':
    main()