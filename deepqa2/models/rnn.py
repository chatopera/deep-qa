# Copyright 2015 Conchylicultor. All Rights Reserved.
# Modifications copyright (C) 2016 Carlos Segura
# Modifications copyright (C) 2017 Hai Liang Wang
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

"""
Model to predict the next sentence given an input sequence

"""
import sys
import os
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(
    os.path.realpath(__file__)), os.pardir))

from dataset.textdata import Batch
from utils import log

logger = log.getLogger(__name__)


class ProjectionOp:
    """ Single layer perceptron
    Project input tensor on the output dimension
    """

    def __init__(self, shape, scope=None, dtype=None):
        """
        Args:
            shape: a tuple (input dim, output dim)
            scope (str): encapsulate variables
            dtype: the weights type
        """
        assert len(shape) == 2

        self.scope = scope

        # Projection on the keyboard
        with tf.variable_scope('weights_' + self.scope):
            self.W = tf.get_variable(
                'weights',
                shape,
                # initializer=tf.truncated_normal_initializer()  # TODO: Tune
                # value (fct of input size: 1/sqrt(input_dim))
                dtype=dtype
            )
            self.b = tf.get_variable(
                'bias',
                shape[1],
                initializer=tf.constant_initializer(),
                dtype=dtype
            )

    def getWeights(self):
        """ Convenience method for some tf arguments
        """
        return self.W, self.b

    def __call__(self, X):
        """ Project the output of the decoder into the vocabulary space
        Args:
            X (tf.Tensor): input value
        """
        with tf.name_scope(self.scope):
            return tf.matmul(X, self.W) + self.b


class Model:
    """
    Implementation of a seq2seq model.
    Architecture:
        Encoder/decoder
        2 LTSM layers
    """

    def __init__(self, args, dataset, is_serve=False):
        """
        Args:
            args: parameters of the model
            dataset: the dataset object
        """
        logger.info("Model implemented seq2seq on creation...")

        self.dataset = dataset  # Keep a reference on the dataset
        self.args = args  # Keep track of the parameters of the model
        self.dtype = tf.float32
        self.is_serve = is_serve

        # Placeholders
        self.encoderInputs = None
        self.decoderInputs = None  # Same that decoderTarget plus the <go>
        self.decoderTargets = None
        self.decoderWeights = None  # Adjust the learning to the target sentence size

        # Main operators
        self.lossFct = None
        self.optOp = None
        self.outputs = None  # Outputs of the network, list of probability for each words

        # Construct the graphs
        self.buildNetwork()

    def buildNetwork(self):
        """ Create the computational graph
        """

        # TODO: Create name_scopes (for better graph visualisation)
        # TODO: Use buckets (better perfs)

        # Parameters of sampled softmax (needed for attention mechanism and a
        # large vocabulary size)
        outputProjection = None
        # Sampled softmax only makes sense if we sample less than vocabulary
        # size.
        if 0 < self.args.train_softmax_samples < len(self.dataset['word2id']):
            outputProjection = ProjectionOp(
                (self.args.train_hidden_size, len(self.dataset['word2id'])),
                scope='softmax_projection',
                dtype=self.dtype
            )

            def sampledSoftmax(inputs, labels):
                # Add one dimension (nb of true classes, here 1)
                labels = tf.reshape(labels, [-1, 1])

                # We need to compute the sampled_softmax_loss using 32bit floats to
                # avoid numerical instabilities.
                localWt = tf.cast(tf.transpose(outputProjection.W), tf.float32)
                localB = tf.cast(outputProjection.b,               tf.float32)
                localInputs = tf.cast(inputs,
                                      tf.float32)

                return tf.cast(
                    tf.nn.sampled_softmax_loss(
                        localWt,  # Should have shape [num_classes, dim]
                        localB,
                        localInputs,
                        labels,
                        # The number of classes to randomly sample per batch
                        self.args.train_softmax_samples,
                        len(self.dataset['word2id'])),  # The number of classes
                    self.dtype)

        # Creation of the rnn cell
        encoDecoCell = tf.nn.rnn_cell.BasicLSTMCell(
            self.args.train_hidden_size, state_is_tuple=True)  # Or GRUCell, LSTMCell(args.hiddenSize)
        # encoDecoCell = tf.nn.rnn_cell.DropoutWrapper(encoDecoCell,
        # input_keep_prob=1.0, output_keep_prob=1.0)  # TODO: Custom values
        # (WARNING: No dropout when testing !!!)
        encoDecoCell = tf.nn.rnn_cell.MultiRNNCell(
            [encoDecoCell] * self.args.train_num_layers, state_is_tuple=True)

        # Network input (placeholders)

        with tf.name_scope('placeholder_encoder'):
            self.encoderInputs = [tf.placeholder(tf.int32,   [None, ]) for _ in range(
                self.args.train_max_length_enco)]  # Batch size * sequence length * input dim

        with tf.name_scope('placeholder_decoder'):
            self.decoderInputs = [tf.placeholder(tf.int32,   [None, ], name='inputs') for _ in range(
                self.args.train_max_length_deco)]  # Same sentence length for input and output (Right ?)
            self.decoderTargets = [tf.placeholder(
                tf.int32,   [None, ], name='targets') for _ in range(self.args.train_max_length_deco)]
            self.decoderWeights = [tf.placeholder(
                tf.float32, [None, ], name='weights') for _ in range(self.args.train_max_length_deco)]

        # Define the network
        # Here we use an embedding model, it takes integer as input and convert them into word vector for
        # better word representation
        decoderOutputs, states = tf.nn.seq2seq.embedding_rnn_seq2seq(
            # List<[batch=?, inputDim=1]>, list of size args.maxLength
            self.encoderInputs,
            # For training, we force the correct output (feed_previous=False)
            self.decoderInputs,
            encoDecoCell,
            len(self.dataset['word2id']),
            # Both encoder and decoder have the same number of class
            len(self.dataset['word2id']),
            embedding_size=self.args.train_num_embedding,  # Dimension of each word
            output_projection=outputProjection.getWeights() if outputProjection else None,
            # When we serve, we use previous output as next
            # input (feed_previous)
            feed_previous=self.is_serve
        )

        if self.is_serve:
            if not outputProjection:
                self.outputs = decoderOutputs
            else:
                self.outputs = [outputProjection(
                    output) for output in decoderOutputs]
            # TODO: Attach a summary to visualize the output
        else:
            # For training only
            # Finally, we define the loss function
            self.lossFct = tf.nn.seq2seq.sequence_loss(
                decoderOutputs,
                self.decoderTargets,
                self.decoderWeights,
                len(self.dataset['word2id']),
                # If None, use default SoftMax
                softmax_loss_function=sampledSoftmax if outputProjection else None
            )
            tf.scalar_summary('loss', self.lossFct)  # Keep track of the cost

            # Initialize the optimizer
            opt = tf.train.AdamOptimizer(
                learning_rate=self.args.train_learning_rate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-08
            )
            self.optOp = opt.minimize(self.lossFct)

    def step(self, batch):
        """ Forward/training step operation.
        Does not perform run on itself but just return the operators to do so. Those have then to be run
        Args:
            batch (Batch): Input data on testing mode, input and target on output mode
        Return:
            (ops), dict: A tuple of the (training, loss) operators or (outputs,) in testing mode with the associated feed dictionary
        """

        # Feed the dictionary
        feedDict = {}
        ops = None

        if not self.is_serve: # Training
            for i in range(self.args.train_max_length_enco):
                feedDict[self.encoderInputs[i]] = batch.encoderSeqs[i]
            for i in range(self.args.train_max_length_deco):
                feedDict[self.decoderInputs[i]] = batch.decoderSeqs[i]
                feedDict[self.decoderTargets[i]] = batch.targetSeqs[i]
                feedDict[self.decoderWeights[i]] = batch.weights[i]

            ops = (self.optOp, self.lossFct)
        else:  # Serve (batchSize == 1)
            for i in range(self.args.train_max_length_enco):
                feedDict[self.encoderInputs[i]]  = batch.encoderSeqs[i]
            feedDict[self.decoderInputs[0]]  = [self.dataset['word2id']['<go>']]

            ops = (self.outputs,)

        # Return one pass operator
        return ops, feedDict
