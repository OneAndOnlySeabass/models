# Code written by Sebastiaan Koning (OneAndOnlySeabass), 
# using parts of the TensorFlow AstroNet source code under the Apache License 2.0.
#
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
A recurrent model for classifying light curves.

See the base class (in astro_model.py) for a description of the general
framework of AstroModel and its subclasses.

The architecture of this model is:


                                     predictions
                                          ^
                                          |
                                       logits
                                          ^
                                          |
                                (fully connected layers)
                                          ^
                                          |
                                   pre_logits_concat
                                          ^
                                          |
                                    (concatenate)

              ^                           ^                          ^
              |                           |                          |
   (recurrent blocks 1)         (recurrent blocks 2)    ...          |
              ^                           ^                          |
              |                           |                          |
     time_series_feature_1     time_series_feature_2    ...     aux_features

Code below is under construction.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from astronet.astro_model import astro_model
from astronet.astro_rnn_model import configurations
from astronet.astro_rnn_model import rnn_layer_builder

config = configurations.base()

class AstroRNNModel(astro_model.AstroModel):
    """A model for classifying light curves using a recurrent neural net."""
    def __init__(self, features, labels, hparams, mode):
        """Basic setup. The actual TensorFlow graph is constructed in build().

        Args:
        features: A dictionary containing "time_series_features" and
          "aux_features", each of which is a dictionary of named input Tensors.
          All features have dtype float32 and shape [batch_size, length].
        labels: An int64 Tensor with shape [batch_size]. May be None if mode is
          tf.estimator.ModeKeys.PREDICT.
        hparams: A ConfigDict of hyperparameters for building the model.
        mode: A tf.estimator.ModeKeys to specify whether the graph should be built
            for training, evaluation or prediction.

        Raises:
        ValueError: If mode is invalid.
        """
        super(AstroRNNModel, self).__init__(features, labels, hparams, mode)
    
    def _build_rnn_layers(self, inputs, hparams, scope="rnn"):
        """ Builds recurrent layers and the embedding layer.
    
        Args:
            inputs: A Tensor of shape [batch_size, length].
            hparams: Object containing RNN hyperparameters.
            scope: Name of the variable scope.

        Returns:
            A Tensor of shape [batch_size, output_size].
        
        Author's note on functioning of code:
            At the time of writing this code, the CuDNN GRU layers are not 
            working yet. This is an open TensorFlow issue. 
            GitHub link: https://github.com/tensorflow/tensorflow/issues/20972
        """
        # Set use of CuDNN layers below. Note that GRUCuDNN layers currently do not work with this model.
        use_cudnn_layers = False
        
        with tf.variable_scope(scope):
            if use_cudnn_layers == True:
                net = tf.expand_dims(inputs, -1) # [batch, length, input_dims]
                rnn = rnn_layer_builder.build_cudnn_layers(
                    hparams.rnn_num_layers,
                    hparams.rnn_num_units,
                    hparams.rnn_activation,
                    hparams.rnn_memory_cells,
                    hparams.rnn_direction,
                    hparams.rnn_dropout,
                    scope
                    )
             
                net = rnn(net)
                
                # Flatten.
                net[0].get_shape().assert_has_rank(3)
                net_shape = net[0].get_shape().as_list()
                output_dim = net_shape[1] * net_shape[2]
                net = tf.reshape(net[0], [-1, output_dim], name="flatten")
                
            elif use_cudnn_layers == False:
                net = tf.expand_dims(inputs, -1) # [batch, length, cell_state_size]
                rnn = rnn_layer_builder.build_general_layers(
                    hparams.rnn_num_layers,
                    hparams.rnn_num_units,
                    hparams.rnn_activation,
                    hparams.rnn_memory_cells,
                    hparams.rnn_dropout,
                    hparams.rnn_direction,
                    scope
                    )                    
                if hparams.rnn_direction == "uni":
                    net, output_state = tf.nn.dynamic_rnn(
                        cell=rnn, 
                        inputs=net,
                        dtype=tf.float32, 
                        scope=scope)     
                elif hparams.rnn_direction == "bi":
                    net, os_fw, os_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                        cells_fw=rnn,
                        cells_bw=rnn,
                        inputs=net,
                        dtype=tf.float32,
                        scope=scope)

                else:
                    raise Error("Unrecognized rnn_direction. Use 'uni' or 'bi'.")
                    
                # Flatten.
                net.get_shape().assert_has_rank(3)
                net_shape = net.get_shape().as_list()
                output_dim = net_shape[1] * net_shape[2]
                net = tf.reshape(net, [-1, output_dim], name="flatten")                 
                
            else:
                raise ValueError("Hyperparameter use_cudnn_layers and/or use_MultiRNNCell \
                    is not a boolean.")
        return net
        
    def build_time_series_hidden_layers(self):
        """Builds hidden layers for the time series features.

        Inputs:
        self.time_series_features

        Outputs:
        self.time_series_hidden_layers
        """
        
        time_series_hidden_layers = {}
        for name, time_series in self.time_series_features.items():
            time_series_hidden_layers[name] = self._build_rnn_layers(
                inputs=time_series,
                hparams=self.hparams.time_series_hidden[name],
                scope=name + "_hidden")

        self.time_series_hidden_layers = time_series_hidden_layers