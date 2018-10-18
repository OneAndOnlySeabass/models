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
    
    def build_rnn_layers(self, inputs, hparams, scope="rnn"):
    """ Builds recurrent layers and the embedding layer.
    
    Args:
      inputs: A Tensor of shape [batch_size, length].
      hparams: Object containing RNN hyperparameters.
      scope: Name of the variable scope.

    Returns:
        A Tensor of shape [batch_size, output_size]
    """
        pass # TODO: set up architecture to build the RNN layers & Embedding layer
        
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