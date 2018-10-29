# Code written by Sebastiaan Koning (OneAndOnlySeabass), 
# using parts of the TensorFlow AstroNet source code under the Apache License 2.0.
#
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Configurations for astro_rnn_model building, training and evaluation.

Available configurations:
  * base: One time series feature per input example. Default is "global_view".
  * local_global: Two time series features per input example.
      - A "global" view of the entire orbital period.
      - A "local" zoomed-in view of the transit event.

Code finished in draft. Not tested yet!
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from astronet.astro_model import configurations as parent_configs

def base():
  """Base configuration for a CNN model with a single global view."""
  config = parent_configs.base()
  
  # Add configuration for the recurrent layers of the global_view feature.
  config["hparams"]["time_series_hidden"] = {
      "global_view": {
          "rnn_num_layers": 3,
          "rnn_num_units": 128,
          "rnn_memory_cells": None, # None for standard RNN or "lstm" or "gru"
          "rnn_activation": "tanh" # "tanh" or "relu". ReLU not available for CuDNN LSTM/GRU.
          "dropout": 0.0,
          "bidirectional": False
      },
  }
  config["hparams"]["num_pre_logits_hidden_layers"] = 2
  config["hparams"]["pre_logits_hidden_layer_size"] = 128
  config["hparams"]["use_cudnn_layers"] = True # Only set to True if GPU is present
  return config
  
def local_global():
  """Base configuration for a CNN model with separate local/global views."""
  config = parent_configs.base()

  # Override the model features to be local_view and global_view time series.
  config["inputs"]["features"] = {
      "local_view": {
          "length": 201,
          "is_time_series": True,
      },
      "global_view": {
          "length": 2001,
          "is_time_series": True,
      },
  }
  # Add configuration for the recurrent layers of the local_view feature.
  config["hparams"]["time_series_hidden"] = {
      "local_view": {
          "rnn_num_layers": 2,
          "rnn_num_units": 128,
          "rnn_memory_cells": None, # None for standard RNN or "lstm" or "gru"
          "rnn_activation": "tanh" # "tanh" or "relu". ReLU not available for CuDNN LSTM/GRU.
          "dropout": 0.0,
          "bidirectional": False
      },
  }
  # Add configuration for the recurrent layers of the global_view feature.
  config["hparams"]["time_series_hidden"] = {
      "global_view": {
          "rnn_num_layers": 3,
          "rnn_num_units": 128,
          "rnn_memory_cells": None, # None for standard RNN or "lstm" or "gru"
          "rnn_activation": "tanh" # "tanh" or "relu". ReLU not available for CuDNN LSTM/GRU.
          "dropout": 0.0,
          "bidirectional": False
      },
  }
  config["hparams"]["num_pre_logits_hidden_layers"] = 2
  config["hparams"]["pre_logits_hidden_layer_size"] = 128
  config["hparams"]["use_cudnn_layers"] = True # Only set to True if GPU is present
  return config
  