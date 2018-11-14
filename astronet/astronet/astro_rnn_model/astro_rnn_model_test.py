# Code written by Sebastiaan Koning (OneAndOnlySeabass), 
# using parts of the TensorFlow AstroNet source code under the Apache License 2.0.
#
# Copyright 2018 The TensorFlow Authors.
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

"""
Tests for astro_rnn_model.AstroRNNModel.
Code finished in concept.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from astronet.astro_rnn_model import astro_rnn_model
from astronet.astro_rnn_model import configurations
from astronet.ops import input_ops
from astronet.ops import testing
from astronet.util import configdict

class AstroRNNModelTest(tf.test.TestCase):
  
  def assertShapeEquals(self, shape, tensor_or_array):
    """Asserts that a Tensor or Numpy array has the expected shape.

    Args:
      shape: Numpy array or anything that can be converted to one.
      tensor_or_array: tf.Tensor, tf.Variable, Numpy array or anything that can
          be converted to one.
    """
    if isinstance(tensor_or_array, (np.ndarray, np.generic)):
      self.assertAllEqual(shape, tensor_or_array.shape)
    elif isinstance(tensor_or_array, (tf.Tensor, tf.Variable)):
      self.assertAllEqual(shape, tensor_or_array.shape.as_list())
    else:
      raise TypeError("tensor_or_array must be a Tensor or Numpy ndarray")
  
  def testOneTimeSeriesFeature(self):
    # Build config.
    feature_spec = {
        "time_feature_1": {
            "length": 20,
            "is_time_series": True,
        }
    }
    hidden_spec = {
        "time_feature_1": {
        "rnn_num_layers": 2,
        "rnn_num_units": 16,
        "rnn_memory_cells": "lstm",
        "rnn_activation": "tanh",
        "rnn_dropout": 0.0,
        "rnn_direction": "bi"
        }
    }
    config = configurations.base()
    config["inputs"]["features"] = feature_spec
    config["hparams"]["time_series_hidden"] = hidden_spec
    config = configdict.ConfigDict(config)
    
    # Build model.
    features = input_ops.build_feature_placeholders(config.inputs.features)
    labels = input_ops.build_labels_placeholder()
    model = astro_rnn_model.AstroRNNModel(features, labels, config.hparams,
                                          tf.estimator.ModeKeys.TRAIN)
    model.build()
    
    # Execute the TensorFlow graph.
    scaffold = tf.train.Scaffold()
    scaffold.finalize()
    with self.test_session() as sess:
      sess.run([scaffold.init_op, scaffold.local_init_op])
      step = sess.run(model.global_step)
      self.assertEqual(0, step)

      # Fetch predictions.
      features = testing.fake_features(feature_spec, batch_size=16)
      labels = testing.fake_labels(config.hparams.output_dim, batch_size=16)
      feed_dict = input_ops.prepare_feed_dict(model, features, labels)
      predictions = sess.run(model.predictions, feed_dict=feed_dict)
      self.assertShapeEquals((16, 1), predictions)
    
  def testTwoTimeSeriesFeatures(self):
    # Build config.
    feature_spec = {
        "time_feature_1": {
            "length": 20,
            "is_time_series": True,
        },
        "time_feature_2": {
            "length": 5,
            "is_time_series": True,
        },
        "aux_feature_1": {
            "length": 1,
            "is_time_series": False,
        },
    }
    hidden_spec = {
        "time_feature_1": {
            "rnn_num_layers": 2,
            "rnn_num_units": 16,
            "rnn_memory_cells": "lstm",
            "rnn_activation": "tanh",
            "rnn_dropout": 0.0,
            "rnn_direction": "bi"
        },
        "time_feature_2": {
            "rnn_num_layers": 1,
            "rnn_num_units": 4,
            "rnn_memory_cells": "lstm",
            "rnn_activation": "tanh",
            "rnn_dropout": 0.0,
            "rnn_direction": "bi"
        }
    }
    config = configurations.base()
    config["inputs"]["features"] = feature_spec
    config["hparams"]["time_series_hidden"] = hidden_spec
    config = configdict.ConfigDict(config)
    
    # Build model
    features = input_ops.build_feature_placeholders(config.inputs.features)
    labels = input_ops.build_labels_placeholder()
    model = astro_rnn_model.AstroRNNModel(features, labels, config.hparams,
                                          tf.estimator.ModeKeys.TRAIN)
    model.build()
    
    # Execute the TensorFlow graph.
    scaffold = tf.train.Scaffold()
    scaffold.finalize()
    with self.test_session() as sess:
      sess.run([scaffold.init_op, scaffold.local_init_op])
      step = sess.run(model.global_step)
      self.assertEqual(0, step)

      # Fetch predictions.
      features = testing.fake_features(feature_spec, batch_size=16)
      labels = testing.fake_labels(config.hparams.output_dim, batch_size=16)
      feed_dict = input_ops.prepare_feed_dict(model, features, labels)
      predictions = sess.run(model.predictions, feed_dict=feed_dict)
      self.assertShapeEquals((16, 1), predictions)

    
if __name__ == "__main__":
  tf.test.main()
