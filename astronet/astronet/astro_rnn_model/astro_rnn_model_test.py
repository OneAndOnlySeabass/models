# Code written by Sebastiaan Koning (OneAndOnlySeabass), 
# using parts of the TensorFlow AstroNet source code under the Apache License 2.0.
#
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Tests for astro_rnn_model.AstroRNNModel.
Code will be provided in a later stadium
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
    pass # TODO: build the testing structures
  
  def testTwoTimeSeriesFeatures(self):
    pass # TODO: build the testing structures