# Code written by Sebastiaan Koning (OneAndOnlySeabass)
"""
Code used to build the RNN layers in CuDNN form or non-CuDNN form.

Code under construction. 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def build_general_layers(layers, units, activation, memory_cell, dropout, name):
    """ 
    Builds layers not optimized for CuDNN if use_cudnn_layers is False
    and use_MultiRNNCell is True in astro_rnn_model.py.
        
    Args:
        layers: Number of layers to create
        units: number of units in the layer
        activation: Activation function to use.
        memory_cell: Indicate type of memory cell to use. Use None for basic
        RNN, or use "lstm" or "gru".
        dropout: Amount of dropout to apply to each layer. Use 0.0 for no dropout.
        name: Name to give to each layer.
    Returns:
        A MultiRNNCell according to given args.  
    """
    if memory_cell is None:
        cells = []
        for n in range(layers):
            cell = tf.contrib.rnn.BasicRNNCell(
                units, 
                activation=activation,
                name=(name+str(n))
                )
            if dropout > 0.0:
                keep = 1 - dropout
                cell = tf.contrib.rnn.DropoutWrapper(cell,
                    input_keep_prob=keep,
                    output_keep_prob=keep,
                    state_keep_prob=keep
                    )                
            cells.append(cell)
        
    elif memory_cell == "lstm":
        cells = []
        for n in range(layers):       
            cell = tf.nn.rnn_cell.LSTMCell(
                units, 
                activation=activation,
                name=(name+str(n))
                )
            if dropout > 0.0:
                keep = 1 - dropout
                cell = tf.contrib.rnn.DropoutWrapper(cell,
                    input_keep_prob=keep,
                    output_keep_prob=keep,
                    state_keep_prob=keep
                    )
            cells.append(cell)
            
    elif memory_cell == "gru":
        cells = []
        for n in range(layers):       
            cell = tf.contrib.rnn.GRUCell(
                units, 
                activation=activation,
                name=(name+str(n))
                )
            if dropout > 0.0:
                keep = 1 - dropout
                cell = tf.contrib.rnn.DropoutWrapper(cell,
                    input_keep_prob=keep,
                    output_keep_prob=keep,
                    state_keep_prob=keep
                    )
            cells.append(cell)
    else:
        raise Error("Invalid memory_cell type. Allowed: None, 'lstm' or 'gru'.")
    
    rnn_net = tf.contrib.rnn.MultiRNNCell(cells)
    
    return rnn_net
    
def build_cudnn_layers(layers, units, activation, memory_cell, direction, dropout, name):
    """ 
    Builds layers optimized for CuDNN if hparams.use_cudnn_layers is True.
        
    Args:
        layers: The number of layers.
        units: The number of units per layer.
        activation: Activation function to use. "tanh" or "relu".
        memory_cell: Indicate type of memory cell to use. Use None for basic
        RNN, or use "lstm" or "gru".
        direction: Unidirectional or bidirectional.
        dropout: The amount of dropout to apply to the model. Use 0.0 for no dropout.
        name: The name of the model
    Returns:
        A collection of CuDNN optimized layers according to given args.  
    """
    # The assert statement below can be removed if the issue around CuDNNGRU is resolved
    assert memory_cell != "gru", ("GRUs cannot be used in CuDNN layers right now, see documentation. \
                                    Use the non-CuDNN implementation or use LSTM.")
    
    if activation == "relu" and memory_cell in ["lstm", "gru"]:
        raise Error("ReLu activation cannot be used in combination \
                    with LSTM.")
        
    if memory_cell is None:
        if activation == "tanh":
            if direction == "uni":
                cudnn_net = tf.contrib.cudnn_rnn.CudnnRNNTanh(
                    layers,
                    units,
                    direction='unidirectional',
                    dropout=dropout,
                    name=name
                    )
                    
            elif direction == "bi":
                cudnn_net = tf.contrib.cudnn_rnn.CudnnRNNTanh(
                    layers,
                    units,
                    direction='bidirectional',
                    dropout=dropout,
                    name=name
                    )
                    
            else:
                raise ValueError("Unrecognized direction. Use uni or bi")
                    
        elif activation == "relu":
            if direction == "uni":
                cudnn_net = tf.contrib.cudnn_rnn.CudnnRNNRelu(
                    layers,
                    units,
                    direction='unidirectional',
                    dropout=dropout,
                    name=name
                    )
                    
            elif direction == "bi":
                cudnn_net = tf.contrib.cudnn_rnn.CudnnRNNRelu(
                    layers,
                    units,
                    direction='bidirectional',
                    dropout=dropout,
                    name=name
                    )
            else:
                raise ValueError("Unrecognized direction. Use uni or bi")
                    
        else:
            raise ValueError("Unrecognized activation function. Use relu or tanh.")
            
    elif memory_cell == "lstm":
        if direction == "uni":
            cudnn_net = tf.contrib.cudnn_rnn.CudnnLSTM(
                        layers,
                        units,
                        direction='unidirectional',
                        dropout=dropout,
                        name=name
                        )
        
        elif direction == "bi":
            cudnn_net = tf.contrib.cudnn_rnn.CudnnLSTM(
                        layers,
                        units,
                        direction='bidirectional',
                        dropout=dropout,
                        name=name
                        )
        
        else:
            raise ValueError("Unrecognized activation function. Use relu or tanh.")
    
    # Commented this part of the code as CuDNNGRU does not work with estimators.
    # Can be uncommented if this issue is fixed.
    #elif memory_cell == "gru":
    #    if direction == "uni":
    #        cudnn_net = tf.contrib.cudnn_rnn.CudnnGRU(
    #                layers,
    #                units,
    #                direction='unidirectional',
    #                dropout=dropout,
    #                name=name
    #                )
    #
    #    elif direction == "bi":
    #        cudnn_net = tf.contrib.cudnn_rnn.CudnnGRU(
    #                layers,
    #                units,
    #                direction='bidirectional',
    #                dropout=dropout,
    #                name=name
    #                )
    #
    #    else:
    #        raise ValueError("Unrecognized activation function. Use relu or tanh.")
                                
    else:
        raise Error("Unrecognized memory cell. Use None or 'lstm'")
    return cudnn_net