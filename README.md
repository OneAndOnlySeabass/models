# TensorFlow Models fork for building RNN variants of AstroNet
A project by Sebastiaan Koning (OneAndOnlySeabass)

This repository was forked to build RNN, LSTM and GRU variants of AstroNet, as well as re-use the AstroNet FC and CNN code for experiments. These experiments were conducted as part of my master thesis "Comparing convolutional neural networks and recurrent neural networks for exoplanet detection" for Data Science: Business and Governance at Tilburg University. Once my thesis is finalized, I will add it to this repository. This repository will remain a permanent fork as I keep only the AstroNet model here and not any of the other models.

The added RNN model has received a dedicated directory within the astronet directory and works in largely the same way as the other AstroNet model types. All configurations can be set in the configurations.py file, except for the use of CuDNN RNNs, which is set in astro_rnn_model.py under the function "build_rnn_layers". Most model setups have been successfully tested, with the notable exception of a CuDNN GRU, which does not work with the Estimator class model generated in AstroNet due to an unresolved issue in the TensorFlow source code. You can follow this issue here: https://github.com/tensorflow/tensorflow/issues/20972 

Users are able to adjust the following RNN hyperparameters:
- use_cudnn_layers: Whether to use a CuDNN implementation. Only possible to use if a GPU is present in your machine.
- rnn_num_layers: Number of RNN layers to use.
- rnn_num_units: Number of units per layer.
- rnn_memory_cells: Indicates which memory cell to use. None for standard RNN, 'lstm' or 'gru'.
- rnn_activation: Which activation function to use. Default is tanh, which is the only option for the CuDNN LSTM. For the CuDNN RNN, you can                  use either tanh or relu. For non-CuDNN RNNs, all supported TensorFlow activation functions can be used.
- rnn_dropout: Amount of dropout to apply to each RNN layer. Use 0.0 for no dropout.
- rnn_direction: Indicates if the network should be unidirectional ('uni') or bidirectional ('bi).

Finally, I would like to make a note regarding code optimization. As my focus during this project was mostly on correct implementation rather than code optimization, parts of this code might not be well optimized, especially in the non-CuDNN RNNs. Not only in the network architectures itself, but also in the way inputs are fed to the network and (intermediate) outputs are processed. If you are able to further optimize the code, I encourage you to do so and make a pull request outlining your optimizations made.

## License
The original AstroNet source code has been used and modified in my project for scientific purposes only under the Apache License 2.0. You can obtain this license using the link below.

[Apache License 2.0](LICENSE)

Link to original AstroNet source code: https://github.com/google-research/exoplanet-ml 

## Contact
If you would like to contact me regarding my project, you can e-mail me at (first name).(last name) AT hetnet.nl
