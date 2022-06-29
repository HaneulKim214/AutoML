"""
For storing constant variables
"""

model_hyp_search_space = {'dnn_units_minmax': [32, 512],
                          'dnn_layers_ss': [1, 2, 3],
                          'dropout_ss': [0.1, 0.2]}
compile_hyp_search_space = {'active_func_ss': ['relu', 'tanh'],
                            'learning_rate_minmax': [1e-4, 1e-1],
                            'optimizer_ss': ['adam']}