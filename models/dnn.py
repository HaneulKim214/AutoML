import tensorflow as tf

def simple_dnn(inputs, dnn_dim, dnn_dropout, active_func):
    """
    Function that builds simple deep neural net model
     
    :param inputs: 
    :param dnn_dim: 
    :param dnn_dropout: 
    :param active_func: 
    :return: 
    """
    flatten_layer = tf.keras.layers.Flatten()(inputs)

    # create hidden layers
    dnn_units = hp.Int(f"0_units", min_value=dnn_units_min, max_value=dnn_units_max)
    dense = tf.keras.layers.Dense(units=dnn_units, activation=active_func)(flatten_layer)
    for layer_i in range(hp.Choice("n_layers", dnn_layers_ss) - 1):
        dnn_units = hp.Int(f"{layer_i}_units", min_value=dnn_units_min, max_value=dnn_units_max)
        dense = tf.keras.layers.Dense(units=dnn_units, activation=active_func)(dense)
        if hp.Boolean("dropout"):
            dense = tf.keras.layers.Dropout(rate=0.25)(dense)
    outputs = tf.keras.layers.Dense(units=10)(dense)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

model_hyp_search_space = {'dnn_units_minmax': [32, 512],
                          'dnn_layers_ss': [1, 2, 3],
                          'dropout_ss': [0.1, 0.2]}
compile_hyp_search_space = {'active_func_ss': ['relu', 'tanh'],
                            'learning_rate_minmax': [1e-4, 1e-1],
                            'optimizer_ss': ['adam']}