import keras_tuner as kt
import tensorflow as tf


hyp_search_space = {'dnn_units_minmax': [32, 512],
                    'dnn_layers_ss': [1, 2, 3],
                    'dropout_ss': [0.1, 0.2],
                    'active_func_ss': ['relu', 'tanh'],
                    'learning_rate_minmax': [1e-4, 1e-1],
                    'optimizer_ss': ['adam']}
dnn_layers_ss = hyp_search_space['dnn_layers_ss']
dnn_units_min, dnn_units_max = hyp_search_space['dnn_units_minmax']
dropout_ss = hyp_search_space['dropout_ss']
active_func_ss = hyp_search_space['active_func_ss']
optimizer_ss = hyp_search_space['optimizer_ss']
lr_min, lr_max = hyp_search_space['learning_rate_minmax']

def build_model(hp):
    dnn_layers_ss = [1, 2, 3]
    dnn_units_min, dnn_units_max = 32, 512
    dropout_ss = [0.1, 0.2]
    active_func_ss = ['relu', 'tanh']
    optimizer_ss = ['adam']
    lr_min, lr_max = 1e-4, 1e-1

    active_func = hp.Choice('activation', active_func_ss)
    optimizer = hp.Choice('optimizer', optimizer_ss)
    lr = hp.Float('learning_rate', min_value=lr_min, max_value=lr_max, sampling='log')

    inputs = tf.keras.Input(shape=(28, 28))
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

    if optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == "SGD":
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    else:
        raise ("Not supported optimizer")

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model