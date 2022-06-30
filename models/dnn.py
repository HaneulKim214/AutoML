import keras_tuner as kt
import tensorflow as tf


def simple_dnn(inputs, dnn_units, dnn_dropout, active_func):
    """
    Function that builds simple deep neural net model
     
    :param inputs: 
    :param dnn_dim: 
    :param dnn_dropout: 
    :param active_func: 
    :return: 
    """
    flatten_layer = tf.keras.layers.Flatten()(inputs)
    dense = tf.keras.layers.Dense(units=dnn_units[0], activation=active_func)(flatten_layer)
    for n_units in dnn_units[1:]:
        dense = tf.keras.layers.Dense(units=n_units, activation=active_func)(dense)
        if dnn_dropout:
            dense = tf.keras.layers.Dropout(rate=0.25)(dense)
    outputs = tf.keras.layers.Dense(units=10)(dense)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


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

class SemiCustomHyperModel(kt.HyperModel):
    def __init__(self, model_hyp_search_space, compile_hyp_search_space, inputs, loss_fn):
        """
        :param model_hyp_search_space: dict, search space w.r.t model
        :param compile_hyp_search_space: dict, search space containing hyperparameters relating to compile
        """
        self.dnn_layers_ss = model_hyp_search_space['dnn_layers_ss']
        self.dnn_units_min, self.dnn_units_max = model_hyp_search_space['dnn_units_minmax']
        self.dropout_ss = model_hyp_search_space['dropout_ss']

        self.active_func_ss = compile_hyp_search_space['active_func_ss']

        self.optimizer_ss = compile_hyp_search_space['optimizer_ss']
        self.lr_min, self.lr_max =  compile_hyp_search_space['learning_rate_minmax']

        self.inputs = inputs
        self.loss_fn = loss_fn
        self.n_epochs = 10


    def build(self, hp):
        """build your model"""
        # Note hp.choice, needs to happen here b.c. build function is the function
        # that called everytime tuner starts a new search
        # select hyperparameters for each build
        active_func = hp.Choice('activation', self.active_func_ss)
        n_layers = hp.Choice("n_layers", self.dnn_layers_ss)
        dnn_dropout = hp.Boolean("dropout")
        dnn_units = []
        for layer_i in range(n_layers):
            n_units = hp.Int(f"{layer_i}_units", min_value=self.dnn_units_min, max_value=self.dnn_units_max)
            dnn_units.append(n_units)

        # Choose model that you want and simply pass in as parameters hyperparameters chosen at each build.
        possible_model_dict = {"simple_dnn": simple_dnn}
        model = possible_model_dict['simple_dnn'](self.inputs, dnn_units, dnn_dropout, active_func)
        return model