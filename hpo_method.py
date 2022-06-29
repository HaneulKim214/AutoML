import keras_tuner as kt
import tensorflow as tf


def build_tuner(model, hpo_method, objective, dir_name):
    if hpo_method == "RandomSearch":
        tuner = kt.RandomSearch(model, objective=objective, max_trials=3, executions_per_trial=1,
                                project_name=hpo_method, directory=dir_name)
    elif hpo_method == "Hyperband":
        tuner = kt.Hyperband(model, objective=objective, max_epochs=3, executions_per_trial=1,
                             project_name=hpo_method)
    elif hpo_method == "BayesianOptimization":
        tuner = kt.BayesianOptimization(model, objective=objective, max_trials=3, executions_per_trial=1,
                                        project_name=hpo_method)

    return tuner


class HyperModel(kt.HyperModel):
    def __init__(self, model_hyp_search_space, compile_hyp_search_space, inputs, loss_fn):
        """
        :param model_hyp_search_space: dict, search space w.r.t model
        :param compile_hyp_search_space: dict, search space containing hyperparameters relating to compile
        """
        self.dnn_layer_ss = model_hyp_search_space['dnn_layers_ss']
        self.dnn_units_min, self.dnn_units_min = model_hyp_search_space['dnn_units_minmax']
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
        active_func = hp.Choice('activation', self.active_func_ss)
        optimizer = hp.Choice('optimizer', self.optimizer_ss)
        lr = hp.Float('learning_rate', min_value=self.lr_min, max_value=self.lr_max, sampling='log')
        dnn_dropout = hp.Boolean("dropout")

        dnn_layers = []

        flatten_layer = tf.keras.layers.Flatten()(self.inputs)

        # create hidden layers
        dnn_units = hp.Int(f"0_units", min_value=self.dnn_units_min, max_value=self.dnn_units_max)
        dense = tf.keras.layers.Dense(units=dnn_units, activation=self.active_func)(flatten_layer)
        for layer_i in range(hp.Choice("n_layers", self.dnn_layers_ss) - 1):
            dnn_units = hp.Int(f"{layer_i}_units", min_value=self.dnn_units_min, max_value=self.dnn_units_max)
            dnn_layers.append(dnn_units)
            dense = tf.keras.layers.Dense(units=dnn_units, activation=self.active_func)(dense)
            if hp.Boolean("dropout"):
                dense = tf.keras.layers.Dropout(rate=0.25)(dense)
        outputs = tf.keras.layers.Dense(units=10)(dense)
        model = tf.keras.Model(inputs=self.inputs, outputs=outputs)

    def fit(self, hp, model, x, y, validation_data, callbacks=None, **kwargs):



        if self.optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        elif self.optimizer == 'sgd':
            optimizer = tf.keras.optimizer.SGD(learning_rate=self.lr)
        else:
            raise ("[ERROR] Not supported optimizer")

        epoch_val_loss = tf.keras.metrics.Mean()

        @tf.function
        def run_train_step(images, labels):
            with tf.GradientTape() as tape:
                logits = model(images)
                loss = loss_fn(labels, logits)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        @tf.function
        def run_val_step(images, labels):
            logits = model(images)
            loss = loss_fn(labels, logits)

            epoch_val_loss.update_state(loss)

        for epoch in range(self.n_epochs):
            # for x,y in train
            pass




loss_fn = tf.keras.losses.SparseCategorical(from_logits=True)

model_hyp_search_space = {'dnn_units_minmax': [32, 512],
                    'dnn_layers_ss': [1, 2, 3],
                    'dropout_ss': [0.1, 0.2]}
compile_hyp_search_space = {'active_func_ss': ['relu', 'tanh'],
                    'learning_rate_minmax': [1e-4, 1e-1],
                    'optimizer_ss': ['adam']}