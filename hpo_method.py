import keras_tuner as kt
import numpy as np
import os
import tensorflow as tf
from pathlib import Path

from models.dnn import simple_dnn


def get_project_name(path, project_name):
    """
    Check if project_name exists given path. Generate new project_name that doesn't already exists.
    This is because keras tuner don't run when same name directory that saves trial history already exists.

    :param path:
    :return:
    """
    version = 2
    while project_name in os.listdir(path):
        # assuming projectname is in {ProjectName}_v1 form. --> better to use regex however too lazy.. :)
        project_name = project_name.split("_")[0] + f"_v{version}"
        version += 1
    return project_name

def build_tuner(model, hpo_method, num_trials, **kwargs):
    objective = kwargs.get("objective")
    dir_name = kwargs.get("dir_name", "Untitled")

    current_path = os.path.dirname(os.path.abspath(__file__))
    tuner_path = os.path.join(current_path, dir_name)
    project_name = get_project_name(tuner_path, project_name=hpo_method)

    if hpo_method == "RandomSearch":
        tuner = kt.RandomSearch(model, objective=objective, max_trials=num_trials, executions_per_trial=1,
                                project_name=project_name, directory=dir_name)
    elif hpo_method == "Hyperband":
        tuner = kt.Hyperband(model, objective=objective, max_epochs=num_trials, executions_per_trial=1,
                             project_name=project_name, directory=dir_name, overwrite=True)
    elif hpo_method == "BayesianOptimization":
        tuner = kt.BayesianOptimization(model, objective=objective, max_trials=num_trials, executions_per_trial=1,
                                        project_name=project_name, directory=dir_name, overwrite=True)

    return tuner


class HyperModel(kt.HyperModel):
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

    def fit(self, hp, model, ds_train, metrics, ds_valid=None, callbacks=[], verbose=True, **kwargs):
        @tf.function
        def _run_train_step(images, labels):
            with tf.GradientTape() as tape:
                logits = model(images, training=True)
                loss = self.loss_fn(labels, logits)
                if model.losses: # adding regularization term if exists
                    loss += tf.math.add_n(model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # update metrics
            epoch_train_loss_metric.update_state(loss)
            for metric in metrics:
                metric.update_state(y_batch, logits)

            # print(metric.result())
            # total = len(logits)
            # true_cnt = np.sum(np.equal(y_batch, np.argmax(logits, axis=1)))
            # print(f"Accuracy = {round(true_cnt / total, 4)}")

        @tf.function
        def _run_val_step(images, labels):
            logits = model(images, training=False)
            loss = self.loss_fn(labels, logits)
            epoch_val_loss_metric.update_state(loss)
            for metric in metrics:
                metric.update_state(y_batch, logits)

        history = self.prepare_history_dict(metrics)
        epochs = kwargs.get('epochs', 10)

        optimizer_name = hp.Choice('optimizer', self.optimizer_ss)
        lr = hp.Float('learning_rate', min_value=self.lr_min, max_value=self.lr_max, sampling='log')
        optimizer = self.get_optimizer(optimizer_name, lr)

        epoch_val_loss_metric = tf.keras.metrics.Mean()
        epoch_train_loss_metric = tf.keras.metrics.Mean()
        for callback in callbacks:
            callback.model = model

        for epoch in range(epochs):
            for x_batch, y_batch in ds_train:
                _run_train_step(x_batch, y_batch)
            epoch_train_loss = round(float(epoch_train_loss_metric.result().numpy()), 3)
            history['loss'].append(epoch_train_loss)
            history = self.save_metrics("train", metrics, history)
            metrics = self.reset_metrics(metrics)
            epoch_train_loss_metric.reset_states()

            if ds_valid != None:
                for x_batch, y_batch in ds_valid:
                    _run_val_step(x_batch, y_batch)
                epoch_val_loss = round(float(epoch_val_loss_metric.result().numpy()), 3)
                history['val_loss'].append(epoch_val_loss)
                history = self.save_metrics('valid', metrics, history)
                metrics = self.reset_metrics(metrics)
                epoch_val_loss_metric.reset_states()

            if verbose:
                print()
                print(f"Epoch {epoch + 1}/{epochs}")
                tr_metric_names = ['loss'] + [metric.name for metric in metrics]
                val_metric_names = ['val_loss'] + [f"val_{metric.name}" for metric in metrics]
                print_string = "train"
                for tr_metric_name in tr_metric_names:
                    print_string += f" - {tr_metric_name}: {history[tr_metric_name][epoch]}"
                print(print_string)
                if ds_valid != None:
                    print_string = 'validation'
                    for val_metric_name in val_metric_names:
                        print_string += f" - {val_metric_name}: {history[val_metric_name][epoch]}"
                    print(print_string)
        return history

    @staticmethod
    def get_optimizer(optimizer_name, lr):
        if optimizer_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        elif optimizer_name == 'sgd':
            optimizer = tf.keras.optimizer.SGD(learning_rate=lr)
        else:
            raise ("[ERROR] Not supported optimizer")
        return optimizer

    @staticmethod
    def reset_metrics(metrics):
        for metric in metrics:
            metric.reset_states()
        return metrics

    @staticmethod
    def prepare_history_dict(metrics):
        """
        Fill empty dictionary with given metrics.

        Ex:
        if self.metrics=[accuracy]
        then
        returns {'accuracy':[]}

        :return: dict, for storing training history for each trial
        """
        history = {}
        history['loss'] = []
        for metric in metrics:
            history[metric.name] = []

        history['val_loss'] = []
        for metric in metrics:
            history[f"val_{metric.name}"] = []
        return history

    @staticmethod
    def save_metrics(type, metrics, history):
        """update history and add prefix depending on type"""
        prefix = ""
        if type == 'valid':
            prefix = "val_"
        for metric in metrics:
            # TODO: if not float inversion, it does not get rounded...
            history[f'{prefix}{metric.name}'].append(float(metric.result().numpy()))
        return history















