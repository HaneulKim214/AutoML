import keras_tuner as kt
import tensorflow as tf

from hpo_method import build_tuner, HyperModel
from const import model_hyp_search_space, compile_hyp_search_space









if __name__ == '__main__':
    (img_train, label_train), (img_test, label_test) = tf.keras.datasets.fashion_mnist.load_data()
    train_ds = tf.data.Dataset.from_tensor_slices((img_train / 255.0, label_train)).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((img_test / 255.0, label_test)).batch(32)

    metrics = [
        # tf.keras.metrics.Accuracy(),
               # tf.keras.metrics.Recall(),
               # tf.keras.metrics.Precision(),
               # tf.keras.metrics.AUC()
               ]
    inputs = tf.keras.Input(shape=(28,28))
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    objective = kt.Objective('val_accuracy', direction='max')
    hypermodel = HyperModel(model_hyp_search_space, compile_hyp_search_space, inputs, loss_fn)
    tuner = build_tuner(hypermodel, "RandomSearch", objective, "simple_dnn_v1")


    tuner.search(train_ds, validation_data=test_ds, metrics=metrics, epochs=3)
    best_hps = tuner.get_best_hyperparameters()[0]
