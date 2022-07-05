import keras_tuner as kt
import tensorflow as tf

from hpo_method import build_tuner, HyperModel
from const import model_hyp_search_space, compile_hyp_search_space
from models.dnn import simple_dnn


if __name__ == '__main__':
    (img_train, label_train), (img_test, label_test) = tf.keras.datasets.fashion_mnist.load_data()
    train_ds = tf.data.Dataset.from_tensor_slices((img_train / 255.0, label_train)).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((img_test / 255.0, label_test)).batch(32)

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, verbose=1)]
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy()
        # ,tf.keras.metrics.Accuracy()
        # ,tf.keras.metrics.Recall()
        # ,tf.keras.metrics.Precision()
        # ,tf.keras.metrics.AUC()
    ]

    inputs = tf.keras.Input(shape=(28,28))
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    hypermodel = HyperModel(model_hyp_search_space, compile_hyp_search_space, inputs, loss_fn)

    objective = kt.Objective('val_sparse_categorical_accuracy', direction='max')
    kwargs = {"objective":objective, "dir_name":"simple_dnn_v1"}
    tuner = build_tuner(hypermodel, hpo_method="RandomSearch", num_trials=3, **kwargs)
    tuner.search(train_ds, ds_valid=test_ds, metrics=metrics, epochs=3, callbacks=callbacks)

    # train with best hyperparameter on full dataset.
    total_ds = train_ds.concatenate(test_ds)
    best_hps = tuner.get_best_hyperparameters()[0]
    best_model = hypermodel.build(best_hps)
    history = tuner.hypermodel.fit(best_hps, best_model, total_ds, metrics=metrics, epochs=10)

