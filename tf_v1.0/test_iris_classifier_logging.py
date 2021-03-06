from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import tensorflow.contrib as contrib
import numpy as np
import os


tf.logging.set_verbosity(tf.logging.INFO)

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

# Load datasets.
local_file = os.path.join("../tmp/", IRIS_TRAINING)
training_set = contrib.learn.datasets.base.load_csv_with_header(
    filename=local_file,
    target_dtype=np.int,
    features_dtype=np.float32)

local_file = os.path.join("../tmp/", IRIS_TEST)
test_set = contrib.learn.datasets.base.load_csv_with_header(
    filename=local_file,
    target_dtype=np.int,
    features_dtype=np.float32)

# validation
validation_metrics = {"accuracy": tf.contrib.metrics.streaming_accuracy,
                      "precision": tf.contrib.metrics.streaming_precision,
                      "recall": tf.contrib.metrics.streaming_recall}
validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    test_set.data,
    test_set.target,
    every_n_steps=50,
    metrics=validation_metrics,
    early_stopping_metric="loss",
    early_stopping_metric_minimize=True,
    early_stopping_rounds=200)

# Model
# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="/tmp/iris_model")

# Fit model
classifier.fit(x=training_set.data, y=training_set.target, steps=2000)

accuracy_score = classifier.evaluate(x=test_set.data, y=test_set.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))

# Classify two new flower samples.
new_samples = np.array(
    [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
y = list(classifier.predict(new_samples, as_iterable=True))
print('Predictions: {}'.format(str(y)))