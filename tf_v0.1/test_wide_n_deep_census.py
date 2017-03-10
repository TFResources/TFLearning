import tensorflow as tf
import tensorflow.contrib as contrib

# Categorical base columns.
gender = contrib.layers.sparse_column_with_keys(column_name="gender", keys=["Female", "Male"])
race = contrib.layers.sparse_column_with_keys(column_name="race", keys=[
  "Amer-Indian-Eskimo", "Asian-Pac-Islander", "Black", "Other", "White"])
education = contrib.layers.sparse_column_with_hash_bucket("education", hash_bucket_size=1000)
relationship = contrib.layers.sparse_column_with_hash_bucket("relationship", hash_bucket_size=100)
work_class = contrib.layers.sparse_column_with_hash_bucket("work_class", hash_bucket_size=100)
occupation = contrib.layers.sparse_column_with_hash_bucket("occupation", hash_bucket_size=1000)
native_country = contrib.layers.sparse_column_with_hash_bucket("native_country", hash_bucket_size=1000)

# Continuous base columns.
age = contrib.layers.real_valued_column("age")
age_buckets = contrib.layers.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
education_num = contrib.layers.real_valued_column("education_num")
capital_gain = contrib.layers.real_valued_column("capital_gain")
capital_loss = contrib.layers.real_valued_column("capital_loss")
hours_per_week = contrib.layers.real_valued_column("hours_per_week")

wide_columns = [
  gender, native_country, education, occupation, work_class, relationship, age_buckets,
  tf.contrib.layers.crossed_column([education, occupation], hash_bucket_size=int(1e4)),
  tf.contrib.layers.crossed_column([native_country, occupation], hash_bucket_size=int(1e4)),
  tf.contrib.layers.crossed_column([age_buckets, education, occupation], hash_bucket_size=int(1e6))]


deep_columns = [
  tf.contrib.layers.embedding_column(work_class, dimension=8),
  tf.contrib.layers.embedding_column(education, dimension=8),
  tf.contrib.layers.embedding_column(gender, dimension=8),
  tf.contrib.layers.embedding_column(relationship, dimension=8),
  tf.contrib.layers.embedding_column(native_country, dimension=8),
  tf.contrib.layers.embedding_column(occupation, dimension=8),
  age, education_num, capital_gain, capital_loss, hours_per_week]


import tempfile
model_dir = tempfile.mkdtemp()
m = tf.contrib.learn.DNNLinearCombinedClassifier(
    model_dir=model_dir,
    linear_feature_columns=wide_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[100, 50])


def input_fn(df):
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values)
                       for k in CONTINUOUS_COLUMNS}
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(df[k].size)],
        values=df[k].values,
        shape=[df[k].size, 1])
                        for k in CATEGORICAL_COLUMNS}
    # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols.items() + categorical_cols.items())
    # Converts the label column into a constant Tensor.
    label = tf.constant(df[LABEL_COLUMN].values)
    # Returns the feature columns and the label.
    return feature_cols, label


def train_input_fn():
    return input_fn(df_train)


def eval_input_fn():
    return input_fn(df_test)



import os
import pandas as pd

# Data sets
IRIS_TRAINING = "adult.data"
IRIS_TEST = "adult.test"

# Load datasets.
COLUMNS = ["age", "work_class", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]

local_file = os.path.join("../tmp/", IRIS_TRAINING)
df_train = pd.read_csv(local_file, names=COLUMNS, skipinitialspace=True)
local_file = os.path.join("../tmp/", IRIS_TEST)
df_test = pd.read_csv(local_file, names=COLUMNS, skipinitialspace=True, skiprows=1)

LABEL_COLUMN = "label"
df_train[LABEL_COLUMN] = (df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
df_test[LABEL_COLUMN] = (df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

CATEGORICAL_COLUMNS = ["work_class", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]


m.fit(input_fn=train_input_fn, steps=200)
results = m.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
    print "%s: %s" % (key, results[key])