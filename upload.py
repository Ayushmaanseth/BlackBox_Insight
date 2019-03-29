import urllib.request, urllib.parse, urllib.error
import os,subprocess
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow import data
import math
from tensorflow.python.data import Dataset
import functools
import urllib.request, urllib.parse, urllib.error
import webbrowser
import eli5

def upload_for_tensorboard(file_path,model_path=None):
    os.system('tensorboard --logdir . &')
    #os.system('docker run -p 8500:8500 --mount type=bind,source=%s,target=/models/my_model/ -e MODEL_NAME=my_model -t tensorflow/serving &' % model_path)
    what_if_tool_path = ('http://localhost:6006/#whatif&inferenceAddress1=%s&modelName1=my_model&examplesPath=%s' %(urllib.parse.quote('localhost:8500'), urllib.parse.quote(file_path)))
    return what_if_tool_path
# Creates a tf feature spec from the dataframe and columns specified.
def create_feature_spec(df, columns):
    feature_spec = {}
    for f in columns:
        if df[f].dtype is np.dtype(np.int64):
            feature_spec[f] = tf.FixedLenFeature(shape=(), dtype=tf.int64)
        elif df[f].dtype is np.dtype(np.float64):
            feature_spec[f] = tf.FixedLenFeature(shape=(), dtype=tf.float32)
        else:
            feature_spec[f] = tf.FixedLenFeature(shape=(), dtype=tf.string)
    return feature_spec

# Parses a serialized tf.Example into input features and target feature from
# the provided label feature name and feature spec.
def parse_tf_example(example_proto, label, feature_spec):
    parsed_features = tf.parse_example(serialized=example_proto, features=feature_spec)
    target = parsed_features.pop(label)
    return parsed_features, target

# An input function for providing input to a model from tf.Examples from tf record files.
def tfrecords_input_fn(files_name_pattern, feature_spec, label, mode=tf.estimator.ModeKeys.EVAL,
                       num_epochs=None,
                       batch_size=64):
    shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False
    file_names = tf.matching_files(files_name_pattern)
    dataset = data.TFRecordDataset(filenames=file_names)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=2 * batch_size + 1)

    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda tf_example: parse_tf_example(tf_example, label, feature_spec))
    dataset = dataset.repeat(num_epochs)
    return dataset


# Creates simple numeric and categorical feature columns from a feature spec and a
# list of columns from that spec to use.
#
# NOTE: Models might perform better with some feature engineering such as bucketed
# numeric columns and hash-bucket/embedding columns for categorical features.
def create_feature_columns(df,columns, feature_spec):
    ret = []
    for col in columns:
        if feature_spec[col].dtype is tf.int64 or feature_spec[col].dtype is tf.float32:
            ret.append(tf.feature_column.numeric_column(col))
        else:
            ret.append(tf.feature_column.indicator_column(
                tf.feature_column.categorical_column_with_vocabulary_list(col, list(df[col].unique()))))
    return ret
def write_df_as_tfrecord(df, filename, columns=None):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    writer = tf.python_io.TFRecordWriter(filename)
    if columns == None:
        columns = df.columns.values.tolist()
    for index, row in df.iterrows():
        example = tf.train.Example()
        for col in columns:
            if df[col].dtype is np.dtype(np.int64):
                example.features.feature[col].int64_list.value.append(int(row[col]))
            elif df[col].dtype is np.dtype(np.float64):
                example.features.feature[col].float_list.value.append(row[col])
            elif row[col] == row[col]:
                example.features.feature[col].bytes_list.value.append(row[col].encode('utf-8'))
        writer.write(example.SerializeToString())
    writer.close()

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):

    """Trains a linear regression model of multiple features.

    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """

    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(10000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def run_model(file_path,model_path,columns,target,zero_value=None):
    csv_path = file_path
    file_name = csv_path.split('/')[-1]
    file_first_name = file_name.split('.')[0]
    os.chdir('/home/datasets')
    if file_first_name in os.listdir():
        print("Dataset and model already there")
        tfrecord_path = '/home/datasets/' + file_first_name + '/' + file_first_name + '.tfrecord'
        model_path = '/home/datasets/' + file_first_name + '/' + 'trained_model'

    else:
        print("Dataset and model not there, creating dir...")
        os.mkdir('/home/datasets/' + file_first_name)
        tfrecord_path = '/home/datasets/' + file_first_name + '/' + file_first_name + '.tfrecord'
        print("Dataset dir created at",tfrecord_path)
        os.mkdir('/home/datasets/' + file_first_name + '/' + 'trained_model')
        model_path = '/home/datasets/' + file_first_name + '/' + 'trained_model'
        print("Model dir created at",model_path)
        df = pd.read_csv(csv_path,names=columns,skipinitialspace=True)

        #csv_columns = df.columns.values.tolist()
        #print(columns)
        print("File read...columns = ",columns)
        target_column = target
        if target_column not in columns:
            print("target column error")
            return("target column error")
        elif zero_value not in df[target_column].tolist():
            print(zero_value == df[target_column].tolist()[1])
            return("zero value error")
        df[target] = np.where(df[target] == zero_value, 0, 1)
        input_features = columns
        if target_column in input_features:
            input_features.remove(target_column)
        print(input_features)
        features_and_labels = input_features + [target_column]
        print(features_and_labels)
        #tfrecord_path = '/home/datasets/data.tfrecord'
        print("Features and labels recorded...")
        write_df_as_tfrecord(df,tfrecord_path)
        print("Dataset saved at ",tfrecord_path)
        feature_spec = create_feature_spec(df,features_and_labels)
        train_inpf = functools.partial(tfrecords_input_fn, tfrecord_path,
                                        feature_spec, target_column)
        classifier = tf.estimator.LinearClassifier(feature_columns=create_feature_columns(df,input_features, feature_spec))
        classifier.train(train_inpf, steps=1000)
        print("Model trained...")
        serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        classifier.export_savedmodel(model_path, serving_input_fn)
        print("Model exported at ",model_path)

    os.system('docker run -p 8500:8500 --mount type=bind,source=%s,target=/models/my_model/ -e MODEL_NAME=my_model -t tensorflow/serving &' % model_path)
    what_if_tool_path = ('http://localhost:6006/#whatif&inferenceAddress1=%s&modelName1=my_model&examplesPath=%s'
                            %(urllib.parse.quote('localhost:8500'), urllib.parse.quote(tfrecord_path)))
    print(what_if_tool_path)
    return what_if_tool_path
    #return tfrecord_path

def run_protobuf_model(file_path,model_path,columns,target,zero_value=None):
    csv_path = file_path
    df = pd.read_csv(csv_path,names=columns,skipinitialspace=True)

    #csv_columns = df.columns.values.tolist()
    #print(columns)
    print("File read...columns = ",columns)
    target_column = target
    if target_column not in columns:
        print("target column error")
        return("target column error")
    elif zero_value not in df[target_column].tolist():
        print(zero_value == df[target_column].tolist()[1])
        return("zero value error")
    df[target] = np.where(df[target] == zero_value, 0, 1)
    input_features = columns
    if target_column in input_features:
        input_features.remove(target_column)
    print(input_features)
    features_and_labels = input_features + [target_column]
    print(features_and_labels)
    tfrecord_path = '/home/datasets/data.tfrecord'
    print("Features and labels recorded...")
    write_df_as_tfrecord(df,tfrecord_path)
    print("Dataset saved at ",tfrecord_path)

    os.system('docker run -p 8500:8500 --mount type=bind,source=%s,target=/models/my_model/ -e MODEL_NAME=my_model -t tensorflow/serving &' % model_path)
    what_if_tool_path = ('http://localhost:6006/#whatif&inferenceAddress1=%s&modelName1=my_model&examplesPath=%s'
                            %(urllib.parse.quote('localhost:8500'), urllib.parse.quote(tfrecord_path)))
    print(what_if_tool_path)
    return what_if_tool_path
    #return tfrecord_path
