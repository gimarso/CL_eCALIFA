#!/usr/bin/env python3
"""
train.py

This script trains a SimSiam model using data from the CL and CALIFA surveys.
It handles:
  - Configuration and setup
  - Data parsing and normalization
  - Model architecture definition
  - Custom loss and training steps
  - Checkpointing and callbacks

Usage:
    python train.py
"""

import os
import ast
import time
import random
import pickle
import copy

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, EarlyStopping

# Local imports
from config import get_config as gconfig

# ------------------------------------------------------------------------------
# Global Setup: Seeds, GPU configuration, and Timing
# ------------------------------------------------------------------------------
t0 = time.time()
np.random.seed(42)
tf.random.set_seed(42)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# ------------------------------------------------------------------------------
# Configuration and Paths
# ------------------------------------------------------------------------------
# Path to the configuration file
NAME_CONFIG = '../config.cfg'
config = gconfig(NAME_CONFIG)

# Folders and files from configuration
folder_data = config.get('FOLDER_INFO', 'folder_data')
folder_aux = config.get('FOLDER_INFO', 'folder_aux')
file_pca = config.get('FILES_INFO', 'file_pca')
n_pca_component = ast.literal_eval(config.get('FILES_INFO', 'n_pca_component'))
folder_models = config.get('FOLDER_INFO', 'folder_models')

# Model configuration
name_model = config.get('MODEL_INFO', 'name_model')
mode = config.get('MODEL_INFO', 'mode')
normalization = config.get('MODEL_INFO', 'normalization')
BATCH_SIZE = ast.literal_eval(config.get('MODEL_INFO', 'batch_size'))
EPOCHS = ast.literal_eval(config.get('MODEL_INFO', 'epochs'))
initial_lr = ast.literal_eval(config.get('MODEL_INFO', 'initial_lr'))
decay_rate = ast.literal_eval(config.get('MODEL_INFO', 'decay_rate'))

# Mode-specific settings
if mode == 'high':
    input_dim = (192, 184, n_pca_component + 1)
    n_gal = 772
    tot_n_gal = 898
    folder_train = os.path.join(folder_data, 'ecube_pairs/train_sample/')
    folder_test = os.path.join(folder_data, 'ecube_pairs/test_sample/')
    file_pca_stadistic = os.path.join(folder_aux, 'stadistic_' + file_pca)
elif mode == 'low':
    input_dim = (96, 92, n_pca_component + 1)
    n_gal = 802
    tot_n_gal = 928
    folder_train = os.path.join(folder_data, 'cube_pairs/train_sample/')
    folder_test = os.path.join(folder_data, 'cube_pairs/test_sample/')
    file_pca_stadistic = os.path.join(folder_aux, 'stadistic_' + file_pca)
else:
    raise ValueError("Unknown mode: {}. Expected 'high' or 'low'.".format(mode))

# ------------------------------------------------------------------------------
# Load PCA Statistics and Build Normalization Tensors
# ------------------------------------------------------------------------------
with open(file_pca_stadistic, 'rb') as output:
    pca_stadistic = pickle.load(output)

if normalization == 'min_max':
    max_pca = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    min_pca = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    for v, key in enumerate(pca_stadistic.keys()):
        max_pca = max_pca.write(v, pca_stadistic[key]['max'])
        min_pca = min_pca.write(v, pca_stadistic[key]['min'])
    max_pca = max_pca.stack()
    min_pca = min_pca.stack()
elif normalization == 'median_mad':
    median_pca = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    mad_pca = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    for v, key in enumerate(pca_stadistic.keys()):
        median_pca = median_pca.write(v, pca_stadistic[key]['median'])
        mad_pca = mad_pca.write(v, pca_stadistic[key]['mad'])
    median_pca = median_pca.stack()
    mad_pca = mad_pca.stack()
elif normalization == 'mean_sigma':
    mean_pca = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    sigma_pca = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    for v, key in enumerate(pca_stadistic.keys()):
        mean_pca = mean_pca.write(v, pca_stadistic[key]['mean'])
        sigma_pca = sigma_pca.write(v, pca_stadistic[key]['sigma'])
    mean_pca = mean_pca.stack()
    sigma_pca = sigma_pca.stack()
    # Add zero and one for the first channel
    zero_tensor = tf.constant([0.], dtype=tf.float32)
    one_tensor = tf.constant([1.], dtype=tf.float32)
    mean_pca = tf.concat([zero_tensor, mean_pca], 0)
    sigma_pca = tf.concat([one_tensor, sigma_pca], 0)
else:
    raise ValueError("Unknown normalization type: {}.".format(normalization))

# ------------------------------------------------------------------------------
# Data Parsing and Normalization Functions
# ------------------------------------------------------------------------------
def _parse_function(example_proto):
    """
    Parses a single TFRecord example.
    """
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'shape': tf.io.VarLenFeature(tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    shape = tf.sparse.to_dense(parsed_example['shape'])
    image = tf.io.decode_raw(parsed_example['image'], out_type=tf.float32)
    image = tf.reshape(image, shape)
    image = tf.transpose(image, perm=[1, 2, 0])
    return image

def normalize_cube(cube, normalization_type='min_max'):
    """
    Normalize a cube using one of the normalization strategies.
    """
    cube = cube[:, :, 0:n_pca_component+1]
    
    if normalization_type == 'min_max':
        where_zero = tf.reduce_sum(cube, axis=0) == 0
        cube = (cube - min_pca[tf.newaxis, tf.newaxis, 0:n_pca_component]) / (
            max_pca[tf.newaxis, tf.newaxis, 0:n_pca_component] - min_pca[tf.newaxis, tf.newaxis, 0:n_pca_component])
        cube = tf.where(where_zero, 0.0, cube)
    elif normalization_type == 'median_mad':
        where_zero = tf.reduce_sum(cube, axis=0) == 0
        cube = (cube[:, :, 0:n_pca_component] - median_pca[tf.newaxis, tf.newaxis, 0:n_pca_component]) / (
            mad_pca[tf.newaxis, tf.newaxis, 0:n_pca_component])
        cube = tf.where(where_zero, 0.0, cube)
    elif normalization_type == 'mean_sigma':
        where_zero = tf.reduce_sum(cube, axis=0) == 0
        cube = (cube - mean_pca[tf.newaxis, tf.newaxis, 0:n_pca_component+1]) / (
            sigma_pca[tf.newaxis, tf.newaxis, 0:n_pca_component+1])
        cube = tf.where(where_zero, 0.0, cube)
    else:
        raise ValueError("Normalization type {} not supported.".format(normalization_type))
    
    # Replace NaN or inf values with zero.
    where_inf = tf.math.logical_or(tf.math.is_nan(cube), tf.math.is_inf(cube))
    cube = tf.where(where_inf, 0.0, cube)
    return cube

def create_normalizer(norm_type):
    """
    Returns a tf.function-wrapped normalizer for the specified normalization type.
    """
    @tf.function
    def normalize_fn(cube):
        return normalize_cube(cube, normalization_type=norm_type)
    return normalize_fn

def create_dataset(data_files_a, data_files_b, batch_size, n_gal, sample='train', norm_type='min_max'):
    """
    Creates a generator-based dataset yielding paired normalized cubes.
    """
    assert len(data_files_a) == len(data_files_b), "Number of data files must match"

    num_realizations = len(data_files_a)
    indices = np.arange(num_realizations)
    np.random.shuffle(indices)
    indices_galaxies = np.arange(n_gal)
    normalize_cube_allgal = create_normalizer(norm_type)

    while True:
        for i in range(num_realizations):
            dataset_a = tf.data.TFRecordDataset(data_files_a[i]).map(_parse_function)
            dataset_b = tf.data.TFRecordDataset(data_files_b[i]).map(_parse_function)
            dataset_a = dataset_a.map(normalize_cube_allgal)
            dataset_b = dataset_b.map(normalize_cube_allgal)

            tensor_a_list = list(tf.data.Dataset.as_numpy_iterator(dataset_a))
            tensor_b_list = list(tf.data.Dataset.as_numpy_iterator(dataset_b))
            random_gal = np.sort(np.random.choice(indices_galaxies, size=batch_size, replace=False))
            batch_data_a = np.array([tensor_a_list[i] for i in random_gal])
            batch_data_b = np.array([tensor_b_list[i] for i in random_gal])
            yield batch_data_a, batch_data_b

# ------------------------------------------------------------------------------
# Model Architecture and Loss Definition
# ------------------------------------------------------------------------------
def encoder_2D():
    """
    Defines the 2D encoder architecture.
    """
    dropout_rate = 0.25
    kernel_initializer = 'he_normal'
    act_function = 'relu'
    dilation_rate = 1
    reg_strength = 0

    inputs = Input(shape=input_dim)

    # Block 0
    x0 = layers.Conv2D(64, kernel_size=(5, 5), activation=act_function, dilation_rate=dilation_rate,
                       padding='same', strides=(1, 1), kernel_regularizer=regularizers.l2(reg_strength))(inputs)
    x0 = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x0)
    x0 = layers.BatchNormalization()(x0)

    # Block 1
    x1 = layers.Conv2D(128, kernel_size=(3, 3), activation=act_function, dilation_rate=dilation_rate,
                       padding='same', strides=(1, 1), kernel_regularizer=regularizers.l2(reg_strength))(x0)
    x1 = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
    x1 = layers.BatchNormalization()(x1)

    # Block 2
    x2 = layers.Conv2D(256, kernel_size=(3, 3), activation=act_function, dilation_rate=dilation_rate,
                       padding='same', strides=(1, 1), kernel_regularizer=regularizers.l2(reg_strength))(x1)
    x2 = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
    x2 = layers.BatchNormalization()(x2)

    # Block 3
    x3 = layers.Conv2D(512, kernel_size=(3, 3), activation=act_function, dilation_rate=dilation_rate,
                       padding='same', strides=(1, 1), kernel_regularizer=regularizers.l2(reg_strength))(x2)
    x3 = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
    x3 = layers.BatchNormalization()(x3)

    # Fully-connected layers
    x = layers.GlobalMaxPooling2D()(x3)
    x = layers.Dense(512, activation=act_function, kernel_initializer=kernel_initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(rate=dropout_rate)(x)
    x = layers.Dense(128, activation=act_function, kernel_initializer=kernel_initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(rate=dropout_rate)(x)
    x = layers.Dense(64, activation=act_function, kernel_initializer=kernel_initializer)(x)
    outputs = layers.BatchNormalization()(x)

    model = Model(inputs, outputs)
    return model

def _contrastive_loss(z1, z2, temperature=0.5):
    """
    Computes a contrastive loss between two representations.
    """
    z1 = K.l2_normalize(z1, axis=-1)
    z2 = K.l2_normalize(z2, axis=-1)
    exp_z1_z2 = K.exp(tf.divide(K.dot(z1, tf.transpose(z2)), temperature))
    diagonal_terms = tf.linalg.diag_part(exp_z1_z2)
    sum_off_term_columns = K.sum(exp_z1_z2, axis=0) - diagonal_terms
    sum_off_term_rows = K.sum(exp_z1_z2, axis=1) - diagonal_terms
    loss_columns = -K.log(diagonal_terms / sum_off_term_columns)
    loss_rows = -K.log(diagonal_terms / sum_off_term_rows)
    loss = loss_columns + loss_rows
    batch_size = tf.cast(tf.shape(z1)[0], tf.float32)
    normalized_loss = K.sum(loss) / batch_size
    return normalized_loss

# ------------------------------------------------------------------------------
# SimSiam Model Definition
# ------------------------------------------------------------------------------
class SimSiam(tf.keras.Model):
    """
    SimSiam model encapsulating the encoder and custom training steps.
    """
    def __init__(self, encoder):
        super(SimSiam, self).__init__()
        self.encoder = encoder
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")
        self._temperature = 0.5

    @property
    def metrics(self):
        return [self.loss_tracker, self.val_loss_tracker]

    @tf.function
    def train_step(self, data):
        X_train, X_train_t = data
        with tf.GradientTape() as tape:
            z1 = self.encoder(X_train, training=True)
            z2 = self.encoder(X_train_t, training=True)
            loss = _contrastive_loss(z1, z2, self._temperature)
        gradients = tape.gradient(loss, self.encoder.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @tf.function
    def test_step(self, data):
        X_val, X_val_t = data
        z1_val = self.encoder(X_val)
        z2_val = self.encoder(X_val_t)
        val_loss = _contrastive_loss(z1_val, z2_val, self._temperature)
        self.val_loss_tracker.update_state(val_loss)
        return {"val_loss": self.val_loss_tracker.result()}

# ------------------------------------------------------------------------------
# Custom Callback for Checkpointing Encoder Weights
# ------------------------------------------------------------------------------
class EncoderWeightsCheckpoint(Callback):
    """
    Callback to save encoder weights at regular intervals and when validation loss improves.
    """
    def __init__(self, encoder, filepath, best_filepath, save_best_only=False, monitor='val_val_loss', mode='min', save_freq=5):
        super().__init__()
        self.encoder = encoder
        self.filepath = filepath
        self.best_filepath = best_filepath
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.save_freq = save_freq
        self.best = np.Inf if mode == 'min' else -np.Inf
        self.global_batch = 0

    def on_train_batch_end(self, batch, logs=None):
        self.global_batch += 1
        if self.global_batch % self.save_freq == 0:
            step_filepath = self.filepath.format(batch=self.global_batch)
            self.encoder.save_weights(step_filepath)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_value = logs.get(self.monitor)
        if current_value is not None:
            improved = (self.mode == 'min' and current_value < self.best) or (self.mode == 'max' and current_value > self.best)
            if improved:
                self.best = current_value
                self.encoder.save_weights(self.best_filepath)

# ------------------------------------------------------------------------------
# Main Training Routine
# ------------------------------------------------------------------------------
def main():
    # Prepare training data file paths
    data_folder_a = os.path.join(folder_train, 'A/')
    data_folder_b = os.path.join(folder_train, 'B/')
    data_files_a = sorted([os.path.join(data_folder_a, f) for f in os.listdir(data_folder_a)
                           if not f.startswith('.') and f.endswith('.tfrecord')])
    data_files_b = sorted([os.path.join(data_folder_b, f) for f in os.listdir(data_folder_b)
                           if not f.startswith('.') and f.endswith('.tfrecord')])

    # Load additional data if needed (e.g., spectral wave information)
    wave = np.load(folder_aux + 'wave_4000_6800.npz')

    # Calculate training steps per epoch
    STEPS_PER_EPOCH = len(data_files_a) * n_gal // BATCH_SIZE
    print('MODE:', mode)
    print('BATCH_SIZE:', BATCH_SIZE)
    print('STEPS_PER_EPOCH:', STEPS_PER_EPOCH)
    print('NUMBER OF EPOCHS:', EPOCHS)

    # Create training dataset
    train_dataset = tf.data.Dataset.from_generator(
        lambda: create_dataset(data_files_a, data_files_b, BATCH_SIZE, n_gal, sample='train', norm_type=normalization),
        output_signature=(
            tf.TensorSpec(shape=(BATCH_SIZE, input_dim[0], input_dim[1], input_dim[2]), dtype=tf.float32),
            tf.TensorSpec(shape=(BATCH_SIZE, input_dim[0], input_dim[1], input_dim[2]), dtype=tf.float32),
        )
    ).prefetch(tf.data.AUTOTUNE)

    # Prepare validation dataset
    n_gal_test = tot_n_gal - n_gal
    BATCH_SIZE_test = n_gal_test  # Use all test galaxies if possible

    test_folder_a = os.path.join(folder_test, 'A/')
    test_folder_b = os.path.join(folder_test, 'B/')
    test_files_a = sorted([os.path.join(test_folder_a, f) for f in os.listdir(test_folder_a)
                           if not f.startswith('.') and f.endswith('.tfrecord')])
    test_files_b = sorted([os.path.join(test_folder_b, f) for f in os.listdir(test_folder_b)
                           if not f.startswith('.') and f.endswith('.tfrecord')])

    val_dataset = tf.data.Dataset.from_generator(
        lambda: create_dataset(test_files_a, test_files_b, BATCH_SIZE_test, n_gal_test, sample='test', norm_type=normalization),
        output_signature=(
            tf.TensorSpec(shape=(BATCH_SIZE_test, input_dim[0], input_dim[1], input_dim[2]), dtype=tf.float32),
            tf.TensorSpec(shape=(BATCH_SIZE_test, input_dim[0], input_dim[1], input_dim[2]), dtype=tf.float32),
        )
    ).prefetch(tf.data.AUTOTUNE)

    VAL_STEPS_PER_EPOCH = len(test_files_a) * n_gal_test // BATCH_SIZE_test

    # Define callbacks
    early_stopping = EarlyStopping(monitor="val_val_loss", patience=3, restore_best_weights=True)
    checkpoint_filepath = os.path.join(folder_models, name_model + '_{batch:04d}.h5')
    best_checkpoint_filepath = os.path.join(folder_models, name_model + '_best.h5')
    encoder_checkpoint_callback = EncoderWeightsCheckpoint(
        encoder=None,  # to be set after model creation
        filepath=checkpoint_filepath,
        best_filepath=best_checkpoint_filepath,
        save_best_only=True,
        monitor='val_val_loss',
        mode='min',
        save_freq=200
    )

    # Set up learning rate schedule and optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=STEPS_PER_EPOCH * 2,
        decay_rate=decay_rate,
        staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # Initialize or load the model if not already present
    model_json_path = os.path.join(folder_models, name_model + '.json')
    if not os.path.exists(model_json_path):
        encoder_model = encoder_2D()
        print(encoder_model.summary())

        # Build SimSiam model
        simsiam = SimSiam(encoder_model)
        simsiam.compile(optimizer=optimizer)
        
        # Save model architecture as JSON
        with open(model_json_path, 'w') as json_file:
            json_file.write(simsiam.encoder.to_json())

        # Save initial weights
        initial_weights_path = os.path.join(folder_models, name_model + '_initial.h5')
        simsiam.encoder.save_weights(initial_weights_path)

        # Set encoder for the checkpoint callback now that the model is created
        encoder_checkpoint_callback.encoder = simsiam.encoder

        # Start training
        history = simsiam.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=EPOCHS,
                steps_per_epoch=STEPS_PER_EPOCH,
                validation_steps=VAL_STEPS_PER_EPOCH,
                callbacks=[early_stopping, encoder_checkpoint_callback]
            )
    else:
        print("Model JSON already exists. Skipping training.")

if __name__ == "__main__":
    main()