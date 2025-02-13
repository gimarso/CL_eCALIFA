#!/usr/bin/env python3
"""
create_latent_space.py

This script computes projections of galaxy data using a trained model.
It handles:
  - Parsing and normalizing TFRecord data
  - Generating sky masks from image data
  - Computing latent representations (projections) from a trained model
  - Loading galaxy parameters from FITS catalogs and pickle files
  - Saving the computed projections and associated galaxy properties

Usage:
    python create_latent_space.py
"""

import os
import ast
import pickle
import joblib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import label
from astropy.io import fits

from tensorflow.keras import layers, Model
from tensorflow.keras.models import model_from_json
import tensorflow.keras.backend as K

# Local import for configuration
from config import get_config as gconfig

# ------------------------------------------------------------------------------
# Visualization style
# ------------------------------------------------------------------------------
plt.style.use('dark_background')

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------
def cosine_similarity(x, y):
    """
    Computes the cosine similarity between tensors x and y.
    """
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return K.batch_dot(x, y, axes=-1)


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


def normalize_cube(cube, n_pca_component,normalization='min_max'):
    """
    Normalize a cube using the specified normalization strategy.
    Uses global tensors (min_pca, max_pca, median_pca, etc.) defined later.
    """
    # Use only the PCA components
    cube = cube[:, :, 0:n_pca_component+1]
    
    if normalization == 'min_max':
        where_zero = tf.reduce_sum(cube, axis=0) == 0
        cube = (cube - min_pca[tf.newaxis, tf.newaxis, 0:n_pca_component]) / (
            max_pca[tf.newaxis, tf.newaxis, 0:n_pca_component] - min_pca[tf.newaxis, tf.newaxis, 0:n_pca_component])
        cube = tf.where(where_zero, 0.0, cube)
    elif normalization == 'median_mad':
        where_zero = tf.reduce_sum(cube, axis=0) == 0
        cube = (cube[:, :, 0:n_pca_component] - median_pca[tf.newaxis, tf.newaxis, 0:n_pca_component]) / (
            mad_pca[tf.newaxis, tf.newaxis, 0:n_pca_component])
        cube = tf.where(where_zero, 0.0, cube)
    elif normalization == 'mean_sigma':
        where_zero = tf.reduce_sum(cube, axis=0) == 0
        cube = (cube - mean_pca[tf.newaxis, tf.newaxis, 0:n_pca_component+1]) / (
            sigma_pca[tf.newaxis, tf.newaxis, 0:n_pca_component+1])
        cube = tf.where(where_zero, 0.0, cube)
    else:
        # No normalization provided, pass as is
        pass

    # Replace NaNs or infinities with zero
    where_inf = tf.math.logical_or(tf.math.is_nan(cube), tf.math.is_inf(cube))
    cube = tf.where(where_inf, 0.0, cube)
    return cube


def create_normalizer(norm_type,n_pca_component):
    """
    Returns a tf.function-wrapped cube normalization function.
    """
    @tf.function
    def normalize_cube_with_type(cube):
        return normalize_cube(cube, n_pca_component,normalization=norm_type)
    return normalize_cube_with_type


def generate_sky_mask(images, channel_index=0, sky_threshold_factor=0.1):
    """
    Generate a mask for pixels that are above the sky background level
    and are part of a contiguous region.
    
    Parameters:
      images: numpy array of shape (num_galaxies, height, width, num_channels)
      channel_index: int, channel index to use for sky detection
      sky_threshold_factor: float, factor of the maximum value to consider as sky limit

    Returns:
      masks: boolean numpy array of shape (num_galaxies, height, width)
    """
    images = images.numpy()
    num_galaxies = images.shape[0]
    masks = np.zeros(images.shape[:3], dtype=bool)

    for i in range(num_galaxies):
        galaxy_image = 10 ** images[i, :, :, channel_index]
        # Use the top five values to compute a median threshold
        top_five = np.partition(galaxy_image.flatten(), -5)[-5:]
        average_top = np.median(top_five)
        sky_threshold = sky_threshold_factor * average_top

        initial_mask = galaxy_image > sky_threshold
        labeled_array, num_features = label(initial_mask)
        if num_features > 0:
            max_label = max(range(1, num_features + 1), key=lambda x: (labeled_array == x).sum())
            masks[i] = labeled_array == max_label

    return masks


def compute_projection(data_files_a, data_files_b, n_pca_component,batch_size, normalization='None'):
    """
    Computes the projection (latent representations) and pixel ratio for a set
    of data files. Both branches (A and B) are processed.
    
    Parameters:
      data_files_a, data_files_b: lists of TFRecord files (paths)
      batch_size: integer batch size for processing
      normalization: normalization strategy to be used

    Returns:
      z: median normalized latent representation from the model
      pz: median normalized intermediate projection from the intermediate model
      all_ratio: array of pixel ratios for sky regions
    """
    assert len(data_files_a) == len(data_files_b), "Number of data files must match"

    num_realizations = len(data_files_a)
    indices = np.arange(num_realizations)
    normalize_cube_allgal = create_normalizer(normalization,n_pca_component)

    all_z = []
    all_pz = []
    all_ratio = []

    for i in indices:
        print("Processing realization:", i)
        # Load and parse datasets
        dataset_a = tf.data.TFRecordDataset(data_files_a[i]).map(_parse_function)
        dataset_b = tf.data.TFRecordDataset(data_files_b[i]).map(_parse_function)

        dataset_a = dataset_a.map(normalize_cube_allgal).batch(batch_size)
        dataset_b = dataset_b.map(normalize_cube_allgal).batch(batch_size)

        za, pza, ratio_a = [], [], []
        for batch in dataset_a:
            pred = model(batch, training=False)
            za.extend(pred.numpy())
            pred_inter = intermediate_model(batch, training=False)
            pza.extend(pred_inter.numpy())
            sky_masks = generate_sky_mask(batch, channel_index=0, sky_threshold_factor=0.08)
            ratio_a.extend(sky_masks.sum(axis=1).sum(axis=1) / tot_num_pix)

        zb, pzb, ratio_b = [], [], []
        for batch in dataset_b:
            pred = model(batch, training=False)
            zb.extend(pred.numpy())
            pred_inter = intermediate_model(batch, training=False)
            pzb.extend(pred_inter.numpy())
            sky_masks = generate_sky_mask(batch, channel_index=0, sky_threshold_factor=0.08)
            ratio_b.extend(sky_masks.sum(axis=1).sum(axis=1) / tot_num_pix)

        all_z.append(np.array(za))
        all_z.append(np.array(zb))
        all_pz.append(np.array(pza))
        all_pz.append(np.array(pzb))
        all_ratio.append(np.array(ratio_a))
        all_ratio.append(np.array(ratio_b))

    # Normalize and compute the median projection
    all_z = np.array(all_z)
    all_z = all_z / np.linalg.norm(all_z, axis=2, keepdims=True)
    z = np.median(all_z, axis=0)
    z = z / np.linalg.norm(z, axis=1, keepdims=True)

    all_pz = np.array(all_pz)
    all_pz = all_pz / np.linalg.norm(all_pz, axis=2, keepdims=True)
    pz = np.median(all_pz, axis=0)
    pz = pz / np.linalg.norm(pz, axis=1, keepdims=True)

    return z, pz, np.array(all_ratio)


def create_intermediate_model(original_model):
    """
    Creates an intermediate model that outputs the activations from a specific
    layer of the original model (here, the 'dense' layer).
    """
    print("Creating intermediate model from:")
    print(original_model.summary())

    # Adjust the layer name if needed; here we use the layer named 'dense'
    conv_output = original_model.get_layer(name='dense').output
    intermediate_model = Model(inputs=original_model.input, outputs=conv_output)
    return intermediate_model

# ------------------------------------------------------------------------------
# Main Routine
# ------------------------------------------------------------------------------
def main():
    # ----------------------- Configuration and Paths --------------------------
    name_config = '../config.cfg'
    config = gconfig(name_config)

    folder_data       = config.get('FOLDER_INFO', 'folder_data')
    folder_aux        = config.get('FOLDER_INFO', 'folder_aux')
    file_pca          = config.get('FILES_INFO', 'file_pca')
    folder_models     = config.get('FOLDER_INFO', 'folder_models')
    folder_projection = config.get('FOLDER_INFO', 'folder_projection')
    file_projection   = os.path.join(folder_projection, config.get('MODEL_INFO', 'name_model') + '.pkl')
    
    n_pca_component   = ast.literal_eval(config.get('FILES_INFO', 'n_pca_component'))
    name_model        = config.get('MODEL_INFO', 'name_model')
    mode              = config.get('MODEL_INFO', 'mode')
    normalization     = config.get('MODEL_INFO', 'normalization')
    

    
    # ----------------------- Mode-specific Settings ---------------------------
    global input_dim, n_gal, tot_n_gal, tot_num_pix, folder_train, folder_test, file_param, file_train_test, file_order_train, file_order_test
    if mode == 'high':
        input_dim    = (192, 184, n_pca_component+1)
        n_gal        = 772
        tot_n_gal    = 898
        tot_num_pix  = 17624
        folder_train = os.path.join(folder_data, 'ecube_pairs/train_sample/')
        folder_test  = os.path.join(folder_data, 'ecube_pairs/test_sample/')
        file_param   = os.path.join(folder_aux, 'sample_edr.pkl')
        file_train_test = os.path.join(folder_aux, 'train_test_edr.npz')
        file_order_train = os.path.join(folder_aux, 'order_train_ecube.pkl')
        file_order_test  = os.path.join(folder_aux, 'order_test_ecube.pkl')
        file_pca_stadistic = os.path.join(folder_aux, 'stadistic_' + file_pca)
    elif mode == 'low':
        input_dim    = (96, 92, n_pca_component+1)
        n_gal        = 802
        tot_n_gal    = 928
        tot_num_pix  = 4800
        folder_train = os.path.join(folder_data, 'cube_pairs/train_sample/')
        folder_test  = os.path.join(folder_data, 'cube_pairs/test_sample/')
        file_param   = os.path.join(folder_aux, 'sample_dr.pkl')
        file_train_test = os.path.join(folder_aux, 'train_test_dr.npz')
        file_order_train = os.path.join(folder_aux, 'order_train_cube.pkl')
        file_order_test  = os.path.join(folder_aux, 'order_test_cube.pkl')
        file_pca_stadistic = os.path.join(folder_aux, 'stadistic_' + file_pca)
    else:
        raise ValueError("Unknown mode: {}. Expected 'high' or 'low'.".format(mode))
    
    # ----------------------- Load PCA Statistics ------------------------------
    with open(file_pca_stadistic, 'rb') as output:
        pca_stadistic = pickle.load(output)

    global min_pca, max_pca, median_pca, mad_pca, mean_pca, sigma_pca
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
        # Prepend zero and one for the first channel
        zero_tensor = tf.constant([0.], dtype=tf.float32)
        one_tensor = tf.constant([1.], dtype=tf.float32)
        mean_pca = tf.concat([zero_tensor, mean_pca], 0)
        sigma_pca = tf.concat([one_tensor, sigma_pca], 0)
    else:
        raise ValueError("Normalization {} not supported.".format(normalization))
    
    # ----------------------- Load Galaxy Parameters ---------------------------
    with open(file_param, 'rb') as file:
        gal_params = pickle.load(file)

    with open(file_order_train, 'rb') as file:
        order_train = pickle.load(file)
    with open(file_order_test, 'rb') as file:
        order_test = pickle.load(file)

    train_test_info = np.load(file_train_test)
    ID_train = gal_params['ID'][train_test_info['train_sample']][order_train]
    ID_test  = gal_params['ID'][train_test_info['test_sample']][order_test]

    # Load FITS catalogues
    file_catalogue_EDR_small = os.path.join(folder_aux, 'galaxies_properties.fits')
    file_catalogue_EDR = os.path.join(folder_aux, 'eCALIFA.pyPipe3D.fits')

    y_small = fits.open(file_catalogue_EDR_small)
    y = fits.open(file_catalogue_EDR)
    header = y[1].header

    # Initialize dictionaries for galaxy properties
    all_gal_train_param = {header[key].strip(): [] for key in header.keys() if key.startswith('TTYPE')}
    all_gal_test_param  = {header[key].strip(): [] for key in header.keys() if key.startswith('TTYPE')}

    # Populate training parameters
    for i in range(len(ID_train)):
        ww = np.where(y[1].data['cubename'] == ID_train[i])
        if len(ww[0]) > 0:
            for key in all_gal_train_param.keys():
                all_gal_train_param[key].append(y[1].data[key][ww[0][0]])
        else:
            for key in all_gal_train_param.keys():
                all_gal_train_param[key].append(np.nan)
    all_gal_train_param['z'] = []
    all_gal_train_param['type'] = []

    for i in range(len(ID_train)):
        ww = np.where(y_small[1].data['cubename'] == ID_train[i])
        if len(ww[0]) > 0:
            all_gal_train_param['type'].append(y_small[1].data['type'][ww[0][0]])
            all_gal_train_param['z'].append(y_small[1].data['z'][ww[0][0]])
        else:
            all_gal_train_param['type'].append(np.nan)
            all_gal_train_param['z'].append(np.nan)
    all_gal_train_param['ID'] = ID_train
    for key in all_gal_train_param.keys():
        all_gal_train_param[key] = np.array(all_gal_train_param[key])

    # Populate test parameters
    for i in range(len(ID_test)):
        ww = np.where(y[1].data['cubename'] == ID_test[i])
        if len(ww[0]) > 0:
            for key in all_gal_test_param.keys():
                all_gal_test_param[key].append(y[1].data[key][ww[0][0]])
        else:
            for key in all_gal_test_param.keys():
                all_gal_test_param[key].append(np.nan)
    all_gal_test_param['z'] = []
    all_gal_test_param['type'] = []
    for i in range(len(ID_test)):
        ww = np.where(y_small[1].data['cubename'] == ID_test[i])
        if len(ww[0]) > 0:
            all_gal_test_param['type'].append(y_small[1].data['type'][ww[0][0]])
            all_gal_test_param['z'].append(y_small[1].data['z'][ww[0][0]])
        else:
            all_gal_test_param['type'].append(np.nan)
            all_gal_test_param['z'].append(np.nan)
    all_gal_test_param['ID'] = list(ID_test)
    for key in all_gal_test_param.keys():
        all_gal_test_param[key] = np.array(all_gal_test_param[key])

    # Correction for new galaxies (if needed)
    new_galaxies = {
        'ID': np.array(['SN2012fk_0', 'SN2012fk_1', 'ARP118_0', 'ARP118_1',
                        'NGC5421NED02_0', 'NGC5421NED02_1', 'SQbF1_1', 'SQbF1_0',
                        'Arp142_0', 'Arp142_1', 'NGC7436B_0', 'NGC7436B_1']),
        'z': np.array([0.03132, 0.03132, 0.0277, 0.0277, 0.02616, 0.02616,
                       0.01913, 0.01913, 0.02269, 0.02269, 0.02457, 0.02457]),
        'type': np.array(['E4', 'E5', 'I', 'E5', 'E3', 'Sa', 'Sa', 'E3', 'I', 'E4', 'E2', 'E5'])
    }

    for i in range(len(new_galaxies['ID'])):
        wtr = ID_train == new_galaxies['ID'][i]
        wte = ID_test == new_galaxies['ID'][i]
        if wtr.sum() > 0:
            for key in new_galaxies.keys():
                all_gal_train_param[key][wtr] = new_galaxies[key][i]
        elif wte.sum() > 0:
            for key in new_galaxies.keys():
                all_gal_test_param[key][wte] = new_galaxies[key][i]

    # Correction for a specific galaxy ('Arp141_0')
    ww = (all_gal_train_param['ID'] == 'Arp141_0')
    if ww.sum() > 0:
        all_gal_train_param['type'][ww][0] = 'I'
    else:
        ww = (all_gal_test_param['ID'] == 'Arp141_0')
        all_gal_test_param['type'][ww][0] = 'I'

    # ----------------------- Load Trained Model -------------------------------
    file_model = os.path.join(folder_models, name_model)
    with open(file_model + '.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    model_loaded = model_from_json(loaded_model_json)
    model_loaded.load_weights(file_model + '_best.h5')

    # Set the global models for projection computations
    global model, intermediate_model
    model = model_loaded
    intermediate_model = create_intermediate_model(model)

    # ----------------------- Data Files for Projection ------------------------
    data_folder_a = os.path.join(folder_train, 'A/')
    data_folder_b = os.path.join(folder_train, 'B/')
    data_files_a = sorted([os.path.join(data_folder_a, f) for f in os.listdir(data_folder_a)
                           if not f.startswith('.') and f.endswith('.tfrecord')])
    data_files_b = sorted([os.path.join(data_folder_b, f) for f in os.listdir(data_folder_b)
                           if not f.startswith('.') and f.endswith('.tfrecord')])

    test_folder_a = os.path.join(folder_test, 'A/')
    test_folder_b = os.path.join(folder_test, 'B/')
    test_files_a = sorted([os.path.join(test_folder_a, f) for f in os.listdir(test_folder_a)
                           if not f.startswith('.') and f.endswith('.tfrecord')])
    test_files_b = sorted([os.path.join(test_folder_b, f) for f in os.listdir(test_folder_b)
                           if not f.startswith('.') and f.endswith('.tfrecord')])

    # ----------------------- Compute Projections ------------------------------
    print("Computing projection for training data...")
    z_train, pz_train, ratio_pixel_train = compute_projection(data_files_a, data_files_b, n_pca_component, batch_size=16,normalization=normalization)
    print("Computing projection for test data...")
    z_test, pz_test, ratio_pixel_test = compute_projection(test_files_a, test_files_b, n_pca_component, batch_size=16,normalization=normalization)

    # ----------------------- Save Projection Results --------------------------
    dic = {
        'z_train': z_train,
        'z_test': z_test,
        'pz_train': pz_train,
        'pz_test': pz_test,
        'prop_train': all_gal_train_param,
        'prop_test': all_gal_test_param
    }
    dic['prop_train']['ratio_pix'] = ratio_pixel_train
    dic['prop_test']['ratio_pix'] = ratio_pixel_test

    with open(file_projection, 'wb') as output_file:
        pickle.dump(dic, output_file)
    print("Projection saved to:", file_projection)


if __name__ == "__main__":
    main()