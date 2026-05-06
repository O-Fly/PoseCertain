"""
This is the main implementation of UrsonNet.

Disclaimer:
Part of this code was adapted from
https://github.com/matterport/Mask_RCNN
Copyright (c)  2017 Matterport, INC.
Licenced under the MIT Licence

TODO:
- layer_regex replace 'fpn' and 'pose_'

"""

import os
import random
import datetime
import re
import math
import logging
from collections import OrderedDict
import multiprocessing
import numpy as np
import skimage.transform
import tensorflow as tf
from ursonet.model import UrsoNet            #在这里import model.py中的Ursonet类

from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
# KE (Keras Engine) is internal in TF2, usually KL or KM covers it,
# or we import specific base classes if needed.
# For this script, standard layers are sufficient.

from ursonet.nn.backbones_resnet import resnet_graph   #重构后将原定义import
from ursonet.nn.backbones_shallow import resnet_shallow_graph
from ursonet.nn.heads_loc import build_loc_graph
from ursonet.nn.heads_ori import build_ori_graph
from ursonet.data.formatting import compose_image_meta, mold_image, unmold_image
from ursonet.data.generator import load_image_gt, data_generator


import imgaug as ia
from imgaug import augmenters as iaa

import utils

# [Modified for TF2] Enable TF1 compatibility mode
# This is crucial because the original code relies on static graphs and sessions
tf.compat.v1.disable_eager_execution()

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion

# [Modified] Updated assertions for TF2 environment
print(f"TensorFlow Version: {tf.__version__}")
assert LooseVersion(tf.__version__) >= LooseVersion("2.0")

############################################################
#  Utility Functions
############################################################

from ursonet.nn.layers import BatchNorm, log

############################################################
#  Resnet Graph
############################################################

# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

############################################################
#  Shallow Resnet
############################################################

# Code adopted from:
# https://github.com/qubvel/classification_models/blob/master/classification_models/resnet/builder.py


############################################################
#  Network Heads
############################################################



############################################################
#  Data Generator
############################################################

############################################################
#  NNetwork Class and Graph Initialization
############################################################


############################################################
#  Data Formatting
############################################################


############################################################
#  Profiling Functions
############################################################

def get_flops(model):
    # [Modified for TF2] Profiling in TF2 is different and session is not available by default.
    # Disabling this function to prevent crash.
    print("Warning: get_flops is not supported in this TF2 port.")
    return 0

    # run_meta = tf.RunMetadata()
    # opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # # We use the Keras session graph in the call to the profiler.
    # flops = tf.profiler.profile(graph=K.get_session().graph,
    #                             run_meta=run_meta, cmd='op', options=opts)

    # return flops.total_float_ops  # Prints the "flops" of the model.
