from datetime import datetime
import os.path
import time
import math
import random
import glob

import numpy as np
import tensorflow as tf


from inputs_domain import generate_batch_to_train
from models_domain import model, compute_loss

#os.environ["CUDA_VISIBLE_DEVICES"]="1"

slim = tf.contrib.slim

# command line options
FLAGS = tf.app.flags.FLAGS



## import pretrained model
tf.app.flags.DEFINE_string('model_ckpt', 'kinetics-i3d/data/checkpoints/dvs_imagenet/model.ckpt',
                           """pretrained model checkpoint""")


# Input flags
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('dvs_image_size', 224,
                            """Spatial size of dvs input.""")
tf.app.flags.DEFINE_integer('brox_image_size', 224,
                            """Spatial size of brox input.""")
tf.app.flags.DEFINE_integer('dvs_depth', 10,
                            """Depth of dvs input.""")
tf.app.flags.DEFINE_integer('brox_depth', 10,
                            """Depth of brox input.""")



mode = 'dvs'



def compute_accuracy(scores, labels):
    labels = tf.cast(labels, tf.int64)
    predictions = tf.argmax(scores, 1)
    correct_predictions = tf.equal(predictions, labels)
    accuracy = 100 * tf.reduce_mean(tf.cast(correct_predictions, "float"))
    return accuracy, predictions

