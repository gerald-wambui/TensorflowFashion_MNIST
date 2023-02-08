#image classifier using convolutional neural net

#imports
import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

#Helper libs
import math
import numpy as np
import matplotlib.pyplot as plt

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)