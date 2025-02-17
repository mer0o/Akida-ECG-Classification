"""
TensorFlow initialization configuration
This must be imported before any other TensorFlow operations
"""

import tensorflow as tf

# Configure threading for deterministic operations
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.experimental.enable_op_determinism()
