#!/usr/bin/env python

"""
TensorFlow MLP Mnist

Reference:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_softmax.py
    See extensive documentation at http://tensorflow.org/tutorials/mnist/beginners/index.md

A very simple MNIST classifer.
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
from six.moves import urllib, xrange
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import Utils.benchmark_util as util
import numpy as np

NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * 1
ONE_HOT = True

TOWER_NAME = 'mlp_tower'

FLAGS = tf.app.flags.FLAGS
# max_iteration = (epochs * numExamples)/batchSize (15 * 60000)/128
tf.app.flags.DEFINE_string('core_type', 'CPU', 'Directory to put the training data.')
tf.app.flags.DEFINE_integer('max_iter', 9000, 'Number of iterations to run trainer.')
tf.app.flags.DEFINE_integer('test_iter', 100, 'Number of iterations to run test.')
tf.app.flags.DEFINE_integer('hidden1_units', 1000, 'Number of units in hidden layer 1.')
tf.app.flags.DEFINE_integer('batch_size', 100, 'Batch size. Must divide evenly into the dataset sizes.')
tf.app.flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
tf.app.flags.DEFINE_float('learning_rate', 6e-4, 'Initial learning rate.')
tf.app.flags.DEFINE_float('momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('l2', 1e-4, 'Weight decay.')
tf.app.flags.DEFINE_integer('seed', 42, 'Random seed.')


def _inference(images):
    """Build the MNIST model up to where it may be used for inference.
    """
    util.LOGGER.debug("Build Model")
    with tf.variable_scope('hidden1'):
        weights = util.init_weights([IMAGE_PIXELS, FLAGS.hidden1_units], FLAGS.seed, FLAGS.l2)
        biases = tf.Variable(tf.zeros([FLAGS.hidden1_units], dtype=util.DTYPE), dtype=util.DTYPE, name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
    with tf.variable_scope('softmax_linear'):
        weights = util.init_weights([FLAGS.hidden1_units, NUM_CLASSES], FLAGS.seed, FLAGS.l2)
        biases = tf.Variable(tf.zeros([NUM_CLASSES], dtype=util.DTYPE), dtype=util.DTYPE, name='biases')
        logits = tf.nn.softmax(tf.matmul(hidden1, weights) + biases)
    return logits


def run():
    total_time = time.time()
    num_gpus = util.NUM_GPUS[FLAGS.core_type]

    data_load_time = time.time()
    # Import data
    mnist = util.load_data(input_data, ONE_HOT)
    data_load_time = time.time() - data_load_time
    with tf.Graph().as_default():
        util.LOGGER.debug("Load Data")
        images_placeholder, labels_placeholder = util.placeholder_inputs(ONE_HOT, IMAGE_PIXELS, NUM_CLASSES)

        # Build model
        logits = _inference(images_placeholder)

        # Define loss and optimizer
        cross_entropy = -tf.reduce_sum(labels_placeholder*tf.log(logits)) # softmax & cross entropy
        train_op = util.setup_optimizer(cross_entropy, FLAGS.learning_rate, FLAGS.momentum)

        config = tf.ConfigProto(device_count={'GPU': num_gpus})
        sess = tf.InteractiveSession(config=config)
        sess.run(tf.initialize_all_variables())

        train_time = time.time()
        util.LOGGER.debug("Train Model")
        for iter in xrange(FLAGS.max_iter):
            feed_dict = util.fill_feed_dict(mnist.train, images_placeholder, labels_placeholder, FLAGS.batch_size)
            _, loss_value = sess.run([train_op, cross_entropy], feed_dict=feed_dict)
            if iter % 100 == 0:  util.LOGGER.debug('Iter %d: loss = %.2f' % (iter, loss_value))
        train_time = time.time() - train_time

    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

    # Test trained model
    test_time = time.time()
    util.do_eval(sess, logits, images_placeholder, labels_placeholder, mnist.test, ONE_HOT, FLAGS.test_iter, FLAGS.batch_size)
    test_time = time.time() - test_time
    sess.close

    total_time = time.time() - total_time
    print("****************Example finished********************")
    util.printTime('Data load', data_load_time)
    util.printTime('Train', train_time)
    util.printTime('Test', test_time)
    util.printTime('Total', total_time)


if __name__ == "__main__":
    run()

