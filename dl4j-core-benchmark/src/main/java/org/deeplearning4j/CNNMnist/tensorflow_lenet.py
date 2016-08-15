#!/usr/bin/env python

"""
TensorFlow Lenet

 Reference:
    https://www.tensorflow.org/versions/r0.9/tutorials/mnist/beginners/index.html
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist.py

TensorFlow install instructions: https://tensorflow.org/get_started/os_setup.html
MNIST tutorial: https://tensorflow.org/tutorials/mnist/tf/index.html
"""

# Lessons Learned
# cpu BiasOp only support NHWC
# limits to using tf.float64 on certain functions - avoid
# cuDNN required for CNNs on GPU but hard to compile above 4
# To maintain model in session prevents from setting different test batch size from training like other platform examples on GPU?


import numpy as np
import re
import tensorflow as tf
import time
from six.moves import xrange
from tensorflow.examples.tutorials.mnist import mnist
from tensorflow.examples.tutorials.mnist import input_data
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import Utils.benchmark_util as util
import pdb
import numpy as np


NUM_CLASSES = 10
HEIGHT = 28
WIDTH = 28
CHANNELS = 1
IMAGE_PIXELS = mnist.IMAGE_PIXELS
ONE_HOT = True

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0

TOWER_NAME = 'tower'

FLAGS = tf.app.flags.FLAGS
# max_iteration = (epochs * numExamples)/batchSize (11 * 60000)/66
tf.app.flags.DEFINE_string('core_type', 'CPU', 'Directory to put the training data.')
tf.app.flags.DEFINE_integer('max_iter', 9000, 'Number of iterations to run trainer.')
tf.app.flags.DEFINE_integer('test_iter', 100, 'Number of iterations to run trainer.')
tf.app.flags.DEFINE_integer('ccn_depth1', 20, 'Number of units in feed forward layer 1.')
tf.app.flags.DEFINE_integer('ccn_depth2', 50, 'Number of units in feed forward layer 1.')
tf.app.flags.DEFINE_integer('ffn1', 500, 'Number of units in feed forward layer 1.')
tf.app.flags.DEFINE_integer('batch_size', 100, 'Batch size. Must divide evenly into the dataset sizes.')
tf.app.flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.1, 'Decay factor.')
tf.app.flags.DEFINE_float('bias_learning_rate', 0.02, 'Initial bias rate.') #
tf.app.flags.DEFINE_float('momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('l2', 1e-4, 'Weight decay.')
tf.app.flags.DEFINE_float('decay_rate', 1e-3, 'Learning rate decay rate.')
tf.app.flags.DEFINE_float('policy_power', 0.75, 'Policy power.') # current inverse_time_decay is missing this as part of denom calc
tf.app.flags.DEFINE_integer('seed', 42, 'Random seed.')


def _inference(images, use_cudnn):
    """Build the MNIST model up to where it may be used for inference.
    """
    util.LOGGER.debug("Build Model")
    with tf.variable_scope('cnn1') as scope:
        images = tf.reshape(images, [FLAGS.batch_size, HEIGHT, WIDTH,  CHANNELS])
        kernel = util.init_weights([5, 5, CHANNELS, FLAGS.ccn_depth1], FLAGS.seed, FLAGS.batch_size)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], "VALID", data_format= util.DATA_FORMAT,
                            use_cudnn_on_gpu=use_cudnn) #VALID no padding
        biases = util.init_bias([FLAGS.ccn_depth1])
        bias = tf.nn.bias_add(conv, biases, data_format=util.DATA_FORMAT)
        conv1 = tf.identity(bias, name=scope.name)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID',
                           data_format=util.DATA_FORMAT, name='maxpool1')
    with tf.variable_scope('cnn2') as scope:
        kernel = util.init_weights([5, 5, FLAGS.ccn_depth1, FLAGS.ccn_depth2], FLAGS.seed, FLAGS.batch_size)
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], "VALID", data_format=util.DATA_FORMAT,
                            use_cudnn_on_gpu=use_cudnn)
        biases = util.init_bias([FLAGS.ccn_depth2])
        bias = tf.nn.bias_add(conv, biases, data_format=util.DATA_FORMAT)
        conv2 = tf.identity(bias, name=scope.name)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID',
                           data_format=util.DATA_FORMAT, name='maxpool2')
    with tf.variable_scope('ffn1') as scope:
        reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = util.init_weights([dim, FLAGS.ffn1], FLAGS.seed, FLAGS.batch_size)
        biases = util.init_bias([FLAGS.ffn1])
        hidden1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    with tf.variable_scope('softmax_linear') as scope:
        weights = util.init_weights([FLAGS.ffn1, NUM_CLASSES], FLAGS.seed, FLAGS.batch_size)
        biases = util.init_bias([NUM_CLASSES])
        logits = tf.nn.softmax(tf.add(tf.matmul(hidden1, weights), biases, name=scope.name))
    return logits


def _setup_loss(logits, labels, core_type="CPU"):
    """Calculates the loss from the logits and the labels.
    """
    # TODO setup int16 for fp16 if needed
    labels = tf.to_int32(labels) if(ONE_HOT is False and util.DTYPE == tf.float32) else labels
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy') if(ONE_HOT is False) else
                                   tf.nn.softmax_cross_entropy_with_logits(logits, labels, name='xentropy'))
    if FLAGS.core_type != "MULTI":
        tf.scalar_summary(cross_entropy.op.name, cross_entropy)
    else:
        tf.add_to_collection("losses", cross_entropy)
        tf.add_n(tf.get_collection('losses'), name='total_loss')
    return cross_entropy


def run_training(train_data, num_gpus, use_cudnn):
    """Train for a number of steps."""
    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        util.LOGGER.debug("Load Data")
        images_placeholder, labels_placeholder = util.placeholder_inputs(ONE_HOT, IMAGE_PIXELS, NUM_CLASSES)
        logits = _inference(images_placeholder, use_cudnn)

        loss = _setup_loss(logits, labels_placeholder)
        train_op = util.setup_optimizer(loss, FLAGS.learning_rate, FLAGS.momentum)

        config = tf.ConfigProto(device_count={'GPU': num_gpus})
        config.gpu_options.allow_growth = True
        sess = tf.InteractiveSession(config=config)
        sess.run(tf.initialize_all_variables())

        # Start the training loop.
        train_time = time.time()
        util.LOGGER.debug("Train Model")
        for iter in xrange(FLAGS.max_iter):
            feed_dict = util.fill_feed_dict(train_data, images_placeholder, labels_placeholder, FLAGS.batch_size)
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            # Write the summaries and print an overview fairly often.
            if iter % 100 == 0: util.LOGGER.debug('Iter %d: loss = %.2f (%.3f sec)' % (iter, loss_value, 0.0))
        train_time = time.time() - train_time
        return sess, logits, images_placeholder, labels_placeholder, train_time

'''
Multi-GPUs
'''
def tower_loss(data, scope):
    """Calculate the total loss on a single tower running the CIFAR model.
    """
    images, labels = data.next_batch(FLAGS.batch_size)

    # Build inference Graph.
    logits = _inference(tf.cast(images, util.DTYPE), False)

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = _setup_loss(logits, labels)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')

    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        loss_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', l.op.name)

        # Name each loss as '(raw)' and name the moving average version of the loss
        tf.scalar_summary(loss_name +' (raw)', l)
        tf.scalar_summary(loss_name, loss_averages.average(l))
    pdb.set_trace()
    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
    return total_loss


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    """
    print("TOWER GRADS AVG*******", tower_grads)
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        pdb.set_trace()
        average_grads.append(grad_and_var)
    return average_grads


def run_multi_training(data, num_gpus, use_cudnn):
    """Train for a number of iterations."""
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        images_placeholder, labels_placeholder = util.placeholder_inputs(ONE_HOT, IMAGE_PIXELS, NUM_CLASSES)

        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0), trainable=False)

        # Calculate the learning rate schedule.
        num_batches_per_epoch = (NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size)
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(FLAGS.learning_rate, # tech initial learning rate higher than standard
                                        global_step,
                                        decay_steps,
                                        FLAGS.learning_rate_decay_factor,
                                        staircase=True)

        # Create an optimizer that performs gradient descent.
        # opt = tf.train.MomentumOptimizer(lr, FLAGS.momentum)
        opt = tf.train.GradientDescentOptimizer(lr)
        # Calculate the gradients for each model tower.
        tower_grads = []
        for i in xrange(num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:

                    # Calculate the loss for one tower. One model constructed per tower and variables shared across
                    loss = tower_loss(data, scope)

                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()

                    # Retain the summaries from the final tower.
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                    # Calculate the gradients for the batch of data on this tower.
                    grads = opt.compute_gradients(loss)
                    # grads = opt.compute_gradients(loss, tf.get_collection(tf.GraphKeys.SUMMARIES, scope))
                    print("TOWER GRADS*******", grads)

                    # Keep track of the gradients across all towers.
                    tower_grads.append(grads)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)

        summaries.append(tf.scalar_summary('learning_rate', lr))

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.histogram_summary(var.op.name + '/gradients', grad))

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.histogram_summary(var.op.name, var))

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        summary_op = tf.merge_summary(summaries)

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        tf.train.start_queue_runners(sess=sess)
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

        train_time = time.time()
        util.LOGGER.debug("Train Model")
        for iter in xrange(FLAGS.max_iter):
            # feed_dict = util.fill_feed_dict(data, images_placeholder, labels_placeholder)
            _, loss_value = sess.run([train_op, loss])

            if iter % 100 == 0: util.LOGGER.debug('Iter %d: loss = %.2f' % (iter, loss_value))
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if iter % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, iter)

    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

    train_time = time.time() - train_time
    return sess, train_time, images_placeholder#, labels_placeholder


def run():
    total_time = time.time()

    data_load_time = time.time()
    data_sets = util.load_data(input_data, ONE_HOT)
    data_load_time = time.time() - data_load_time

    num_gpus = util.NUM_GPUS[FLAGS.core_type]
    use_cudnn = True if (core_type != "CPU") else False

    if core_type != 'MULTI':
        sess, logits, images_placeholder, labels_placeholder, train_time = run_training(data_sets.train, num_gpus, use_cudnn)
    else:
        sess, train_time, images_placeholder, labels_placeholder = run_multi_training(data_sets.train, num_gpus, use_cudnn)
        logits = _inference(images_placeholder, use_cudnn)

    test_time = time.time()
    data_sets.test
    util.do_eval(sess, logits, images_placeholder, labels_placeholder, data_sets.test, ONE_HOT, FLAGS.test_iter, FLAGS.batch_size)
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
