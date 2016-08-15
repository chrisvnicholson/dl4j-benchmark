# TensorFlow Benchmark Util

import os
import tensorflow as tf
from six.moves import xrange
import logging

DTYPE = tf.float32
DEVICE = '/cpu:0'
NUM_GPUS = {'CPU': 0, 'GPU': 1, 'MULTI': 4}
DATA_DIR = os.getcwd() + "/dl4j-core-benchmark/src/main/resources/tf_data/"
DATA_FORMAT = 'NHWC' # number examples, height, width, channels

# create logger
LOGGER = logging.getLogger('simple_example')
LOGGER.setLevel(logging.DEBUG)

# create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
LOGGER.addHandler(ch)

def load_data(input_data, one_hot, core_type):
    if(one_hot is False):
        data = input_data.read_data_sets(DATA_DIR)
    else:
        data = input_data.read_data_sets(DATA_DIR, one_hot=True)
    # if (core_type != "CPU" and not fp16): tf.cast(data.uint8image, tf.float32)
    return data


def placeholder_inputs(one_hot, num_pixels, num_classes):
    """Generate placeholder variables to represent the input tensors.
    """
    images_placeholder = tf.placeholder(DTYPE, [None, num_pixels])
    labels_placeholder = tf.placeholder(DTYPE, [None]) if(one_hot is False) else \
        tf.placeholder(DTYPE, [None, num_classes])
    return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl, batch_size):
    """Fills the feed_dict for training the given step.
    """
    images_feed, labels_feed = data_set.next_batch(batch_size)

    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
    }
    return feed_dict

def init_bias(shape):
    with tf.device(DEVICE):
        return tf.get_variable(tf.zeros(shape, dtype=DTYPE), name='biases')

def init_weights(shape, seed, l2):
    with tf.device(DEVICE):
        weights = tf.get_variable("weights", shape,
                                  initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=seed, dtype=DTYPE), dtype=DTYPE)
    weight_decay = tf.mul(tf.nn.l2_loss(weights), l2, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    return weights


def setup_optimizer(loss, learning_rate, momentum):
    tf.scalar_summary(loss.op.name, loss)
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    return optimizer.minimize(loss, global_step=global_step)


def evaluation_topk(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    """
    labels = tf.to_int32(labels)
    correct = tf.nn.in_top_k(logits, labels, 10) # needs labels to be rank
    return tf.reduce_sum(tf.cast(correct, tf.int32))


def prediction(logits, labels):
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    return tf.reduce_sum(tf.cast(correct_pred, tf.float32), 0)


# TODO params are ugly - need to condense in container or turn into class
def do_eval(sess, logits, images_placeholder, labels_placeholder, data, one_hot, test_iter, batch_size):
    """Runs one evaluation against the full epoch of data.
    """
    LOGGER.debug("Evaluate Model")
    correct_count = 0
    num_examples = data.num_examples
    for _ in xrange(test_iter):
        if one_hot is False:
            feed_dict = fill_feed_dict(data, images_placeholder, labels_placeholder)
            correct_count += sess.run(evaluation_topk(logits, labels_placeholder), feed_dict=feed_dict)
        else:
            feed_dict = fill_feed_dict(data, images_placeholder, labels_placeholder, batch_size)
            correct_count += sess.run(prediction(logits, labels_placeholder), feed_dict=feed_dict)

    print("Accuracy: %.2f" % ((correct_count / num_examples) * 100))


def printTime(time_type, time):
    min = int(round(time/60))
    sec = int(round(time - min*60))
    milli = time * 1000
    print(time_type + ' load time: %s min %s sec | %s millisec' %(min, sec, milli))
