import os

from tensorflow.python.training import training_util

import local_mnist as input_data
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'mnist_mcn_log',
                           """Directory where to write event logs """
                           """and checkpoint.""")

def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, W, name=None):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def inference(x, keep_prob):
    with tf.name_scope("hidden1"):
        W_conv1 = weight_variable([5, 5, 1, 32], name="weights")
        tf.summary.histogram('W_conv1', W_conv1)
        b_conv1 = bias_variable([32], name="biases")
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, "conv2d") + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)
    #
    with tf.name_scope("hidden2"):
        W_conv2 = weight_variable([5, 5, 32, 64], name="weights")
        b_conv2 = bias_variable([64], name="biases")
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, "conv2d") + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
    #
    with tf.name_scope("fully1"):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    #
    with tf.name_scope("fully2"):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv


def loss(y_conv, y_):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_, name="loss_entropy")
                                   , name="loss_entropy_mean")
    return cross_entropy


def training(cross_entropy):
    tf.summary.scalar('loss', cross_entropy)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, name="minimize")
    return train_step


def validation(y_conv, y_):
    with tf.name_scope("validation"):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


def run_training():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, shape=[None, 784])
        y_ = tf.placeholder(tf.float32, shape=[None, 10])
        keep_prob = tf.placeholder(tf.float32)

        y_conv = inference(x, keep_prob)
        cross_entropy = loss(y_conv, y_)
        train_step = training(cross_entropy)
        accuracy = validation(y_conv, y_)

        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.summary.merge_all()

        sess = tf.InteractiveSession()
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        training_util.write_graph(
            sess.graph.as_graph_def(add_shapes=True),
            FLAGS.train_dir,
            "graph.pbtxt")

        for step in xrange(2000):
            batch = mnist.train.next_batch(50)
            feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5}
            _, loss_value = sess.run([train_step, cross_entropy], feed_dict=feed_dict)

            if step % 100 == 0:
                saver.save(sess, os.path.join(FLAGS.train_dir, "model.ckpt"), step)
                train_accuracy = accuracy.eval(feed_dict=feed_dict)
                print("step %d, loss_value %.2f,training accuracy %g" % (step, loss_value, train_accuracy))
                # Update the events file.
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
        # Test
        print("test accuracy %g" % accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


def main(argv=None):
    run_training()

if __name__ == '__main__':
    tf.app.run()