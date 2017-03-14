import local_mnist as input_data
import tensorflow as tf
from tensorflow.python import debug as tf_debug

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

with tf.Graph().as_default():

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.matmul(x, W) + b

    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
        tf.summary.scalar('loss', cross_entropy)

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy, name="minimize")

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()
    sess = tf.InteractiveSession()
    summary_writer = tf.summary.FileWriter("MNIST_train", graph=sess.graph)
    sess.run(tf.global_variables_initializer())

    sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    for i in range(1000):
        batch = mnist.train.next_batch(100)
        feed_dict = {x: batch[0], y_: batch[1]}
        sess.run(train_step, feed_dict=feed_dict)
        # Update the events file.
        if i % 500 == 0:
            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, i)
            summary_writer.flush()

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels}))
