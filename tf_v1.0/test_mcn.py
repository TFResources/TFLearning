
import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()

input = tf.Variable(tf.random_normal([1, 28, 28, 1]))

filter = tf.Variable(tf.random_normal([5, 5, 1, 32]))

sess.run(tf.global_variables_initializer())

output = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

print sess.run(output).shape

output = tf.nn.max_pool(output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print '\n\n'

print sess.run(output).shape