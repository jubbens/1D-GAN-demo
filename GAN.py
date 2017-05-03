import tensorflow as tf
import random
import matplotlib
# For remote X11
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt

# Options
lr = 0.0001
max_iter = 10
inner_iter = 10
hidden_size = 30
batch_size = 10
num_demo_samples = 100

# The input noise for the generator
g_input = tf.random_uniform(shape=(batch_size,1), minval=0.0, maxval=1.0)
test_g_input = tf.random_uniform(shape=(num_demo_samples,1), minval=0.0, maxval=1.0)

# The distribution we are trying to fit
target_distribution = tf.random_normal(shape=(batch_size,1), mean=1.0, stddev=0.5)

# Session
session = tf.Session()

# Network weights
g_weights_hidden = tf.get_variable('g_weights_1', shape=[1, hidden_size], initializer=tf.truncated_normal_initializer(), dtype=tf.float32)
g_bias_hidden = tf.get_variable('g_bias_1', [hidden_size], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
g_weights_output = tf.get_variable('g_weights_2', shape=[hidden_size, 1], initializer=tf.truncated_normal_initializer(), dtype=tf.float32)
g_bias_output = tf.get_variable('g_bias_2', [1], initializer=tf.constant_initializer(0.1), dtype=tf.float32)

d_weights_hidden = tf.get_variable('d_weights_1', shape=[1, hidden_size], initializer=tf.truncated_normal_initializer(), dtype=tf.float32)
d_bias_hidden = tf.get_variable('d_bias_1', [hidden_size], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
d_weights_output = tf.get_variable('d_weights_2', shape=[hidden_size, 1], initializer=tf.truncated_normal_initializer(), dtype=tf.float32)
d_bias_output = tf.get_variable('d_bias_2', [1], initializer=tf.constant_initializer(0.1), dtype=tf.float32)

# Generator network
def generator(x):
    activations_hidden = tf.nn.tanh(tf.add(tf.matmul(x, g_weights_hidden), g_bias_hidden))
    activations_output = tf.add(tf.matmul(activations_hidden, g_weights_output), g_bias_output)

    return activations_output

# Discriminator network
def discriminator(x):
    activations_hidden = tf.nn.tanh(tf.add(tf.matmul(x, d_weights_hidden), d_bias_hidden))
    activations_output = tf.nn.sigmoid(tf.add(tf.matmul(activations_hidden, d_weights_output), d_bias_output))

    return activations_output

# Loss function for generator
def g_get_loss(predicted_generated):
    return tf.reduce_mean(-tf.log(predicted_generated))

# Loss function for discriminator
def d_get_loss(predicted_generated, predicted_real):
    return tf.reduce_mean(-tf.log(predicted_generated) - tf.log(1 - predicted_real))

session.run(tf.global_variables_initializer())

for i in range(max_iter):
    print('Epoch %d' % i)

    # Train discriminator
    for j in range(inner_iter):
        # A batch from the real distribution
        d_input_real = target_distribution

        # A batch from the generator
        d_input_generated = generator(g_input)

        # Pass both through discriminator
        d_score_real = discriminator(d_input_real)
        d_score_generated = discriminator(d_input_generated)

        d_loss = d_get_loss(d_score_generated, d_score_real)

        # Optimize
        d_train_op = tf.train.GradientDescentOptimizer(lr).minimize(d_loss)
	session.run([d_train_op])

        if j == inner_iter-1:
            dl = session.run([d_loss])
            print('Discriminator loss: %f' % dl[0])

    # Train generator
    for j in range(inner_iter):
        # We are feeding the discriminator a fake sample
        d_input = generator(g_input)

        # Pass generated sample through discriminator
        d_score = discriminator(d_input)
        g_loss = g_get_loss(d_score)

        # Optimize
        g_train_op = tf.train.GradientDescentOptimizer(lr).minimize(g_loss)
	session.run([g_train_op])

        if j == inner_iter-1:
            gl = session.run([g_loss])
            print('Generator loss: %f' % gl[0])

# Generate a bunch of samples using the generator and display a histogram of them
print('Generating histogram using generator, please wait...')

g_input = generator(test_g_input)
samples = session.run([g_input])

_, _, _ = plt.hist(samples, int(num_demo_samples/10.0))
plt.show()

print('Done')
