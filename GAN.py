import tensorflow as tf
import random
#import matplotlib
# For remote X11
#matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt

# Options
lr = 0.001
max_iter = 10000
pretrain_iter = 3000
hidden_size = 32
batch_size = 128
num_demo_samples = 500
report_rate = 100
# Number of optimization steps on D for every step on G
k = 1

# The input noise for the generator
g_input = tf.random_uniform(shape=(batch_size,1), minval=-5.0, maxval=5.0)
test_g_input = tf.random_uniform(shape=(num_demo_samples,1), minval=-5.0, maxval=5.0)

# The distribution we are trying to fit
target_distribution = tf.random_normal(shape=(batch_size,1), mean=1.0, stddev=0.5)
test_target_distribution = tf.random_normal(shape=(num_demo_samples,1), mean=1.0, stddev=0.5)

# Session
session = tf.Session()

# Network weights
g_weights_hidden = tf.get_variable('g_weights_1', shape=[1, hidden_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
g_bias_hidden = tf.get_variable('g_bias_1', [hidden_size], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
g_weights_output = tf.get_variable('g_weights_2', shape=[hidden_size, 1], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
g_bias_output = tf.get_variable('g_bias_2', [1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)

generator_vars = [g_weights_hidden, g_bias_hidden, g_weights_output, g_bias_output]

d_weights_hidden = tf.get_variable('d_weights_1', shape=[1, hidden_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
d_bias_hidden = tf.get_variable('d_bias_1', [hidden_size], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
d_weights_hidden_2 = tf.get_variable('d_weights_2', shape=[hidden_size, hidden_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
d_bias_hidden_2 = tf.get_variable('d_bias_2', [hidden_size], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
d_weights_hidden_3 = tf.get_variable('d_weights_3', shape=[hidden_size, hidden_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
d_bias_hidden_3 = tf.get_variable('d_bias_3', [hidden_size], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
d_weights_output = tf.get_variable('d_weights_4', shape=[hidden_size, 1], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
d_bias_output = tf.get_variable('d_bias_4', [1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)

discriminator_vars = [d_weights_hidden, d_bias_hidden, d_weights_hidden_2, d_bias_hidden_2,
                      d_weights_hidden_3, d_bias_hidden_3, d_weights_output, d_bias_output]

# Generator network
def generator(x):
    activations_hidden = tf.nn.tanh(tf.add(tf.matmul(x, g_weights_hidden), g_bias_hidden))
    activations_output = tf.add(tf.matmul(activations_hidden, g_weights_output), g_bias_output)

    return activations_output

# Discriminator network
def discriminator(x):
    activations_hidden = tf.nn.tanh(tf.add(tf.matmul(x, d_weights_hidden), d_bias_hidden))
    activations_hidden_2 = tf.nn.tanh(tf.add(tf.matmul(activations_hidden, d_weights_hidden_2), d_bias_hidden_2))
    activations_hidden_3 = tf.nn.tanh(tf.add(tf.matmul(activations_hidden_2, d_weights_hidden_3), d_bias_hidden_3))
    activations_output = tf.nn.sigmoid(tf.add(tf.matmul(activations_hidden_3, d_weights_output), d_bias_output))

    return activations_output

# Operations for training discriminator

# Get both real and generated samples
d_input_real = target_distribution
d_input_generated = generator(g_input)

# Pass both through discriminator
d_score_real = discriminator(d_input_real)
d_score_generated = discriminator(d_input_generated)

# Optimize discriminator loss
d_loss = tf.reduce_mean(-tf.log(d_score_real) - tf.log(1 - d_score_generated))
d_train_op = tf.train.GradientDescentOptimizer(lr).minimize(d_loss, var_list=discriminator_vars)

# Operations for training generator

# Pass generated sample through discriminator
d_score_verifying = discriminator(d_input_generated)

# Optimize generator loss
g_loss = tf.reduce_mean(-tf.log(d_score_verifying))
g_train_op = tf.train.GradientDescentOptimizer(lr).minimize(g_loss, var_list=generator_vars)

# For testing
test_generated = generator(test_g_input)

# Initialize
session.run(tf.global_variables_initializer())

# The plot
f, ax = plt.subplots(1)

# Perform pre-training
print('Pre-training discriminator...')

for i in range(pretrain_iter):
    session.run([d_train_op])

dl, gl = session.run([d_loss, g_loss])
print('Discriminator loss: %f' % dl)

# Do main optimization
print('Training GAN...')

for i in range(max_iter):
    # Train discriminator
    for j in range(k):
        session.run([d_train_op])

    # Train generator
    session.run([g_train_op])

    if i % report_rate == 0:
        dl, gl, generated_samples, real_samples = session.run([d_loss, g_loss, test_generated, test_target_distribution])
        print('Discriminator loss: %f' % dl)
        print('Generator loss: %f' % gl)

        ax.clear()
        _, _, _ = ax.hist(generated_samples, int(num_demo_samples / 10.0), histtype='step')
        _, _, _ = ax.hist(real_samples, int(num_demo_samples / 10.0), histtype='step')

        plt.draw()
        plt.pause(0.01)

print('Done')
