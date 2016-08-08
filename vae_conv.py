
# coding: utf-8

# # Variational Autoencoder in TensorFlow

# The main motivation for this post was that I wanted to get more experience with both [Variational Autoencoders](http://arxiv.org/abs/1312.6114) (VAEs) and with [Tensorflow](http://www.tensorflow.org/). Thus, implementing the former in the latter sounded like a good idea for learning about both at the same time. This post summarizes the result.
# 
# Note: The post was updated on December 7th 2015:
#   * a bug in the computation of the latent_loss was fixed (removed an erroneous factor 2). Thanks Colin Fang for pointing this out.
#   * Using a Bernoulli distribution rather than a Gaussian distribution in the generator network
# 
# Let us first do the necessary imports, load the data (MNIST), and define some helper functions.

# In[1]:

import numpy as np
import tensorflow as tf
import prettytensor as pt
from deconv import deconv2d
import os
# import matplotlib.pyplot as plt
# get_ipython().magic(u'matplotlib inline')

np.random.seed(0)
tf.set_random_seed(0)

# Load MNIST data in a format suited for tensorflow.
# The script input_data is available under this URL:
# https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/g3doc/tutorials/mnist/input_data.py
import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_samples = mnist.train.num_examples
print n_samples


# Based on this, we define now a class "VariationalAutoencoder" with a [sklearn](http://scikit-learn.org)-like interface that can be trained incrementally with mini-batches using partial_fit. The trained model can be used to reconstruct unseen input, to generate new samples, and to map inputs to the latent space.

# In[2]:

class VariationalAutoencoder(object):
    """ Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.
    
    This implementation uses probabilistic encoders and decoders using Gaussian 
    distributions and  realized by multi-layer perceptrons. The VAE can be learned
    end-to-end.
    
    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """
    def __init__(self, params, network_architecture, transfer_fct=tf.nn.softplus, 
                 learning_rate=0.001):
        self.network_architecture = network_architecture
        self.params = params
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = self.params['batch_size']
        self.hidden_size = self.params["n_z"]
        self.train_writer = tf.train.SummaryWriter(params['summary_folder'])
        
        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])
        
        # Create autoencoder network
        self._create_network()
        # Define loss function based variational upper-bound and 
        # corresponding optimizer
        self._create_loss_optimizer()
        
        # Initializing the tensor flow variables
        init = tf.initialize_all_variables()

        # Launch the session
        global sess
        sess = tf.InteractiveSession()
        sess.run(init)
        global saver
        saver = tf.train.Saver()
        if not os.path.exists(params['model_folder']):
            os.makedirs(params['model_folder'])
            print("+++ Created snapshot folder path: %s" % params['model_folder'])
    
    def _create_network(self):
        # Use recognition network to determine mean and 
        # (log) variance of Gaussian distribution in latent
        # space
        with tf.variable_scope("model", reuse=False) as scope:
            with pt.defaults_scope(activation_fn=tf.nn.elu,
                                   batch_normalize=True,
                                   learned_moments_update_rate=0.0003,
                                   variance_epsilon=0.001,
                                   scale_after_normalization=True):
                hidden_tensor, self.z_mean, self.z_log_sigma_sq =                     self._recognition_network_conv(self.x)

                # Draw one sample z from Gaussian distribution
                n_z = self.params["n_z"]
                eps = tf.random_normal((self.batch_size, n_z), 0, 1, 
                                       dtype=tf.float32)
                # z = mu + sigma*epsilon
                self.z = tf.add(self.z_mean, 
                                tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

                # Use generator to determine mean of
                # Bernoulli distribution of reconstructed input
                self.x_reconstr_mean =                     self._generator_network_conv(self.z)
    
    def _recognition_network_conv(self, input_tensor):
        hidden_tensor = (pt.wrap(input_tensor).
            reshape([self.batch_size, 28, 28, 1]).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 128, edges='VALID').
            dropout(0.9).
            flatten().
            fully_connected(self.params['n_z'] * 2, activation_fn=None)).tensor
        z_mean = hidden_tensor[:, :self.params['n_z']]
        z_log_sigma_sq = tf.sqrt(tf.exp(hidden_tensor[:, self.params['n_z']:]))
        return (hidden_tensor, z_mean, z_log_sigma_sq)

    
    def _generator_network_conv(self, input_sample):
#         epsilon = tf.random_normal([self.batch_size, self.params['hidden_size']])
#         if input_tensor is None:
#             mean = None
#             stddev = None
#             input_sample = epsilon
#         else:
#             mean = input_tensor[:, :self.params['hidden_size']]
#             stddev = tf.sqrt(tf.exp(input_tensor[:, self.params['hidden_size']:]))
#             input_sample = mean + epsilon * stddev
        return (pt.wrap(input_sample).
                reshape([self.batch_size, 1, 1, self.hidden_size]).
                deconv2d(3, 128, edges='VALID').
                deconv2d(5, 64, edges='VALID').
                deconv2d(5, 32, stride=2).
                deconv2d(5, 1, stride=2, activation_fn=tf.nn.sigmoid).
                flatten()).tensor
            
    def _create_loss_optimizer(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution 
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluatio of log(0.0)
        self.reconstr_loss =             -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean)
                           + (1-self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean),
                           1)
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence 
        ##    between the distribution in latent space induced by the encoder on 
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        self.latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq 
                                           - tf.square(self.z_mean) 
                                           - tf.exp(self.z_log_sigma_sq), 1)
        self.cost = tf.reduce_mean(self.reconstr_loss + self.params['reweight'] * self.latent_loss)   # average over batch
        # Use ADAM optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1.0).minimize(self.cost)
#         self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        
    def partial_fit(self, X):
        """Train model based on mini-batch of input data.
        
        Return cost of mini-batch.
        """
        summary_loss = tf.scalar_summary('loss/loss', self.cost)
        summary_loss_reconstr = tf.scalar_summary('loss/rec_loss', tf.reduce_mean(self.reconstr_loss))
        summary_loss_latent = tf.scalar_summary('loss/vae_loss', tf.reduce_mean(self.latent_loss))
        summaries = [summary_loss, summary_loss_reconstr, summary_loss_latent]
        global merged_summaries
        merged_summaries = tf.merge_summary(summaries)
        opt, cost, merged = sess.run((self.optimizer, self.cost, merged_summaries), 
                                  feed_dict={self.x: X})
        return (cost, merged)
    
    def transform(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return sess.run(self.z_mean, feed_dict={self.x: X})
    
    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.
        
        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent 
        space.        
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["n_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return sess.run(self.x_reconstr_mean, 
                             feed_dict={self.z: z_mu})
    
    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        return sess.run(self.x_reconstr_mean, 
                             feed_dict={self.x: X})


# In general, implementing a VAE in tensorflow is relatively straightforward (in particular since we don not need to code the gradient computation). A bit confusing is potentially that all the logic happens at initialization of the class (where the graph is generated), while the actual sklearn interface methods are very simple one-liners.
# 
# We can now define a simple fuction which trains the VAE using mini-batches:

# In[3]:

def train(params, network_architecture, learning_rate=0.001, training_epochs=10, display_step=5):
    global vae
    vae = VariationalAutoencoder(params, network_architecture, 
                                 learning_rate=learning_rate)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / vae.batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, _ = mnist.train.next_batch(vae.batch_size)

            # Fit training using batch data
            cost, merged_summary = vae.partial_fit(batch_xs)
            # Compute average loss
#             avg_cost += cost / n_samples * vae.batch_size
            if i % 10 == 0:
                vae.train_writer.add_summary(merged_summary, epoch * total_batch + i)
            if i % 100 == 0:
                vae.train_writer.flush()
                print "Epoch:", '%04d' % (epoch+1), " batch:", '%04d' % (i+1),                 "cost=", "{:.9f}".format(cost)
        save_path = saver.save(sess,vae.params['model_folder'] + '/model.ckpt')
        print("--> Model saved in file: %s" % save_path)
    return vae

def restore(params, network_architecture):
    global vae
    vae = VariationalAutoencoder(params, network_architecture, 
                                 learning_rate=learning_rate)
    saver.restore(sess,vae.params['model_folder'] + '/model.ckpt')
    print("--> Model restored from file: %s" % save_path)


# ## Illustrating reconstruction quality

# We can now train a VAE on MNIST by just specifying the network topology. We start with training a VAE with a 20-dimensional latent space.

# In[ ]:

network_architecture =     dict(n_input=784) # MNIST data input (img shape: 28*28)
params =     dict(summary_folder='./summary/test1_conv', 
         model_folder='./snapshot/test1_conv',
         reweight=1, 
         batch_size=100, 
         n_z=20)  # dimensionality of latent space)

vae = train(params, network_architecture, training_epochs=2, learning_rate=0.01, display_step=2)


# Based on this we can sample some test inputs and visualize how well the VAE can reconstruct those. In general the VAE does really well.

# In[6]:

x_sample = mnist.test.next_batch(100)[0]
print x_sample.shape
x_reconstruct = vae.reconstruct(x_sample)

plt.figure(figsize=(8, 12))
for i in range(5):

    plt.subplot(5, 2, 2*i + 1)
    plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1)
    plt.title("Test input")
    plt.colorbar()
    plt.subplot(5, 2, 2*i + 2)
    plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1)
    plt.title("Reconstruction")
    plt.colorbar()
plt.tight_layout()


# ## Illustrating latent space

# Next, we train a VAE with 2d latent space and illustrates how the encoder (the recognition network) encodes some of the labeled inputs (collapsing the Gaussian distribution in latent space to its mean). This gives us some insights into the structure of the learned manifold (latent space)

# In[ ]:

network_architecture =     dict(n_hidden_recog_1=500, # 1st layer encoder neurons
         n_hidden_recog_2=500, # 2nd layer encoder neurons
         n_hidden_gener_1=500, # 1st layer decoder neurons
         n_hidden_gener_2=500, # 2nd layer decoder neurons
         n_input=784, # MNIST data input (img shape: 28*28)
         n_z=2)  # dimensionality of latent space

vae_2d = train(network_architecture, training_epochs=75)


# In[ ]:

x_sample, y_sample = mnist.test.next_batch(5000)
z_mu = vae_2d.transform(x_sample)
plt.figure(figsize=(8, 6)) 
plt.scatter(z_mu[:, 0], z_mu[:, 1], c=np.argmax(y_sample, 1))
plt.colorbar()


# An other way of getting insights into the latent space is to use the generator network to plot reconstrunctions at the positions in the latent space for which they have been generated:

# In[ ]:

nx = ny = 20
x_values = np.linspace(-3, 3, nx)
y_values = np.linspace(-3, 3, ny)

canvas = np.empty((28*ny, 28*nx))
for i, yi in enumerate(x_values):
    for j, xi in enumerate(y_values):
        z_mu = np.array([[xi, yi]])
        x_mean = vae_2d.generate(z_mu)
        canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = x_mean[0].reshape(28, 28)

plt.figure(figsize=(8, 10))        
Xi, Yi = np.meshgrid(x_values, y_values)
plt.imshow(canvas, origin="upper")
plt.tight_layout()


# ## Summary
# In summary, tensorflow is well suited to rapidly implement a prototype of machine learning models like VAE. The resulting code could be easily executed on GPUs as well (requiring just that tensorflow with GPU support was installed). VAE allows learning probabilistic encoders and decoders of data in an end-to-end fashion.

# In[ ]:

get_ipython().magic(u'load_ext watermark')
get_ipython().magic(u'watermark -a "Jan Hendrik Metzen" -d -v -m -p numpy,scikit-learn')

