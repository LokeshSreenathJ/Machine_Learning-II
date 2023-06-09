import sys
import argparse
import matplotlib.pyplot as plt
plt.rcParams["axes.grid"] = False
import matplotlib.image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import gzip
import struct
import array
from urllib.request import urlretrieve

import torch
from torch.utils.data import Dataset


BASE_URL = 'http://yann.lecun.com/exdb/mnist/'


# Helper functions and imports
def download(url, filename):
    if not os.path.exists('./data'):
        os.makedirs('./data')
    out_file = os.path.join('./data', filename)
    if not os.path.isfile(out_file):
        urlretrieve(url, out_file)


def mnist():
    def parse_labels(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

    for filename in ['train-images-idx3-ubyte.gz',
                     'train-labels-idx1-ubyte.gz',
                     't10k-images-idx3-ubyte.gz',
                     't10k-labels-idx1-ubyte.gz']:
        download(BASE_URL + filename, filename)

    train_images = parse_images('./data/train-images-idx3-ubyte.gz')
    train_labels = parse_labels('./data/train-labels-idx1-ubyte.gz')
    test_images  = parse_images('./data/t10k-images-idx3-ubyte.gz')
    test_labels  = parse_labels('./data/t10k-labels-idx1-ubyte.gz')
    return train_images, train_labels, test_images, test_labels


# Load and Prepare Data: Load the MNIST dataset, binarize the images, split into a training dataset 
# of 10000 images and a test set of 10000 images.
def load_mnist():
    partial_flatten = lambda x : np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    one_hot = lambda x, k: np.array(x[:,None] == np.arange(k)[None, :], dtype=int)
    train_images, train_labels, test_images, test_labels = mnist()
    train_images = partial_flatten(train_images) / 255.0
    test_images  = partial_flatten(test_images)  / 255.0
    train_labels = one_hot(train_labels, 10)
    test_labels = one_hot(test_labels, 10)
    N_data = train_images.shape[0]
    train_images = torch.from_numpy(np.round(train_images[0:10000])).float()
    train_labels = torch.from_numpy(train_labels[0:10000]).float()
    test_images = torch.from_numpy(np.round(test_images[0:10000])).float()
    test_labels = torch.from_numpy(test_labels[0:10000])
    return N_data, train_images, train_labels, test_images, test_labels


# Partition the training set into minibatches 
def batch_indices(iter, num_batches, batch_size):
    # iter: iteration index
    # num_batches: number of batches
    # batch_size: batch size
    idx = iter % num_batches
    return slice(idx * batch_size, (idx+1) * batch_size)


# write a function to reshape 784 array into a 28x28 image for plotting
def array_to_image(array):
    return np.reshape(np.array(array), [28, 28])


# concatenate the images for plotting
def concat_images(images, row, col, padding = 3):
    result = np.zeros((28*row+(row-1)*padding,28*col+(col-1)*padding))
    for i in range(row):
        for j in range(col):
            result[i*28+(i*padding):i*28+(i*padding)+28, j*28+(j*padding):j*28+(j*padding)+28] = images[i+j*row]
    return result

#Actual Model building starts from here...............

# The "encoder" model q(z|x)
class Encoder(nn.Module):
    def __init__(self, latent_dimension, hidden_units, data_dimension):
        super(Encoder, self).__init__()
        # Input:
        #   latent_dimension: the latent dimension of the encoder
        #   hidden_units: the number of hidden units
        
        self.fc1 = nn.Linear(data_dimension, hidden_units)
        self.fc2_mu = nn.Linear(hidden_units, latent_dimension)
        self.fc2_sigma = nn.Linear(hidden_units, latent_dimension)

    def forward(self, x):
        # Input: x input image [batch_size x data_dimension]
        # Output: parameters of a diagonal gaussian 
        #   mean : [batch_size x latent_dimension]
        #   variance : [batch_size x latent_dimension]

        hidden = torch.tanh(self.fc1(x))
        mu = self.fc2_mu(hidden)
        log_sigma_square = self.fc2_sigma(hidden)
        sigma_square = torch.exp(log_sigma_square)  
        return mu, sigma_square


# "decoder" Model p(x|z)
class Decoder(nn.Module):
    def __init__(self, latent_dimension, data_dimension,hidden_units=500):
        super(Decoder, self).__init__()
        # Input:
        #   latent_dimension: the latent dimension of the encoder
        #   hidden_units: the number of hidden units

        # fc1: a fully connected layer with 500 hidden units. 
        # fc2: a fully connected layer with 500 hidden units.
        self.fc1_dec = nn.Linear(latent_dimension, hidden_units)
        self.fc2_dec = nn.Linear(hidden_units, data_dimension)

    def forward(self, z):
        # input
        #   z: latent codes sampled from the encoder [batch_size x latent_dimension]
        # output 
        #   p: a tensor of the same size as the image indicating the probability of every pixel being 1 [batch_size x data_dimension]
        # The first layer is followed by a tanh non-linearity and the second layer by a sigmoid.
        hidden_dec = torch.tanh(self.fc1_dec(z))
        p = torch.sigmoid(self.fc2_dec(hidden_dec))
        return p


# VAE model
class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.latent_dimension = args.latent_dimension
        self.hidden_units =  args.hidden_units
        self.data_dimension = args.data_dimension
        self.resume_training = args.resume_training
        self.batch_size = args.batch_size
        self.num_epoches = args.num_epoches =200
        self.e_path = args.e_path
        self.d_path = args.d_path

        # load and pre-process the data
        N_data, self.train_images, self.train_labels, test_images, test_labels = load_mnist()

        # Instantiate the encoder and decoder models 
        self.encoder = Encoder(self.latent_dimension, self.hidden_units, self.data_dimension)
        self.decoder = Decoder(self.latent_dimension, self.hidden_units, self.data_dimension)

        # Load the trained model parameters
        if self.resume_training:
            self.encoder.load_state_dict(torch.load(self.e_path))
            self.decoder.load_state_dict(torch.load(self.d_path))

    # Sample from Diagonal Gaussian z~N(μ,σ^2 I) 
    @staticmethod
    def sample_diagonal_gaussian(mu, sigma_square):
        # Inputs:
        #   mu: mean of the gaussian [batch_size x latent_dimension]
        #   sigma_square: variance of the gaussian [batch_size x latent_dimension]
        # Output:
        #   sample: from a diagonal gaussian with mean mu and variance sigma_square [batch_size x latent_dimension]
        sigma = (torch.sqrt(sigma_square)).reshape(mu.shape[0],mu.shape[1])
        sample = mu + torch.randn(size = (mu.shape[0],mu.shape[1]))*sigma        
        return sample

    # Sampler from Bernoulli
    @staticmethod
    def sample_Bernoulli(p):
        # Input: 
        #   p: the probability of pixels labeled 1 [batch_size x data_dimension]
        # Output:
        #   x: pixels'labels [batch_size x data_dimension], type should be torch.float32
        x = torch.bernoulli(p)        
        return x


    # Compute Log-pdf of z under Diagonal Gaussian N(z|μ,σ^2 I)
    @staticmethod
    def logpdf_diagonal_gaussian(z, mu, sigma_square):
        # Input:
        #   z: sample [batch_size x latent_dimension]
        #   mu: mean of the gaussian distribution [batch_size x latent_dimension]
        #   sigma_square: variance of the gaussian distribution [batch_size x latent_dimension]
        # Output:
        #    logprob: log-probability of a diagomnal gaussian [batch_size]
        from torch.distributions.multivariate_normal import MultivariateNormal
        logprob = torch.zeros((mu.shape[0]))
        for i in range(mu.shape[0]):
            dist = MultivariateNormal(mu[i], sigma_square[i]*torch.eye((mu.shape[1])))
            logprob[i] = dist.log_prob(z[i])        
        return logprob

    # Compute log-pdf of x under Bernoulli 
    @staticmethod
    def logpdf_bernoulli(x, p):
        # Input:
        #   x: samples [batch_size x data_dimension]
        #   p: the probability of the x being labeled 1 (p is the output of the decoder) [batch_size x data_dimension]
        # Output:
        #   logprob: log-probability of a bernoulli distribution [batch_size]
        logprob = (x*torch.log(p) + (1-x)*torch.log(1-p)).sum(axis =1)        
        return logprob
    
    # Sample z ~ q(z|x)
    def sample_z(self, mu, sigma_square):
        # input:
        #   mu: mean of the gaussian [batch_size x latent_dimension]
        #   sigma_square: variance of the gaussian [batch_size x latent_dimension]
        # Output:
        #   zs: samples from q(z|x) [batch_size x latent_dimension] 
        zs = self.sample_diagonal_gaussian(mu, sigma_square)
        return zs 


    # Variational Objective
    def elbo_loss(self, sampled_z, mu, sigma_square, x, p):
        # Inputs
        #   sampled_z: samples z from the encoder [batch_size x latent_dimension]
        #   mu:
        #   sigma_square: parameters of q(z|x) [batch_size x latent_dimension]
        #   x: data samples [batch_size x data_dimension]
        #   p: the probability of a pixel being labeled 1 [batch_size x data_dimension]
        # Output
        #   elbo: the ELBO loss (scalar)

        # log_q(z|x) logprobability of z under approximate posterior N(μ,σ)
        log_q = self.logpdf_diagonal_gaussian(sampled_z, mu, sigma_square)
        
        # log_p_z(z) log probability of z under prior
        z_mu = torch.FloatTensor([0]*self.latent_dimension).repeat(sampled_z.shape[0], 1)
        z_sigma = torch.FloatTensor([1]*self.latent_dimension).repeat(sampled_z.shape[0], 1)
        log_p_z = self.logpdf_diagonal_gaussian(sampled_z, z_mu, z_sigma)

        # log_p(x|z) - conditional probability of data given latents.
        log_p = self.logpdf_bernoulli(x, p)
        elbo = (log_p+log_p_z-log_q).mean()
        return elbo


    def train(self):
        
        # Set-up ADAM optimizer
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        adam_optimizer = optim.Adam(params)

        # Train for ~200 epochs 
        num_batches = int(np.ceil(len(self.train_images) / self.batch_size))
        num_iters = 200 * num_batches
        
        for i in range(num_iters):
            x_minibatch = self.train_images[batch_indices(i, num_batches, self.batch_size),:]
            adam_optimizer.zero_grad()

            mu, sigma_square = self.encoder(x_minibatch)
            zs = self.sample_z(mu, sigma_square)
            p = self.decoder(zs)
            elbo = self.elbo_loss(zs, mu, sigma_square, x_minibatch, p)
            total_loss = -elbo
            total_loss.backward()
            adam_optimizer.step()

            if i%100 == 0:
                print("Epoch: " + str(i//num_batches) + ", Iter: " + str(i) + ", ELBO:" + str(elbo.item()))

        # Save Optimized Model Parameters
        torch.save(self.encoder.state_dict(), self.e_path)
        torch.save(self.decoder.state_dict(), self.d_path)


    # Generate digits using the VAE
    def visualize_data_space(self):
        # TODO: Sample 10 z from prior 
        z_10= torch.randn(10).reshape(1,10)
        images = torch.zeros((20,self.data_dimension))
        sample_x = torch.zeros((10,self.data_dimension))
        for i in range(10):
            output_dec = self.decoder(z_10[i])
            sample_x[i] = sample_Bernoulli(output_dec)
            images[10+i] = array_to_image(sample_Bernoulli(output_dec).reshape(-1,self.data_dimension))
            output_img = output_dec.reshape(-1,self.data_dimension)
            images[i] = array_to_image(output_img[0])
            
            plt.imshow(images[i])
        concatenated_image= concat_images(images, 10, 2, padding = 3)
        plt.savefig('concatenated_image.png')
        
    # Produce a scatter plot in the latent space, where each point in the plot will be the mean vector 
    # for the distribution $q(z|x)$ given by the encoder. Further, we will colour each point in the plot 
    # by the class label for the input data. Each point in the plot is colored by the class label for 
    # the input data.
    # The latent space should have learned to distinguish between elements from different classes, even though 
    # we never provided class labels to the model!
    def visualize_latent_space(self):
        z_mean = torch.zeros((self.train_images.shape[0],latent_dimension))
        for i in range(self.train_images.shape[0]):
            z_mean[i],_ = self.encoder(self.train_images[i])
            
         
        labels = torch.zeros((self.train_labels.shape[0],1))
        for i in range(self.train_labels.shape[0]):
            labels[i]= torch.Tensor(torch.where(self.train_labels[i]==1))

        # Colour each point depending on the class label 
        plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
        plt.xlabel('Latent dimension 1')
        plt.ylabel('Latent dimension 2')
        plt.show()
        plt.savefig("latentspace_image.png")


    # Function which gives linear interpolation z_α between za and zb
    @staticmethod
    def interpolate_mu(mua, mub, alpha = 0.5):
        return alpha*mua + (1-alpha)*mub


    # A common technique to assess latent representations is to interpolate between two points.
    # Here we will encode 3 pairs of data points with different classes.
    # Then we will linearly interpolate between the mean vectors of their encodings. 
    # We will plot the generative distributions along the linear interpolation.
    def visualize_inter_class_interpolation(self):
        labels = torch.zeros((self.train_labels.shape[0],1))
        for i in range(self.train_labels.shape[0]):
            labels[i]= torch.Tensor(torch.where(self.train_labels[i]==1))
        label_0_idx = torch.where(labels ==0)[0][:2]
        label_1_idx = torch.where(labels==1)[0][:2]
        label_2_idx = torch.where(labels==2)[0][:2]
        sample_1 = self.train_images[label_0_idx]
        sample_2 = self.train_images[label_1_idx]
        sample_3 = self.train_images[label_2_idx]
        
        mean_sample1,_ = self.encoder(sample_1)
        mean_sample2,_ = self.encoder(sample_2)
        mean_sample3,_ = self.encoder(sample_3)

        lin_int_1 = interpolate_mu(mean_sample1[0], mean_sample1[1], alpha = 0.5)
        lin_int_2 = interpolate_mu(mean_sample2[0], mean_sample2[1], alpha = 0.5)
        lin_int_3 = interpolate_mu(mean_sample3[0], mean_sample3[1], alpha = 0.5)
        z_3= torch.empty([3]).normal_(mean=0,std=1)
        lin_images = torch.zeros((3,self.data_dimension))
        lin_images[0] = self.decoder(z_3[0]).reshape(-1,self.data_dimension)[0]
        lin_images[1] = self.decoder(z_3[1]).reshape(-1,self.data_dimension)[0]
        lin_images[2] = self.decoder(z_3[2]).reshape(-1,self.data_dimension)[0]
        plt.imshow(lin_images[0])
        plt.imshow(lin_images[1])
        plt.imshow(lin_images[2])
        # Concatenate these plots into one figure
        
        
        
      

def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--e_path', type=str, default="./e_params.pkl", help='Path to the encoder parameters.')
    parser.add_argument('--d_path', type=str, default="./d_params.pkl", help='Path to the decoder parameters.')
    parser.add_argument('--hidden_units', type=int, default=500, help='Number of hidden units of the encoder and decoder models.')
    parser.add_argument('--latent_dimension', type=int, default='2', help='Dimensionality of the latent space.')
    parser.add_argument('--data_dimension', type=int, default='784', help='Dimensionality of the data space.')
    parser.add_argument('--resume_training', action='store_true', help='Whether to resume training')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--num_epoches', type=int, default=200, help='Number of epochs for training.')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size.')

    args = parser.parse_args()
    return args


def main():
    
    # read the function arguments
    args = parse_args()

    # set the random seed 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # train the model 
    vae = VAE(args)
    vae.train()

    # visualize the latent space
    vae.visualize_data_space()
    vae.visualize_latent_space()
    vae.visualize_inter_class_interpolation()


