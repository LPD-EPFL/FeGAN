# coding: utf-8
###
 # @file   md-gan.py
 # @author Arsany Guirguis  <arsany.guirguis@epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright (c) 2020 Arsany Guirguis.
 #
 # Permission is hereby granted, free of charge, to any person obtaining a copy
 # of this software and associated documentation files (the "Software"), to deal
 # in the Software without restriction, including without limitation the rights
 # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 # copies of the Software, and to permit persons to whom the Software is
 # furnished to do so, subject to the following conditions:
 #
 # The above copyright notice and this permission notice shall be included in all
 # copies or substantial portions of the Software.
 #
 # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 # SOFTWARE.
 #
 # @section DESCRIPTION
 #
 # Implementation of the paper: "MD-GAN: Multi-Discriminator Generative Adversarial Networks for Distributed Datasets"
 # which is authored by Corentin Hardy, Erwan Le Merrer, Bruno Sericola.
 # Link: https://arxiv.org/abs/1811.03850
 # The original paper did not publish an open-source code. This is our implementation, which is done primarily for comparison.
###
#!/usr/bin/env python3
import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
#DIST
import torch.distributed as dist
import torch.multiprocessing as multiprocessing
from torch.multiprocessing import Process
from datasets import DatasetManager
from fid_score import *
from inception import *
from time import sleep, time
import random
import sys
from scipy import stats
from queue import Queue

multiprocessing.set_sharing_strategy('file_system')
torch.autograd.set_detect_anomaly(True)
# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Linear(128 * ds_size ** 2, 1)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


#DIST
""" Gradient averaging. """
def average_models(model, group=None, choose_r0=True, weights=None, elapsed_time=None):
    for param in model.parameters():
        if rank == 0:				#If rank=0 is not in included in this round, put zeros instead
            param.data = torch.zeros(param.size()).cuda() if cuda else torch.zeros(param.size())
        dist.reduce(param.data, dst=0, op=dist.ReduceOp.SUM)
        param.data /= (size - 1)

#DIST
""" Model broadcast. """
def broadcast_model(model, group=None, elapsed_time=None):
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

all_groups = []
all_groups_np = []
choose_r = []
fl_round = -1
def init_groups(size):
    """ 
	Initialization of all distributed groups for the whole training process. We do this in advance so as not to hurt the performance of training.
	The server initializes the group and send it to all workers so that everybody can agree on the working group at some round.
	Args
		size		The total number of machines in the current setup
    """
    global all_groups
    all_groups = []
    for i in range(size-1):
        group = dist.new_group([0,i+1])
        all_groups.append(group)

#DIST
def gather_lbl_count(lbl_count):
    """ 
	This function gathers all labels counts from all workers at the server.
	Args:
		lbl_count: array of frequency of samples of each class at the current worker
	returns:
		workers_classes: array of arrays of labels counts of each class at the server
    """
    gather_list = []
    if rank == 0:
        gather_list = [torch.zeros(len(lbl_count)) for _ in range(size)]
    dist.gather(torch.cuda.FloatTensor(lbl_count) if cuda else torch.FloatTensor(lbl_count), gather_list, dst=0)
    res = [count_list.cpu().detach().numpy() for count_list in gather_list]
    return res

#DIST
rat_per_class=[]
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
def run(rank, size):
    global fl_round
    global rat_per_class
    # Minimizes MSE
    adversarial_loss = torch.nn.MSELoss()
    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Configure data loader
    same_data = False			#set to True if all devices are required to hold the same data
    if same_data:
        os.makedirs("../data/mnist", exist_ok=True)
        train_set = torch.utils.data.DataLoader(
            datasets.MNIST(
                "../data/mnist",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=opt.batch_size,
            shuffle=True,
        )
    else:
        manager = DatasetManager(opt.model, opt.batch_size, opt.img_size, size-1, size, rank, opt.iid, 1)
        train_set, _ = manager.get_train_set(opt.magic_num)

    init_groups(size)
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    #For FID calculations
    if rank == 0:
        fic_model = InceptionV3()
        if cuda:
            fic_model = fic_model.cuda()
        test_set = manager.get_test_set()
        for i,t in enumerate(test_set):
            test_imgs = t[0].cuda() if cuda else t[0]
            test_labels = t[1]

    # ----------
    #  Training
    # ----------
    #DIST
    elapsed_time = time()
    num_batches=0		#This variable acts as a global state variable to sync. between workers and the server
    done_round = True
    group = None
    #The following hack (4 lines) is written to run actually the number of runs that the user is aiming for....because of the skewness of data, the actual number of epochs that would run could be less than that the user is estimating...These few lines solve this issue
    est_len = 50000 // (size * opt.batch_size)		#Given a dataset of 50,000 imgaes, the estimated number of iterations to dataset is 50000/unm_workers
    act_len = len(train_set)
    if act_len < est_len:
        opt.n_epochs = int(opt.n_epochs * (est_len/act_len))
    imgs = []
    for i, (tmps,_) in enumerate(train_set):
        imgs=tmps
        break
    for epoch in range(opt.n_epochs):
        broadcast_model(generator, elapsed_time=elapsed_time)
        fl_round+=1
        num_batches+=1
            # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        #HINT: training the generator is not required on the server, yet PyTorch requires it. It does not affect the runtime anyway

            # -----------------
            #  Train Generator
            # -----------------
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        temp = generator(z)
        if rank == 0:		#MD-GAN trains the generator only on the server
            optimizer_G.zero_grad()
            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
            # Generate a batch of images
            X_g = generator(z)
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
            # Generate a batch of images
            X_d = generator(z)
            for n in range(size-1):
                # Sample noise as generator input
                # Generate a batch of images
                dist.broadcast(tensor=X_g, src=0, group=all_groups[n])
                # Generate a batch of images
                dist.broadcast(tensor=X_d, src=0,group=all_groups[n])

        else: #First, workers receive generated batches by the server
            X_g = torch.zeros(temp.size())
            X_d = torch.zeros(temp.size())
            dist.broadcast(tensor=X_g, src=0, group=all_groups[rank-1])
            dist.broadcast(tensor=X_d, src=0, group=all_groups[rank-1])
            if cuda:
                X_g = X_g.cuda()
                X_d = X_d.cuda()

            # Loss measures generator's ability to fool the discriminator
        if rank == 0:
            d_gen = discriminator(temp)
            g_loss = adversarial_loss(d_gen, valid)
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

        if rank != 0:
            L = 12
            for iter, (imgs_t, _) in enumerate(train_set):
                real_imgs = Variable(imgs_t.type(Tensor))
                if real_imgs.size()[0] != opt.batch_size:			#To avoid mismatch problems
                    continue
                optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = adversarial_loss(discriminator(real_imgs), valid)
                fake_loss = adversarial_loss(discriminator(X_d.detach()), fake)
                d_loss = 0.5 * (real_loss + fake_loss)
                d_loss.backward()
                optimizer_D.step()
                if iter == L-1:
                    break

            optimizer_G.zero_grad()
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
            X_g = generator(z)
            g_loss = adversarial_loss(discriminator(X_g), valid)
            g_loss.backward()
            optimizer_G.step()
        average_models(generator, elapsed_time=elapsed_time)
        del X_g
        del X_d

        #Print stats and generate images only if this is the server
        batches_done = fl_round
        if rank == 0 and fl_round%20 == 0:
            print(
                "Rank %d [Epoch %d/%d] [Batch %d/%d] time %f"
                % (rank, epoch, opt.n_epochs, i, len(train_set), time() - elapsed_time),
                end = ' ' if epoch != 0 else '\n'
            )

            fid_z = Variable(Tensor(np.random.normal(0, 1, (opt.fid_batch, opt.latent_dim))))
            gen_imgs = generator(fid_z)
            mu_gen, sigma_gen = calculate_activation_statistics(gen_imgs, fic_model)
            mu_test, sigma_test = calculate_activation_statistics(test_imgs[:opt.fid_batch], fic_model)
            fid = calculate_frechet_distance(mu_gen, sigma_gen, mu_test, sigma_test)
            print("FL-round {} FID Score: {}".format(fl_round, fid))
            sys.stdout.flush()
#DIST
def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = master
    os.environ['MASTER_PORT'] = port
    dist.init_process_group(backend, rank=rank, world_size=size)
    print("Process {} initialized group".format(rank))
    fn(rank, size)

os.makedirs("images-dist", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=50, help="size of the batches (named B in FL notations)")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=1000, help="number of iterations to calculate the FID.")
#DIST
parser.add_argument("--model", type=str, default='mnist', help="model to train")
parser.add_argument("--local_steps", type=int, default=100, help="number of local steps to be executed in each worker before sending to the server (named E in FL notations).")
parser.add_argument("--frac_workers", type=float, default=0.1, help="fraction of workers that participate in each round computation (named C in FL notations).")
parser.add_argument("--fid_batch", type=int, default=4000, help="number of samples used to evaluate the progress of the GAN (using the FID score).")
parser.add_argument("--rank", type=int, default=-1, help="Rank of this node in the distributed setup.")
parser.add_argument("--size", type=int, default=-1, help="Total number of machines in this experiment.")
parser.add_argument("--iid", type=int, default=1, help="Determines whether data should be distributed in an iid fashion to all workers or not. Takes only 0 or 1 as a value.")
parser.add_argument("--port", type=str, default='29500', help="Port number of the master....required for connections from everybody.")
parser.add_argument("--master", type=str, default='igrida-abacus9', help="The master hostname...should be known by everybody.")
#parser.add_argument("--bench", type=int, default=1, help="If set, time taken by each step is printed.")
parser.add_argument("--weight_scheme", type=str, default='exp', help="Determines the weighting technique used. Currently existing schemes are dirac, linear, and exp.")
parser.add_argument("--magic_num", type=int, default=5000, help="Temporary value that determines the maximum number of samples should be with each class.")
opt = parser.parse_args()
opt.n_epochs *= int((1-opt.frac_workers)*10)		#This is to cope up with the workers that remain idle in fl rounds...to achieve fair comparison with the single-machine implementation
print(opt)
port = opt.port
master = opt.master
#DIST
size = opt.size
rank = opt.rank
model = opt.model
assert opt.iid == 0 or opt.iid == 1
import socket
hostname = socket.gethostname()
if hostname == master:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' if rank==0 else str((rank%1) + 1)		#%1 should be replaced by %(num_gpus-1)...now we are testing with 2 GPUs per machine
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank%2) 			#Other machines can use both GPUs freely..only the master is allowed to take one GPU exclusively

cuda = True if torch.cuda.is_available() else False
print("Using Cuda?\n ", cuda, "Hostname: ", hostname)

init_processes(rank,size, run)
