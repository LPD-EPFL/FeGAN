# coding: utf-8
###
 # @file   dist-dcgan.py
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
 # Running the DCGAN architecture in a distributed fashion following the FeGAN model.
 # This file is based on the implementation of DCGAN for the centralized setup (check dcgan.py).
###
#!/usr/bin/env python
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
# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
random.seed(manualSeed)
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( opt.latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, opt.channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(opt.channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
#DIST
""" Gradient averaging. """
def average_models(model, group=None, choose_r0=True, weights=None):
    global fl_round
    global rat_per_class
    gp_size = len(all_groups_np[fl_round%len(all_groups)])
    if rank == 0 and opt.weight_avg and weights is not None:
        cur_gp = all_groups_np[fl_round%len(all_groups)]
        if opt.weight_scheme == 'exp':
            e_w = [np.exp(w.item()) for w in weights]               #Getting e^w for each w in weights (w here is the success rate of devices' generators)
        else:
            e_w = [w.item() for w in weights]

        e_w = np.array(e_w)
        if not choose_r0:
            e_w/= sum(e_w[1:])
        else:
            e_w/= sum(e_w)
        if opt.weight_scheme == 'dirac':
            e_w = [0 if w < 0.5 else w for w in e_w]		#The threshold here is 0.5
            #Reweighting after removing the harmful/useless updates (could work as a simulation to taking the most forgiving updates)
            if not choose_r0:
                e_w/= sum(e_w[1:])
            else:
                e_w/= sum(e_w)

    for param in model.parameters():
        if rank == 0 and not choose_r0:				#If rank=0 is not in included in this round, put zeros instead
            param.data = torch.zeros(param.size()).cuda() if cuda else torch.zeros(param.size())
        if not opt.weight_avg or weights is None:
            dist.reduce(param.data, dst=0, op=dist.ReduceOp.SUM, group=group)
            param.data /= (gp_size if choose_r0 else gp_size - 1)
        else:
            gather_list = []
            if rank == 0:
                gather_list = [torch.zeros(param.size()).cuda() if cuda else torch.zeros(param.size()) for _ in range(gp_size)]
            dist.gather(param.data, gather_list, dst=0, group=group)
            if rank == 0:
                param.data = torch.zeros(param.size()).cuda() if cuda else torch.zeros(param.size())
                for w,t in zip(e_w,gather_list):
                    param.data+= t*w

#DIST
""" Model broadcast. """
def broadcast_model(model, group=None):
    for param in model.parameters():
        dist.broadcast(param.data, src=0, group=group)

all_groups = []
all_groups_np = []
choose_r = []
fl_round = -1
def init_groups(size, cls_freq_wrk):
    """
	Initialization of all distributed groups for the whole training process. We do this in advance so as not to hurt the performance of training.
	The server initializes the group and send it to all workers so that everybody can agree on the working group at some round.
	Args
		size		The total number of machines in the current setup
		cls_freq_wrk	The frequency of samples of each class at each worker. This is used when the "sample" option is chosen. Otherwise, random sampling is applied and this parameter is not used. 
    """
    global all_groups
    global all_groups_np
    global choose_r
    NUM_CLASSES = 200 if opt.model == 'imagenet' else 10
    all_groups = []
    all_groups_np = []
    choose_r = []
    done = False
    gp_size = int(opt.frac_workers*(size))
    #If opt.sample is set, use the smart sampling, i.e., based on frequency of samples of each class at each worker. Otherwise, use random sampling
    if opt.sample:
        #2D array that records if class i exists at worker j or not
        wrk_cls = [[False for i in range(NUM_CLASSES)] for j in range(size)]
        cls_q = [Queue(maxsize=size) for _ in range(NUM_CLASSES)]
        for i,cls_list in enumerate(cls_freq_wrk):
            wrk_cls[i] = [True if freq != 0 else False for freq in cls_list]
        for worker,class_list in enumerate(reversed(wrk_cls)):
            for cls,exist in enumerate(class_list):
                if exist:
                    cls_q[cls].put(size - worker-1)
	#This array counts the number of samples (per class) taken for training so far. The algorithm will try to make the numbers in this array as equal as possible
        taken_count = [0 for i in range(NUM_CLASSES)]
    while not done:
        if not opt.sample or rank != 0:			#It does not matter what other workers to.....only the server is required to create correct groups
            g = random.sample(range(0, size), gp_size)
        else:
            visited = [False for i in range(size)]	#makes sure that we take any worker only once in the group
            g = []
            for idx in range(gp_size):
                #Choose class (that is minimum represnted so far)...using "taken_count" array
                cls = np.where(taken_count == np.amin(taken_count))[0][0]
                assert cls >= 0 and cls <= len(taken_count)
                #Choose a worker to represnt that class...using wrk_cls and visited array
                done_q = False
                count = 0
                sys.stdout.flush()
                while not done_q:
                    wrkr = cls_q[cls].get()
                    assert wrk_cls[wrkr][cls]
                    if not visited[wrkr] and wrk_cls[wrkr][cls]:
                        #Update the state: taken_count and visited
                        g.append(wrkr)
                        taken_count += cls_freq_wrk[wrkr]
                        visited[wrkr] = True
                        done_q = True
                    cls_q[cls].put(wrkr)
                    count+=1
                    if count == size:	#Such an optimal assignment does not exist
                        done_q = True
        choose_r0 = False
        if rank == 0:
            if 0 in g:
                choose_r0 = True
            else:
                choose_r0 = False
            choose_r.append(choose_r0)
        g.append(0)
        assert len(g) > 1, "Number of sampled nodes per FL round is too low; consider increasing the number of nodes in the deployment or the fraction of chosen ndoes per round"
        if cuda:
            g = torch.cuda.FloatTensor(g)
        else:
            g = torch.FloatTensor(g)
        dist.broadcast(g,src=0)
        g = g.cpu().numpy().tolist()
        if g.count(0) > 1:                              #Make sure there is at most one occurance of "0" in the list of group members
            g.remove(0)
        try:
            group = dist.new_group(g)
        except Exception as e:
            done = True
        all_groups_np.append(np.sort(g))
        all_groups.append(group)
        if len(all_groups) > 50:			#for memory constraints
            done = True
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
        gather_list = [torch.zeros(len(lbl_count)).cuda() if cuda else torch.zeros(len(lbl_count)) for _ in range(size)]
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
    NUM_CLASSES = 200 if opt.model == 'imagenet' else 10
    criterion = torch.nn.BCELoss()
    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(opt.batch_size, opt.latent_dim, 1, 1)
    if cuda:
        fixed_noise = fixed_noise.cuda()

    # Initialize generator and discriminator
    generator = Generator(1)
    generator.apply(weights_init)
    discriminator = Discriminator(1)
    discriminator.apply(weights_init)

    if cuda:
        generator.cuda()
        discriminator.cuda()
        criterion.cuda()

    # Configure data loader
#DIST
    manager = DatasetManager(opt.model, opt.batch_size, opt.img_size, size-1, size, rank, opt.iid, 1)
    train_set, _ = manager.get_train_set(opt.magic_num)

    lbl_count = [0 for _ in range(NUM_CLASSES)]
    all_labels = []
    for i, (imgs, lbls) in enumerate(train_set):
        for lbl in lbls:
            if lbl.item() not in all_labels:
                all_labels.append(lbl.item())
            lbl_count[lbl.item()]+=1
    workers_classes = gather_lbl_count(lbl_count)
    num_per_class = [500 for _ in range(NUM_CLASSES)]
    all_samples = sum(num_per_class)
    rat_per_class = [float(n/all_samples) for n in num_per_class]
    #Calculating entropy at this worker
    ent = stats.entropy(np.array(lbl_count)/sum(lbl_count), rat_per_class)

    #Now, initializing all groups for the whole training process
    print("Rank {} Start init groups".format(rank))
    sys.stdout.flush()
    init_groups(size, workers_classes)
    print("Rank {} Done initializing {} groups".format(rank, len(all_groups)))
    #Calculating entropy of each worker (on the server side) based on these frequencies....
    if rank == 0 and opt.weight_avg:
        entropies = [stats.entropy(np.array(freq_l)/sum(freq_l), rat_per_class) * (sum(freq_l)/ all_samples) for freq_l in workers_classes]
#        print("Entropies are: ", entropies)

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
            test_imgs = t[0].cuda()
            test_labels = t[1]

    # ----------
    #  Training
    # ----------
    #DIST
    elapsed_time = time()
    weak_workers = []
    if weak_percent > 0.0:
        weak_workers = [i for i in range(1,size,round(1/weak_percent))]
    print("Number of simulated weak workers: ", len(weak_workers))
    num_batches=0		#This variable acts as a global state variable to sync. between workers and the server
    done_round = True
    group = None
    #The following hack (4 lines) is written to run actually the number of runs that the user is aiming for....because of the skewness of data, the actual number of epochs that would run could be less than that the user is estimating...These few lines solve this issue
    est_len = 1000000 // (size * opt.batch_size)		#Given a dataset of 50,000 imgaes, the estimated number of iterations to dataset is 50000/unm_workers
    act_len = len(train_set)
    if act_len < est_len:
        opt.n_epochs = int(opt.n_epochs * (est_len/act_len))
    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(train_set):
            #DIST
            if done_round:		#This means that a new round should start....done by sampling a few of workers and give them the latest version of the model(s)
                #First step: Choose a group of nodes to do computations in this round....
                fl_round+=1
                g = all_groups_np[fl_round%len(all_groups)]
                group = all_groups[fl_round%len(all_groups)]
                choose_r0 = False
                if rank == 0:
                    choose_r0 = choose_r[fl_round%len(all_groups)]
                if rank in g:
                    broadcast_model(generator, group)
                    broadcast_model(discriminator, group)
                    done_round = False
                else:		#This node is not chosen in the current group....no work for this node in this round....just continue and wait for a new announcement from the server
                    done_round = True
                    num_batches=num_batches+opt.local_steps	#Advance the pointer for workers that will not work this round
                    continue
            # Adversarial ground truths
            real_imgs = Variable(imgs.type(Tensor))
            valid = Variable(Tensor(real_imgs.size()[0], 1, 1, 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(real_imgs.size()[0], 1, 1, 1).fill_(0.0), requires_grad=False)
            num_batches+=1


            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = torch.randn(real_imgs.size()[0], opt.latent_dim, 1, 1)
            if cuda:
                z = z.cuda()
            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            d_gen = discriminator(gen_imgs)
            g_loss = criterion(d_gen, valid)
            g_loss.backward()

            #DIST
            #Averaging step.......added because of distributed setup now!
            local_steps = opt.local_steps
            if rank in weak_workers:
                local_steps = int(opt.local_steps/2)
            if num_batches%local_steps == 0 and num_batches > 0:
                if opt.weight_avg:
                    #This is a weighting scheme using the entropies based on the frequency of samples of each class at each worker
                    cur_gp = all_groups_np[fl_round%len(all_groups)]
                    if rank  == 0:
                        weights = [entropies[int(wrk)] for wrk in cur_gp]
                    else:	#dummy else
                        weights = [1.0/len(cur_gp) for _ in cur_gp]
                #This weighting is orthogonal to KL-weighting scheme
                average_models(generator, group, choose_r0,weights)
                done_round = True
            if rank == 0 and not choose_r0:
                g_p = generator.parameters()
                for param in generator.parameters():
                    param.grad.data = torch.zeros(param.size()).cuda() if cuda else torch.zeros(param.size())

            optimizer_G.step()
            if rank == 0 and not choose_r0:
                for o,n in zip(g_p, generator.parameters()):
                    if not torch.eq(o,n).all():
                        print("Generator updated while it should not have been!!!! error here.......")

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = criterion(discriminator(real_imgs), valid)
            fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)
            d_loss.backward()

            #DIST
            #Averaging step.......added because of distributed setup now!
            if num_batches%local_steps == 0 and num_batches > 0:
#In the new version, we apply weights anyway.....to account for weak workers not only KL-divergence
                average_models(discriminator, group, choose_r0, weights)
                done_round = True

            if rank == 0 and not choose_r0:
                for param in discriminator.parameters():
                    param.grad.data = torch.zeros(param.size()).cuda() if cuda else torch.zeros(param.size())
            optimizer_D.step()

            #Print stats and generate images only if this is the server
            batches_done = epoch * len(train_set) + i
            if rank == 0 and batches_done % opt.sample_interval == 0:
                print(
                    "Rank %d [Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] time %f"
                    % (rank, epoch, opt.n_epochs, i, len(train_set), d_loss.item(), g_loss.item(), time() - elapsed_time), 
                    end = ' ' if epoch != 0 else '\n'
                )

                # Evaluation setp => output images and calculate FID
                if batches_done % opt.sample_interval == 0 and batches_done != 0:
                    fid_z = torch.randn(64, opt.latent_dim, 1, 1)
                    if cuda:
                        fid_z = fid_z.cuda()
                    del gen_imgs
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
    fn(rank, size)

os.makedirs("images-dist", exist_ok=True)
    
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches (named B in FL notations)")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=1000, help="number of iterations to calculate the FID.")
#DIST
parser.add_argument("--model", type=str, default='celeba', help="model to train")
parser.add_argument("--local_steps", type=int, default=100, help="number of local steps to be executed in each worker before sending to the server (named E in FL notations).")
parser.add_argument("--frac_workers", type=float, default=0.1, help="fraction of workers that participate in each round computation (named C in FL notations).")
parser.add_argument("--fid_batch", type=int, default=4000, help="number of samples used to evaluate the progress of the GAN (using the FID score).")
parser.add_argument("--rank", type=int, default=-1, help="Rank of this node in the distributed setup.")
parser.add_argument("--size", type=int, default=-1, help="Total number of machines in this experiment.")
parser.add_argument("--iid", type=int, default=0, help="Determines whether data should be distributed in an iid fashion to all workers or not. Takes only 0 or 1 as a value.")
parser.add_argument("--weight_avg", type=int, default=0, help="If set, the new weighted averaging with entropies scheme takes place.")
parser.add_argument("--sample", type=int, default=0, help="If set, smart sampling takes place. Otherwise, random sampling is used.")
parser.add_argument("--port", type=str, default='29500', help="Port number of the master....required for connections from everybody.")
parser.add_argument("--master", type=str, default='igrida-abacus9', help="The master hostname...should be known by everybody.")
#parser.add_argument("--bench", type=int, default=1, help="If set, time taken by each step is printed.")
parser.add_argument("--weight_scheme", type=str, default='exp', help="Determines the weighting technique used. Currently existing schemes are dirac, linear, and exp.")
parser.add_argument("--magic_num", type=int, default=5000, help="Temporary value that determines the maximum number of samples should be with each class.")
parser.add_argument("--weak_percent", type=float, default=0.0, help="Determines the percentage of simulated weak workers in the deployment.")
opt = parser.parse_args()
opt.n_epochs *= int((1-opt.frac_workers)*200)		#This is to cope up with the workers that remain idle in fl rounds...to achieve fair comparison with the single-machine implementation
print(opt)
port = opt.port
master = opt.master
#DIST
size = opt.size
rank = opt.rank
model = opt.model
if model != 'mnist' and model != 'fashion-mnist':	#This is CIFAR10 then
    opt.channels = 3

assert opt.iid == 0 or opt.iid == 1
import socket
hostname = socket.gethostname()
if hostname == master:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' if rank==0 else str((rank%2) + 1)		#%1 should be replaced by %(num_gpus-1)...now we are testing with 2 GPUs per machine
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank%2) 			#Other machines can use both GPUs freely..only the master is allowed to take one GPU exclusively

cuda = True if torch.cuda.is_available() else False
print("Using Cuda?\n ", cuda, "Hostname: ", hostname)

weak_percent = opt.weak_percent
assert weak_percent >= 0.0 and weak_percent < 1.0
init_processes(rank,size, run)
