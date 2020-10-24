#!/usr/bin/env python3
import argparse
import os
import numpy as np
import math
import pathlib
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from fid_score import *
from inception import *
from time import time
os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=50, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=1000, help="calculate the FID every SAMPLE_INTERVAL iterations")
parser.add_argument("--fid_batch", type=int, default=150, help="number of samples used to evaluate the progress of the GAN (using the FID score).")
parser.add_argument("--model", type=str, default='fashion-mnist', help="Dataset to be used. Supported datasets now are fashion-mnist and mnist.")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False


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


# !!! Minimizes MSE instead of BCE
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
homedir = str(pathlib.Path.home())
os.makedirs(homedir+"/FeGAN/data/"+opt.model, exist_ok=True)
if opt.model == 'mnist':
    dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        homedir+"/FeGAN/data/mnist",
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
    dataloader = torch.utils.data.DataLoader(
         datasets.FashionMNIST(
              homedir+'/FeGAN/data/fashion-mnist',
              train=True,
              download=True,
              transform=transforms.Compose([transforms.Resize(opt.img_size),
                 transforms.ToTensor(),
                 transforms.Normalize([0.5], [0.5])
            ]),),
        batch_size=opt.batch_size,
        shuffle=True,
)
if opt.model == 'mnist':
    test_set = torch.utils.data.DataLoader(
        datasets.MNIST(
        homedir+"/FeGAN/data/mnist",
        train=False,
        download=False,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=10000,
    shuffle=False,
)
else:
    test_set = torch.utils.data.DataLoader(
         datasets.FashionMNIST(
              homedir+'/FeGAN/data/fashion-mnist',
              train=False,
              download=False,
              transform=transforms.Compose([transforms.Resize(opt.img_size),
                 transforms.ToTensor(),
                 transforms.Normalize([0.5], [0.5])
            ]),),
        batch_size=10000,
        shuffle=False,
)

fic_model = InceptionV3()
if cuda:
    fic_model = fic_model.cuda()
for i,t in enumerate(test_set):
    test_imgs = t[0].cuda() if cuda else t[0]

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------
num_batches=0
elapsed_time = time()
for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        num_batches+=1
        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] time %f"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), time() - elapsed_time)
            )
            fid_z = Variable(Tensor(np.random.normal(0, 1, (opt.fid_batch, opt.latent_dim))))
            gen_imgs = generator(fid_z)
            mu_gen, sigma_gen = calculate_activation_statistics(gen_imgs, fic_model, batch_size=opt.fid_batch)
            mu_test, sigma_test = calculate_activation_statistics(test_imgs[:opt.fid_batch], fic_model, batch_size=opt.fid_batch)
            fid = calculate_frechet_distance(mu_gen, sigma_gen, mu_test, sigma_test)
            print("FID Score: ", fid)
