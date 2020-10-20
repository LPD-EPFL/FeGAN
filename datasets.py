# coding: utf-8
###
 # @file   datasets.py
 # @author Arsany Guirguis  <arsany.guirguis@epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright (c) 2019 Arsany Guirguis.
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
 # Datasets management and partitioning.
###

#!/usr/bin/env python

import pathlib
import torch
import random
from random import Random
import numpy as np
from torchvision import datasets, transforms

datasets_list = ['mnist', 'fashion-mnist', 'cifar10', 'celeba', 'imagenet']
MNIST = datasets_list.index('mnist')
FASHION_MNIST = datasets_list.index('fashion-mnist')
CIFAR10 = datasets_list.index('cifar10')
CELEBA = datasets_list.index('celeba')
IMAGENET = datasets_list.index('imagenet')
class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
       """ Constructor of Partiotion Object
           Args
           data		dataset needs to be partitioned
           index	indices of datapoints that are returned
        """
       self.data = data
       self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        """ Fetching a datapoint given some index
	    Args
            index	index of the datapoint to be fetched
        """
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        """ Constructor of dataPartitioner object
	    Args
	    data	dataset to be partitioned
	    size	Array of fractions of each partition. Its contents should sum to 1
	    seed	seed of random generator for shuffling the data
	"""
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        """ Fetch some partition in the dataset
	    Args
	    partition	index of the partition to be fetched from the dataset
	"""
        return Partition(self.data, self.partitions[partition])

class DatasetManager(object):
    """ Manages training and test sets"""

    def __init__(self, dataset, minibatch, img_size, num_workers, size, rank, iid, num_servers):
        """ Constrctor of DatasetManager Object
	    Args
		dataset		dataset name to be used
		batch		minibatch size to be employed by each worker
		num_workers	number of works employed in the setup
		rank		rank of the current worker
		iid		data is distributed iid or not
	"""
        if dataset not in datasets_list:
            print("Existing datasets are: ", datasets_list)
            raise
        self.dataset = datasets_list.index(dataset)
        self.batch = minibatch * num_workers
        self.img_size = img_size
        self.num_workers = num_workers
        self.num_ps = size - num_workers
        self.rank = rank
        self.iid = iid
        self.num_servers = num_servers

    def fetch_dataset(self, dataset, train=True):
        """ Fetch train or test set of some dataset
		Args
		dataset		dataset index from the global "datasets" array
		train		boolean to determine whether to fetch train or test set
	"""
        homedir = str(pathlib.Path.home())
        if dataset == MNIST:
            return datasets.MNIST(
              homedir+'/FeGAN/data/mnist',
              train=train,
              download=train,
              transform=transforms.Compose([transforms.Resize(self.img_size),
                 transforms.ToTensor(),
                 transforms.Normalize([0.5], [0.5])
            ]))

        elif dataset == FASHION_MNIST:
            return datasets.FashionMNIST(
              homedir+'/FeGAN/data/fashion-mnist',
              train=train,
              download=train,
              transform=transforms.Compose([transforms.Resize(self.img_size),
                 transforms.ToTensor(),
                 transforms.Normalize([0.5], [0.5])
            ]))
        elif dataset == CELEBA:
            return datasets.ImageFolder(
              root=homedir+'/FeGAN/data/celeba',
              transform=transforms.Compose([
                  transforms.Resize(self.img_size),
                  transforms.CenterCrop(self.img_size),
                  transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                  ]))
        elif dataset == IMAGENET:
            return datasets.ImageFolder(
              root=homedir+'/FeGAN/data/tiny-imagenet-200/train',
              transform=transforms.Compose([
                  transforms.Resize(self.img_size),
                  transforms.CenterCrop(self.img_size),
                  transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                  ]))
        if dataset == CIFAR10:
            if train:
              transforms_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
              return datasets.CIFAR10(
               homedir+'/FeGAN/data/cifar10',
               train=True,
               download=True,
               transform=transforms_train)
            else:
              transforms_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
              return datasets.CIFAR10(
                homedir+'/FeGAN/data/cifar10',
                train=False,
                download=False,
                transform=transforms_test)

    def get_train_set(self, magic_num=5000):
        """ Fetch my partition of the train set"""
        train_set = self.fetch_dataset(self.dataset, train=True)
        size = self.num_workers
        bsz = int(self.batch / float(size))
        if self.iid:
            partition_sizes = [1.0 / size for _ in range(size)]
            partition = DataPartitioner(train_set, partition_sizes)
            partition = partition.use(self.rank - self.num_ps)
        else:
            #IMPORTANT::::::::The default experiment, num_cls magic number: 10.0 and cls_max: 3000
            #Tested the following magic numbers: [10.0, 15.0, 20.0, 25.0] and [3000, 4000, 5000]
            NUM_CLASSES = 200 if self.dataset == IMAGENET else 10
            partition = []
            """ This chunk of code will make the distribution of the dataset unbalanced between workers """
            num_cls = NUM_CLASSES if self.rank < self.num_servers else int(self.rank*200.0/self.num_workers)+1
            num_cls = NUM_CLASSES if num_cls > NUM_CLASSES else num_cls	#limit number of classes with each worker
            print("At worker {}, number of classes is {}".format(self.rank, num_cls))
            g = random.sample(range(0, NUM_CLASSES), num_cls)	#This variable determines which classes are they
            assert len(g) > 0, "revise the value given to num_cls variable"
            cls_count = [0 for _ in range(NUM_CLASSES)]		#This counts how many sample of each class has this client chosen so far
            print("At worker {}, number of classes is {}".format(self.rank, num_cls))
            #limiting number of samples per class gives weighting a better environment for beating the vanilla setup
            cls_max = [random.randint(1,magic_num if self.rank < self.num_servers else self.rank**2) for i in range(NUM_CLASSES)]	#Determines the maximum number of class samples for this worker
            #limiting number of samples per class.....otherwise, it is not truly an FL setup
            for i in range(NUM_CLASSES):
                cls_max[i] = (self.rank+1)*magic_num/(size+1) if cls_max[i] > magic_num else cls_max[i]
            assert len(g) != 0, "Please consider changing this magic number to resolve the assertion error"
            for i,t in enumerate(train_set):
                img, label = t
                if label in g and cls_count[label] < cls_max[label] and label <= NUM_CLASSES:
                    partition.append(t)
                    cls_count[label] += 1
                    if self.rank < self.num_servers and sum(cls_count) == 7500:	#A hack for fair comparison....let rank 0 has only 5000 samples
                        break
        print("Using batch size = ", bsz)
        train_set = torch.utils.data.DataLoader(
            partition, batch_size=bsz, shuffle=True, num_workers=2)
        return train_set, bsz

    def get_test_set(self):
        """ Fetch test set, which is global, i.e., same for all entities in the deployment"""
        test_set = self.fetch_dataset(self.dataset, train=False)
        test_set = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=True, num_workers=2)
        return test_set

