# FeGAN

Authors: Arsany Guirguis

This is the system introduced in "FeGAN: Scaling Distributed GANs", co-authored with Erwan Le Merrer, Anne-Marie Kermarrec, and Rachid Guerraoui.
The FeGAN system enables training GANs in the Federated Learning setup.
FeGAN is implemented on PyTorch.

## Requirements
FeGAN was tested with the following versions
* torch (1.6) [tested with cuda version]
* torchvision (0.7.0)
* Python (3.6.10)
* Numpy (1.19.1)
* Scipy (1.5.2)

## Installation
The following steps should be applied for **ALL** machines that are going to contribute to the running of FeGAN.

1. Follow [PyTorch installation guide](https://pytorch.org/) (depending on your environment and preferences). Here are couple of examples:
 * If you want to use conda, download the installer from [here](https://www.anaconda.com/products/individual) and then run `sh Anaconda_$VERSION_NO.sh; conda install python=3.7`. Then, install Pytorch by running: `conda install pytorch torchvision cudatoolkit=10.2 -c pytorch` (assuming using the same tested environment). Note that, for this, you need to add conda path to `PATH` environment variable as follows `export PATH=$HOME/anaconda3/bin:$PATH` then run `source activate base`. **For the distributed setup, you will need to add this latter export line in `.bashrc` file in `$HOME` directory.**
 * If you want to use pip, run `pip install torch torchvision`.

2. Install the other required packages. For instance, if you are using pip, run `pip install numpy==1.19.1 scipy==1.5.2`
## Distributed deployment (FeGAN-based)
* `dist-lsgan.py` (or `dist-dcgan.py`) *To run the distributed LSGAN (or DCGAN) on multiple machines

```
usage: dist-lsgan.py [-h] [--n_epochs N_EPOCHS] [--batch_size BATCH_SIZE] [--lr LR] [--b1 B1] [--b2 B2] [--n_cpu N_CPU]
                     [--latent_dim LATENT_DIM] [--img_size IMG_SIZE] [--channels CHANNELS] [--sample_interval SAMPLE_INTERVAL]
                     [--model MODEL] [--local_steps LOCAL_STEPS] [--frac_workers FRAC_WORKERS] [--fid_batch FID_BATCH] [--rank RANK]
                     [--size SIZE] [--iid IID] [--weight_avg WEIGHT_AVG] [--sample SAMPLE] [--port PORT] [--master MASTER]
                     [--weight_scheme WEIGHT_SCHEME] [--magic_num MAGIC_NUM] [--timeout TIMEOUT] [--num_servers NUM_SERVERS]

optional arguments:
  -h, --help            show this help message and exit
  --n_epochs N_EPOCHS   number of epochs of training
  --batch_size BATCH_SIZE
                        size of the batches (named B in FL notations)
  --lr LR               adam: learning rate
  --b1 B1               adam: decay of first order momentum of gradient
  --b2 B2               adam: decay of first order momentum of gradient
  --n_cpu N_CPU         number of cpu threads to use during batch generation
  --latent_dim LATENT_DIM
                        dimensionality of the latent space
  --img_size IMG_SIZE   size of each image dimension
  --channels CHANNELS   number of image channels
  --sample_interval SAMPLE_INTERVAL
                        number of iterations to calculate the FID
  --model MODEL         dataset to be used. For LSGAN, we support mnist and fashion-mnist
  --local_steps LOCAL_STEPS
                        number of local steps to be executed in each worker before sending to the server (named E in FL notations)
  --frac_workers FRAC_WORKERS
                        fraction of workers that participate in each round computation (named C in FL notations)
  --fid_batch FID_BATCH
                        number of samples used to evaluate the progress of the GAN (using the FID score)
  --rank RANK           rank of this node in the distributed setup
  --size SIZE           total number of machines/devices in this experiment
  --iid IID             if set, data is distributed in an iid fashion on all devices; takes only 0 or 1 as a value
  --weight_avg WEIGHT_AVG
                        if set, KL-weighted averaging runs
  --sample SAMPLE       if set, balanced sampling is applied. Otherwise, random sampling is used
  --port PORT           port number of the master....required for connections from other devices
  --master MASTER       the master hostname...should be known by all devices
  --weight_scheme WEIGHT_SCHEME
                        determines the weighting technique used. Currently existing schemes are dirac, linear, and exp
  --magic_num MAGIC_NUM
                        determines the maximum number of samples per class on each device
  --timeout TIMEOUT     the maximum number of seconds waited by any node in a sub-group before firing a timeout exception
  --num_servers NUM_SERVERS
                        the number of servers deployed. Having multiple servers helps with crash fault tolerance
```
### Practical deployment of FeGAN
We provide bash scripts to deploy and run FeGAN on multiple machines. These scripts serve only as an example on how to use the provided software. The interested user can write whatever `run` script of choice.
1. Create a file "nodes" and fill with hostnames of nodes which should contribute to the experiment (each line should contain one hostname).

2. Run `./run_lsgan.sh size local_steps frac_workers batch_size sample weight_avg model magic_num nodes iid port`. Note that, we add the support for ``weak workers`` to `dist-dcgan.py` script; this simulates machines with low memory and power constraints. The last parameter given to `run_dcgan.sh` denotes the ratio of the number of these weak workers to the total number of workers, as given in the example below.

3. Run `./kill_lsgan.sh nodes` to cleanup.

* Examples: 

a) `./run_lsgan.sh 42 30 0.05 50 1 1 mnist 3000 nodes 0 1257`

b) `./run_dcgan.sh 56 30 0.04 50 1 1 imagenet 500 nodes 0 0.1`

* Notes: 

1. To experiment with Imagenet, run: `wget http://cs231n.stanford.edu/tiny-imagenet-200.zip; unzip tiny-imagenet-200.zip` inside `data` directory. 

2. To experiment with CelebA, check "http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html" and download the dataset inside `data` directory.

3. Currently, LSGAN supports `mnist` and `cifar10` datasets while DCGAN supports `imagenet`. 

4. `celebA` dataset is not compatible with FeGAN as the images in the dataset are not classified so, sampling and weighting techniques of FeGAN are not applicable.

## Single-machine/Centralized deployment

* `lsgan.py` (or `dcgan.py`) *To run the LSGAN (or DCGAN) on a single machine 
```
usage: lsgan.py [-h] [--n_epochs N_EPOCHS] [--batch_size BATCH_SIZE] [--lr LR]
                [--b1 B1] [--b2 B2] [--n_cpu N_CPU] [--latent_dim LATENT_DIM]
                [--img_size IMG_SIZE] [--channels CHANNELS]
                [--sample_interval SAMPLE_INTERVAL] [--fid_batch FID_BATCH]
                [--model MODEL]


optional arguments:
  -h, --help            show this help message and exit
  --n_epochs N_EPOCHS   number of epochs of training
  --batch_size BATCH_SIZE
                        size of the batches
  --lr LR               adam: learning rate
  --b1 B1               adam: decay of first order momentum of gradient
  --b2 B2               adam: decay of first order momentum of gradient
  --n_cpu N_CPU         number of cpu threads to use during batch generation
  --latent_dim LATENT_DIM
                        dimensionality of the latent space
  --img_size IMG_SIZE   size of each image dimension
  --channels CHANNELS   number of image channels
  --sample_interval SAMPLE_INTERVAL
                        number of image channels
  --fid_batch FID_BATCH
                        number of samples used to evaluate the progress of the
                        GAN (using the FID score)
  --model MODEL         dataset to be used. Supported datasets now are
                        fashion-mnist and mnist

```

* Note: DCGAN takes only `--model MODEL` as an argument. The rest are hard-coded.

* Examples: 

a) `python3 lsgan.py --n_epochs 200 --batch_size 50 --fid_batch 100 --model mnist`

b) `python3 dcgan.py --model imagenet`

## MD-GAN
* `md-gan.py` *To run MD-GAN on multiple machines

```
usage: md-gan.py [-h] [--n_epochs N_EPOCHS] [--batch_size BATCH_SIZE]
                 [--lr LR] [--b1 B1] [--b2 B2] [--n_cpu N_CPU]
                 [--latent_dim LATENT_DIM] [--img_size IMG_SIZE]
                 [--channels CHANNELS] [--sample_interval SAMPLE_INTERVAL]
                 [--model MODEL] [--fid_batch FID_BATCH]
                 [--rank RANK] [--size SIZE] [--iid IID]
                 [--port PORT] [--master MASTER] [--magic_num MAGIC_NUM]

optional arguments:
  -h, --help            show this help message and exit
  --n_epochs N_EPOCHS   number of epochs of training
  --batch_size BATCH_SIZE
                        size of the batches (named B in FL notations)
  --lr LR               adam: learning rate
  --b1 B1               adam: decay of first order momentum of gradient
  --b2 B2               adam: decay of first order momentum of gradient
  --n_cpu N_CPU         number of cpu threads to use during batch generation
  --latent_dim LATENT_DIM
                        dimensionality of the latent space
  --img_size IMG_SIZE   size of each image dimension
  --channels CHANNELS   number of image channels
  --sample_interval SAMPLE_INTERVAL
                        number of iterations to calculate the FID
  --model MODEL         dataset to be used (e.g., mnist, fashion-mnist, ..etc.)
  --fid_batch FID_BATCH
                        number of samples used to evaluate the progress of the
                        GAN (using the FID score)
  --rank RANK           rank of this node in the distributed setup
  --size SIZE           total number of machines in this experiment
  --iid IID             determines whether data should be distributed in an
                        iid fashion to all workers or not; takes only 0 or 1
                        as a value.
  --port PORT           the port number of the master....required for connections
                        from all machines.
  --master MASTER       The master hostname...should be known by all machines
  --magic_num MAGIC_NUM
                        determines the maximum number of
                        samples should be with each class
```

* To test with MD-GAN: `./run_md_gan.sh size model nodes` (after putting hostnames in `nodes` file).

## Notes
1. The repo should be cloned on all nodes contributing to the experiment. Also, all nodes should have the same `nodes` file (NFS would be a good choice for that purpose).

2. The bash scripts we provide require **password-less** ssh access among machines contributing to the distributed setup.

3. Experiments in the paper are all done on [Grid5000](https://www.grid5000.fr), on Lille site.

Corresponding author: Arsany Guirguis <arsany.guirguis@epfl.ch>
