# Introduction
This tries to reproduce [Capsule Network](https://arxiv.org/pdf/1710.09829.pdf)
results in pytorch.

# Installation
* Pre-reqs are found [here](https://github.com/teju85/dockerfiles#pre-reqs)
* git clone https://github.com/teju85/dockerfiles
* cd dockerfiles/ubuntu1604
* make pytorch90_70
* cd ../..
* git clone https://github.com/teju85/capsnet

# USAGE
```bash
$ ./dockerfiles/scripts/launch -runas user pytorch:latest-90_70 /bin/bash
container$ cd /work/capsnet
container$ env LANG=C.UTF-8 python net.py
```
Use the '-h' option to net.py to know more about its customizations.

# Accuracy/Perf Numbers
* uhat detach - detaching means to perform backprop only for the last routing iteration
* training epochs of 50 is being used for comparisons below
* accuracy numbers are ratios between 0 and 1
* train and test timings are in seconds
* recon - whether or not the reconstruction loss was enabled
| Dataset | recon? | uhat?      | Train accuracy | Train time | Test accuracy | Test time |
|---------|--------|------------|----------------|------------|---------------|-----------|
| mnist   | no     | detach     | 0.9986         | 40.344     | 0.9938        | 2.717     |
| mnist   | no     | not detach | 0.9996         | 47.462     | 0.9934        | 2.765     |
| cifar   | no     | detach     | 0.9832         | 53.560     | 0.6218        | 5.017     |

## Note
Since w/ and w/o detach really doesn't seem to cause huge differences in
accuracy, but w/ detach runs ~17% faster than w/o it, detach has been made the
default behavior in this repo.

## Regarding Cifar10
Main paper runs this dataset using an ensemble of 7 models to attain 10.6% test
error. In here, we only run one model and that too keeping most of the
hyper-params pretty much the same as those with MNIST.

## Runtime differences between Cifar10 and Mnist
~30% increase in runtime when compared to Mnist is totally attributable to the
difference in input image dimension between these 2 datasets.
