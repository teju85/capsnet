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
container$ python net.py
```
Use the '-h' option to net.py to know more about its customizations.

# Benchmarking
```bash
$ ./dockerfiles/scripts/launch -runas user pytorch:latest-90_70 /bin/bash
container$ cd /work/capsnet
container$ ./benchmark.sh
```

# Accuracy/Perf Numbers
* uhat detach - detaching here means to perform backprop only for the last routing iteration
* training epochs of 25 is being used for comparisons below
* accuracy numbers are ratios between 0 and 1
* train and test timings are in seconds and are per epoch
* measurements are all taken using cuda 9.0 with cudnn 7.0

## P100
| Dataset  | detach? | Train acc | Train time | Test acc | Test time |
|----------|---------|-----------|------------|----------|-----------|
|    mnist |      no |    0.9953 |     41.269 |   0.9920 |     2.463 |
|    mnist |     yes |    0.9927 |     35.511 |   0.9909 |     2.580 |
|  cifar10 |      no |    0.8693 |     54.802 |   0.6422 |     4.250 |
|  cifar10 |     yes |    0.8281 |     45.726 |   0.6690 |     3.969 |

## V100
| Dataset  | detach? | Train acc | Train time | Test acc | Test time |
|----------|---------|-----------|------------|----------|-----------|
|    mnist |      no |    0.9953 |     24.670 |   0.9920 |     1.766 |
|    mnist |     yes |    0.9927 |     20.539 |   0.9909 |     1.744 |
|  cifar10 |      no |    0.8794 |     31.602 |   0.6572 |     2.917 |
|  cifar10 |     yes |    0.8290 |     26.718 |   0.6733 |     2.950 |

# Notes
## Regarding Cifar10
Main paper runs this dataset using an ensemble of 7 models to attain 10.6% test
error. In here, we only run one model and that too keeping most of the
hyper-params pretty much the same as those with MNIST.

## Runtime differences between Cifar10 and Mnist
The increase in runtime when compared to Mnist is totally attributable to the
difference in input image dimension between these 2 datasets.
