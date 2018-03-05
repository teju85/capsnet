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
$ ./dockerfiles/scripts/launch -runas user dsb:2018 /bin/bash
container$ cd /work/capsnet
container$ env LANG=C.UTF-8 python net.py
```
Use the '-h' option to net.py to know more about its customizations.

# Accuracy/Perf Numbers
## With uhat detach
Detaching uhat means to perform back-prop only for the last routing iteration.
```bash
Train epoch:49 time(s):40.344 loss=0.00019366 accuracy:0.9986
Test epoch:49 time(s):2.717 loss=0.00003051 accuracy:0.9938
```
Achieves 99.38% test accuracy and epoch time of 40s on a P100.

## Without uhat detach
```bash
Train epoch:49 time(s):47.462 loss=0.00017785 accuracy:0.9996
Test epoch:49 time(s):2.765 loss=0.00002714 accuracy:0.9934
```
Achieves % test accuracy and epoch time of s on a P100.

Since w/ and w/o detach really doesn't seem to cause huge differences in
accuracy, but w/ detach runs ~17% faster than w/o it, detach has been made the
default behavior in this repo.
