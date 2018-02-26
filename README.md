# Introduction
This tries to reproduce [Capsule Network](https://arxiv.org/pdf/1710.09829.pdf)
results in pytorch.

# Installation
* Pre-reqs are found [here](https://github.com/teju85/dockerfiles#pre-reqs)
* git clone https://github.com/teju85/dockerfiles
* cd dockerfiles/ubuntu1604
* make dsb-2018
* cd ../..
* git clone https://github.com/teju85/capsnet

# USAGE
```bash
$ ./dockerfiles/scripts/launch -runas user dsb:2018 /bin/bash
container$ cd /work/capsnet
container$ env LANG=C.UTF-8 python net.py
```
Use the '-h' option to net.py to know more about its customizations.
