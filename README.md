# Robust Estimation for Center Parameter.
This repository provides a PyTorch implementation of **[Chao Gao, Jiyi Liu, Yuan Yao and Weizhi Zhu, "
Robust Estimation and Generative Adversarial Nets"](https://arxiv.org/abs/1810.02030).**

## Environment
* Python 3.6
* [PyTorch 0.4.1](http://pytorch.org/)

## Files
This repository includes the following files,
* `fgan.py`: The main class for data generating process of Huber's Contamination Model and training process via f-GANs.
* `network.py`: Network structure of Generator, GenenatorXi and Discriminator, where GeneratorXi is used for the family of elliptical distribution.
* `data_loader.py`: Dataset preparation.
* `Demo.ipynb`: A notebook demo illustrating the usage of the code. It includes two core examples that **known** Gaussian type distribution and **unknown** elliptical type distribution (Cauchy in this demo). 