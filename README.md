# PyTorch-ParameterServer
An implementation of parameter server (PS) framework [1] based on Remote Procedure Call (RPC) in PyTorch [2].

## Table of Contents

- [PS-based Architecture](#PS-based-architecture)
- [Implementation](#implementation)
- [Environments](#environments)
- [Quick Start](#quick-start)
	- [Download the code](#download-the-code)
	- [Install dependencies](#install-dependencies)
	- [Prepare datasets](#prepare-datasets)
	- [Train](#train)
	- [Performance](#performance)
- [Usage](#Usage)
- [References](#References)

## PS-based Architecture
<div align=center><img width="80%" src="./architecture.jpg"/></div>
The figure [3] below shows the PS-based architecture. The architecture consists of two logical entities: one (or multiple) PS(s) and multiple workers. The whole dataset is partitioned among workers and the PS maintains model parameters. During training, each worker pulls model parameters from the PS, computes gradients on a mini-batch from its data partition, and pushes the gradients to the PS. The PS updates model parameters with gradients from the workers according to a synchronization strategy and sends the updated parameters back to the workers. The pseudocode [1] of this architecture is shown as follows.
<div align=center><img width="50%" src="./ps-algo.png"/></div>

## Implementation
This code is based on torch.distributed.rpc [4]. It is used to train ResNet50 [5] on Imagenette dataset [6] - a subset of ImageNet [7] with one PS (rank=0) and 4 workers (rank=1,2,3,4). 
## Environments
The code is developed under the following configurations.  
Server: a g3.16xlarge instance with 4 NVIDIA Tesla M60 GPUs on AWS EC2  
System: Ubuntu 18.04  
Software: python==3.6.9, torch==1.9.0, torchvision==0.10.0  
## Quick Start
### Download the code
```bash
git clone https://github.com/xbfu/PyTorch-ParameterServer.git
```
### Install dependencies
```bash
cd PyTorch-ParameterServer
sudo sh install-dependencies.sh
```
### Prepare datasets
```bash
cd PyTorch-ParameterServer
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
tar -zxf imagenette2.tgz
```
### Train
For PS
```python
python public-asgd.py --rank=0
```
For workers
```python
python public-asgd.py --rank=r
```
`r=1,2,3,4` is the rank of each worker.

### Performance
Sync Mode | Training Time (seconds)
:-: | :-:
Single | 858
Syn | 533
Asyn | 268
## Usage
On one machine with multiple GPUs
For PS
```python
python public-asgd.py --rank=0
```
For workers
```python
python public-asgd.py --rank=r
```
`r=1,2,3,4` is the rank of each worker.

On multiple machines
For PS
```python
python public-asgd.py --rank=0 --master_addr=12.34.56.78
```
For workers
```python
python public-asgd.py --rank=r --master_addr=12.34.56.78
```
`r=1,2,3,4` is the rank of each worker. `12.34.56.78` is the IP address of the PS.

## References
[1]. Li M, Andersen D G, Park J W, et al. [Scaling distributed machine learning with the parameter server](https://www.usenix.org/system/files/conference/osdi14/osdi14-paper-li_mu.pdf  )//11th {USENIX} Symposium on Operating Systems Design and Implementation ({OSDI} 14). 2014: 583-598.  
[2]. Pytorch. https://pytorch.org/.  
[3]. Sergeev A, Del Balso M. [Horovod: fast and easy distributed deep learning in TensorFlow](https://arxiv.org/abs/1802.05799). arXiv preprint arXiv:1802.05799, 2018.  
[4]. Distributed RPC Framework. https://pytorch.org/docs/1.9.0/rpc.html.  
[5]. He, K., Zhang, X., Ren, S., & Sun, J. (2016). [Deep residual learning for image recognition](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf). In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).  
[6]. Imagenette. https://github.com/fastai/imagenette.  
[7]. Imagenet. https://image-net.org/.  
