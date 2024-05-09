# AdaGossip

Official PyTorch implementation for  "**AdaGossip: Adaptive Consensus Step-size for Decentralized Deep Learning with Communication Compression**" [Paper](https://arxiv.org/abs/2404.05919)

## Abstract 
Decentralized learning is crucial in supporting on-device learning over large distributed datasets, eliminating the need for a central server. However, the communication overhead remains a major bottleneck for the practical realization of such decentralized setups. To tackle this issue, several algorithms for decentralized training with compressed communication have been proposed in the literature. Most of these algorithms introduce an additional hyper-parameter referred to as consensus step-size which is tuned based on the compression ratio at the beginning of the training. In this work, we propose AdaGossip, a novel technique that adaptively adjusts the consensus step-size based on the compressed model differences between neighboring agents. We demonstrate the effectiveness of the proposed method through an exhaustive set of experiments on various Computer Vision datasets (CIFAR-10, CIFAR-100, Fashion MNIST, Imagenette, and ImageNet), model architectures, and network topologies. Our experiments show that the proposed method achieves superior performance (0−2% improvement in test accuracy) compared to the current state-of-the-art method for decentralized learning with communication compression.

### Datasets
* Fashion-MNIST
* CIFAR-10
* CIFAR-100
* Imagenette
* ImageNet

### Models
* LeNet-5
* ResNet
* VGG
* MobileNet

# Requirements
* found in env.yml file

# Hyper-parameters
* --world_size   = total number of agents
* --graph        = graph topology (default ring)
* --neighbors    = number of neighbors per agent (default 2)
* --optimizer    = global optimizer i.e., [d-psgd, ngc, cga, compngc, compcga]
* --arch         = model to train
* --normtype     = type of normalization layer
* --dataset      = dataset to train
* --batch_size   = batch size for training
* --epochs       = total number of training epochs
* --lr           = learning rate
* --momentum     = momentum coefficient
* --nesterov     = activates nesterov momentum
* --weight_decay = weight decay
* --gamma        = averaging rate for gossip 
* --ada_gossip   = activates the adaptive gossip
* --compressor   = type of compressor function i.e., quantize or sparsify (CHOCO-SGD implementation)
* --k            = compression ratio for sparsification
* --level        = quantization level 1 to 32
* --biased       = activates biaased compression for quantization
* --skew         = amount of skew in the data distribution; 10 = completely iid and 0 = completely non-iid

### Citation
```
@article{aketi2024adagossip,
  title={AdaGossip: Adaptive Consensus Step-size for Decentralized Deep Learning with Communication Compression},
  author={Aketi, Sai Aparna and Hashemi, Abolfazl and Roy, Kaushik},
  journal={arXiv preprint arXiv:2404.05919},
  year={2024}
}
```
