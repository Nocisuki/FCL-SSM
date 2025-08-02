# Code for FCL-SSM
> FCL-SSM: Selective Synthetic Memory Replay for Class-Overlapping Federated Continual Learning

## Requairments
The needed libraries are in requirements.txt.

## Datasets
- **MNIST**  
  - **Description**: Handwritten digit recognition dataset.  
  - **Version**: Original (70,000 images, 60,000 training + 10,000 testing).  
  - **Download**: [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) 

- **CIFAR-10**  
  - **Description**: 60,000 32x32 color images in 10 classes.  
  - **Version**: v1.0 (50,000 training + 10,000 testing).  
  - **Download**: [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) 

- **CIFAR-100**  
  - **Description**: 60,000 32x32 color images in 100 classes.  
  - **Version**: v1.0 (50,000 training + 10,000 testing).  
  - **Download**: [CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)  

## Experiments
To run on MNIST, excuate:

    python main.py --dataset=mnist --method=ours --task=3 --beta=0 --g_rounds=1000 --kd=5 --overlap_rate=0.25

To run on CIFAR-10, excuate:

    python main.py --dataset=cifar10 --method=ours --task=3 --beta=0 --g_rounds=2000 --kd=5 --overlap_rate=0.25

To run on CIFAR-100, excuate:

    python main.py --dataset=cifar100 --method=ours --task=5 --beta=0 --g_rounds=2200 --kd=10 --overlap_rate=0.4

