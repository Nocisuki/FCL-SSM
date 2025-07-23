# Code for FCL-SSM
> FCL-SSM: Selective Synthetic Memory Replay for Class-Overlapping Federated Continual Learning

## Requairments
The needed libraries are in requirements.txt.

## Experiments
To run on MNIST, excute:

    python main.py --dataset=mnist --method=ours --task=3 --beta=0 --g_rounds=1000 --kd=5

To run on CIFAR-10, excute:

    python main.py --dataset=cifar10 --method=ours --task=5 --beta=0 --g_rounds=2000 --kd=5

To run on CIFAR-100, excute:

    python main.py --dataset=cifar100 --method=ours --task=5 --beta=0 --g_rounds=2200 --kd=10

