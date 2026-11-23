"""Explicit user-invoked dataset download; never run during tests."""
import argparse
from torchvision.datasets import CIFAR100
p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);a=p.parse_args()
for train in [True,False]:CIFAR100(a.data_root,train=train,download=True)
