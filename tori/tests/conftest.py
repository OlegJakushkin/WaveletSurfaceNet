"""Make the ``pat`` package importable when tests run from anywhere, and seed torch."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.manual_seed(0)
