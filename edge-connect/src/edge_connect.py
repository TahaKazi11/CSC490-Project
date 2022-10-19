import os
import torch
import numpy as np

from torch.utils.data import DataLoader
from data.dataset import Dataset
from evaluators.metrics import PSNR, EdgeAccuracy
from models.models import EdgeModel, InpaintingModel

class EdgeConnect():
    def __init__(self, cfg):
        self.cfg = cfg

    if config.MODEL == 1:
        model_name = 'edge'
    elif config.MODEL == 2:
        model_name = 'inpaint'
    elif config.MODEL == 3:
        model_name = 'edge_inpaint'
    elif config.MODEL == 4:
        model_name = 'joint'