import cv2
import hydra
import os
import random
import torch
import numpy as np

from omegaconf import DictConfig, OmegaConf
from shutil import copyfile
from src.edge_connect import EdgeConnect

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in cfg.GPU)

    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)

    # initialize random seed
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)

    # build the model and initialize
    model = EdgeConnect(cfg)
    model.load()

    # model training
    if cfg.MODE == 1:
        print('\nstart training...\n')
        model.train()

    # model test
    elif cfg.MODE == 2:
        print('\nstart testing...\n')
        model.test()

    # eval mode
    else:
        print('\nstart eval...\n')
        model.eval()


if __name__ == "__main__":
    main()