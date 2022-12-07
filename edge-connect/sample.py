import cv2
import hydra
import os
import random
import sys
#import torch
import numpy as np

#import torchvision.transforms.functional as F

from omegaconf import DictConfig, OmegaConf
#from PIL import Image
# from src.edge_connect import EdgeConnect
#from skimage.feature import canny


@hydra.main(version_base=None, config_path="../configs/sample.conf", config_name="config")
def main(cfg):
    if len(sys.argv) != 4 or len(sys.argv):
        print("wrong number of arguments. USAGE:\npython3 sample.py <healthy colon> <polyp image> <polyp mask> 0\npython3 sample.py <healthy colon> <polyp edge> <polyp mask> 1")
        return


    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in cfg.GPU)

    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0) 

    # build the model and initialize
    model = EdgeConnect(cfg)
    model.load()

    clean_img_string = argv[1]

    clean_img = cv2.imread(clean_img_string, cv2.IMREAD_COLOR)
    clean_img = cv2.resize(clean_img, (256, 256))
    clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
    clean_gray_img = cv2.cvtColor(clean_img, cv2.COLOR_RGB2GRAY)
    clean_edges = canny(clean_gray_img, sigma=2)

    if argv[4] == '0':
        polyp_img_string = argv[2]
        polyp_img = cv2.imread(polyp_img_string, cv2.IMREAD_GRAYSCALE)
        polyp_img = cv2.resize(polyp_img, (256, 256))
        polyp_edges = canny(polyp_img, sigma=2, mask=mask_img)
    else:
        edge_img_string = argv[2]
        edge_img = cv2.imread(edge_img_string, cv2.IMREAD_GRAYSCALE)
        polyp_edges = cv2.resize(edge_img, (256, 256))

    mask_img_string = argv[3]
    mask_img = cv2.imread(mask_img_string, cv2.IMREAD_GRAYSCALE)
    mask_img = cv2.resize(mask_img, (256, 256))

    cv2.imwrite("seg.jpg", mask_img)
    cv2.imwrite("clean.jpg", clean_img[:,:,::-1])

    clean_img = to_tensor(clean_img).to(model.device)
    clean_gray_img = to_tensor(clean_gray_img).to(model.device)
    clean_edges = to_tensor(clean_edges).to(model.device)
    mask_img = to_tensor(mask_img).to(model.device)
    polyp_edges = to_tensor(polyp_edges).to(model.device)

    merged_edges = (clean_edges * (1 - mask_img)) + polyp_edges
    
    model.inpaint_model.eval()
    
    outputs = model.inpaint_model(clean_img[None, :], merged_edges[None, :], mask_img[None, :])
    
    out_img = (outputs * 255).int().squeeze().permute(1,2,0).cpu()
    out = np.array(out_img)
   
    out_edges = np.array((merged_edges*255).int().permute(1,2,0).cpu())
    cv2.imwrite("test.jpg", out[:,:,::-1])
    cv2.imwrite("merged.jpg", out_edges)

def to_tensor(img):
    img = Image.fromarray(img)
    img_t = F.to_tensor(img).float()
    return img_t
    
if __name__ == "__main__":
    main()
