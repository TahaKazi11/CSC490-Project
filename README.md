# CSC490 - KVASIR Kings
# Improving PolypConnect - KVASIR Kings
The lack of curated colon and polyp datasets limits the robustness of computer assisted Diagnosis systems. Therefore, we propose a GAN-based machine learning pipeline to generate large quantities of novel polyp images with corresponding ground-truth mask data. This is completed by training two separate GAN models, those being a pix2pix model for polyp edge image generation and an inpainting model based on polypconnect for the actual polyp image generation.

## Dataset Information:
Two datasets are used for training each of the models. Furthermore, we also utilize a dataset of generated plausible polyp segmentation masks as input into the pix2pix model post-training and as part of polypconnect's pretaining data. The polypconnect model is trained using the HyperKvasir dataset. More specifically, the unlabelled images dataset is used for pretraining, while the segemented images dataset is used for finetuning. The pix2pix model is trained using a augmentation of the HyperKvasir segmented images dataset which consists of realy polyp segmentation masks (from HyperKavsir) and assoicated edge maps generated using the canny edge detection algorithm on the masked real polyp images.
#### HyperKvasir
* [Download URL](https://datasets.simula.no/hyper-kvasir)
* [Paper Reference](https://www.nature.com/articles/s41597-020-00622-y)

Unlabeled Images: 99417 GI-tract images of varying resolution

Segmented Images: 1000 polyp images and respective segmentation mask at varying resolutions

NOTE: the unlabeled images are not provided in the repo as there are too many images to effectively store on git.

#### Synth. Segmentation Masks
* [Auxilliary Seg Mask Dataset](https://zenodo.org/record/5537151#.Y1b3SEzMKUk)
* [Paper Reference](https://arxiv.org/abs/2106.04463)

10000 synthesized poylp segmentation masks at varying resolution

For examples of the datasets, please refer to the datasets directory.

## PolypConnect (THOMAS):
[Paper Reference](https://arxiv.org/abs/2205.15413).
[Code Reference](https://github.com/andrefagereng/polyp-gan)
### Pipeline:
The polypconnect model for inpainting polyps onto colon images is a 4-step pipeline:
<ol>
  <li>The model is pretrained to learn to inpaint general colon images. This is done by giving the model random colon images from
the unlabeled dataset paired with random segmentation masks from the synthetic segmentation mask dataset.</li>
  <li>The model is finetuned with actual polyp images from the segmented images dataset. This step helps the model to learn the 
features specific to polyps in the colon. Note, we apply augmentations to the 800 image training set to improve the robustness 
of the model. For more information, please see the section on data augmentation. </li>
  <li>The input data for novel image generation is prepared. In the original approach, this is done by taking a real polyp image from the segmented images dataset and generating the edges of the polyp by masking the rest of the image out with the ground truth segmentation mask. Once the edges are generated for the given segmentation mask, the pair is combined with a random colon image. We hope to improve the robustness of the overall pipeline by using pix2pix to generate edges for any segmentation mask. This will allow for more variation in the images generated by the final step. </li>
  <li>Finally, the input data created in step 3 is fed into the finetuned inpainting model. The output consists of the novel polyp image
which can be paired with the segmentation mask utilized to serve as ground truth data.</li>
</ol> 

### Prerequisites:
 * *nix 
 * GPU(s)

### Dependencies:
Install the following libraries using your choice of env/pkg manager (i.e. pip or anaconda). Note the installation for torch, torchvision, and CUDA/ROCM support vary widely release to release. For more information please refer to the [PyTorch documentation](https://pytorch.org)
* torch 
* torchvision 
* hydra-core  
* numpy 
* omegaconf 
* pillow 
* scikit-image 
* opencv-python

### Training:
Prior to training, it is a good idea to create a unique configuration file. The simplest way to do this is to create a new folder under the configs directory with a ".conf" extension (used by hydra). Then, copy the config.yaml in in `configs/base.conf/config.yaml` to your newly created directory. You can then edit this file as you see fit for the specifc training instance you are working on. For mor information on the configuration options, please see the comments in the aformentioned config.yaml found at `configs/base.conf/config.yaml`.

Once configured, training can take place. This can be accomplished by running the following command:

`python3 ./edge-connect/main.py --config-path ../configs/<config dir you created>.conf`

NOTE: the --config-path argument is relative to where main.py is which is why the double dot realtive path is used.

Checkpoints are logged to the directory specified by the PATH option in the config.yaml files.


### Testing:
To test the model, some modification to the config.yaml file is required. This largely consits of changing the MODE parameter to 2. Again, command to start testing is:

`python3 ./edge-connect/main.py --config-path ../configs/<config dir you created>.conf`

### Sampling:
To see the model in action, a python script called "sample.py" is provided in edge-connect. This script it run in one of two ways:

`python3 ./edge-connect/sample.py <base colon image path> <polyp image path> <mask image path> 0`
`python3 ./edge-connect/sample.py <base colon image path> <polyp edge path> <mask image path> 1`

These will take in the inputs, concatenate them, and feed them to the polyp connect model which will then output 4 images:
* test.jpg - final output
* merged.jpg - merged final edges
* clean.jpg - resized base colon image
* seg.jpg - resized mask image

We support both polyp image paths and polyp edge paths to allow for sample with the origional step 3 or our proposed step 3.

## Pix2Pix Model:
We used the Pix2Pix model for our image-to-image translation (I2I) method, where we wanted to translate an image of a polyp segmentation mask to into an polyp edge map. We used the Google Colab code they provided for training, we first used a script they provided in order to create paired images of segmentation masks and corresponding edge maps, that were provided, to feed into the model as training input. Finally, we then changed a few lines of code to feed our specific images. Provided below are the Google Colab codes that were used. We planned to use I2I, such that the current pipeline does not rely on real polyp images - unlike in Polypconnect, where Polypconnect needs real image data. After this, the edge map can be used to inpaint a polyp in the healthy colon.

## Pix2Pix Model Information/Code:
* [Pix2Pix Reference](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
* [Google Colab Code 1](https://colab.research.google.com/drive/1oyL-fmNhVzOxzvKXhHM3B4wFDcGfIuao?usp=sharing) 
* [Google Colab Code 2](https://colab.research.google.com/drive/1vxoApD4cHN2dE7ygtc6gdMeHFSVshpZF?authuser=5#scrollTo=yFw1kDQBx3LN)

## Data Augmentation:
Since we only have 1000 images in our dataset, our model is prone to overfit during training. We decided to use data augmentation to solve this issue. Data augmentation is a technique where transformations are applied to our original dataset, thus creating a larger variation of polyp images for training and fine-tuning of our model. These transforamtions include rotation, flipping vertically, flipping horizontally, random distortion, brightness, skew corner, skew tilt, shear and zoom, and they are applied in a chain with a probability of each transformation occurring and a magnitude specifying how severe these augmentations are. 

The library that we used is Augmentor, it provides tools to aid the augmentation and artificial generation of image data for machine learning tasks. Running the script data_augmentation.py creates 7500 augmented images and their respective masks in each of the following directories: p2m6, p2m7, p2m8, p3m6, p3m7, p3m8, p4m6, p4m7 and p4m8, where p2 represents a probability of 0.2 and m8 represents a magnitude of 8/10. In directories rotation_augment, flip_random_augment, distortion_augment, brightness_augment, skew_augment, shear_augment and zoom_augment, an image and its mask is provided to demonstrate what each transformation looks like, with the original version being 0004a718-546c-41c2-9c69-c4685093a039.jpg in the directory original.

## Members:
* Taha Kazi
* Thomas Mayer
* Venura Perera
* Jeffie Wong
