# CSC490 - KVASIR Kings
Exploring ML applications in colonoscopies through the HyperKvasir dataset. 

Pix2Pix Model:
We used the Pix2Pix model for our image-to-image translation (I2I) method, where we wanted to translate an image of a polyp segmentation mask to into an polyp edge map. We used the Google Colab code they provided for training, we first used a script they provided in order to create paired images of segmentation masks and corresponding edge maps, that were provided, to feed into the model as training input. Finally, we then changed a few lines of code to feed our specific images. Provided below are the Google Colab codes that were used.

## Members:
* Taha Kazi
* Thomas Mayer
* Venura Perera
* Jeffie Wong

## Dataset Information:
* [Download URL](https://datasets.simula.no/hyper-kvasir)
* [Paper Reference](https://www.nature.com/articles/s41597-020-00622-y)

## Pix2Pix Model Information:
* [Pix2Pix Reference](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
* [Google Colab Code 1](https://colab.research.google.com/drive/1oyL-fmNhVzOxzvKXhHM3B4wFDcGfIuao?usp=sharing) 
* [Google Colab Code 2](https://colab.research.google.com/drive/1vxoApD4cHN2dE7ygtc6gdMeHFSVshpZF?authuser=5#scrollTo=yFw1kDQBx3LN)
