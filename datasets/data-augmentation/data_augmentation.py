import Augmentor
import random
import shutil
import os
from sklearn.pipeline import Pipeline

from sqlalchemy import false

# Directory paths
curr_dir = os.path.dirname(__file__)
src_images_dir = curr_dir + '/../segmented-images/images'
src_masks_dir = curr_dir + '/../segmented-images/masks'
train_images_dir = curr_dir + '/../train/images'
train_masks_dir = curr_dir + '/../train/masks'
test_images_dir = curr_dir + '/../test/images'
test_masks_dir = curr_dir + '/../test/masks'

# Source images and masks
src_images_list = sorted(os.listdir(src_images_dir))
src_masks_list = sorted(os.listdir(src_masks_dir))

# Directory paths related to augmentations
original_images_dir = curr_dir + '/original/images'
original_masks_dir = curr_dir + '/original/masks'
rotation_images_dir = curr_dir + '/rotation_augment/images'
rotation_masks_dir = curr_dir + '/rotation_augment/masks'
distortion_images_dir = curr_dir + '/distortion_augment/images'
distortion_masks_dir = curr_dir + '/distortion_augment/masks'
crop_images_dir = curr_dir + '/crop_augment/images'
crop_masks_dir = curr_dir + '/crop_augment/masks'
flip_random_images_dir = curr_dir + '/flip_random_augment/images'
flip_random_masks_dir = curr_dir + '/flip_random_augment/masks'
skew_images_dir = curr_dir + '/skew_augment/images'
skew_masks_dir = curr_dir + '/skew_augment/masks'
shear_images_dir = curr_dir + '/shear_augment/images'
shear_masks_dir = curr_dir + '/shear_augment/masks'
brightness_images_dir = curr_dir + '/brightness_augment/images'
brightness_masks_dir = curr_dir + '/brightness_augment/masks'
random_aug_images_dir = curr_dir + '/random_augments/images'
random_aug_masks_dir = curr_dir + '/random_augments/masks'


# Augmentation related parameters
aug_img_num = 7500
default_magnitude = 0.7


# Copy 5 images to the original/images folder to test augmentations
def copy_to_original_img(num):
    for i in range(num):
        img_path = src_images_dir + '/' + src_images_list[i]
        shutil.copy(img_path, original_images_dir)
        # print(img_path)

# Copy 5 masks to the original/masks folder to test augmentations
def copy_to_original_masks(num):
    for i in range(num):
        img_path = src_masks_dir + '/' + src_masks_list[i]
        shutil.copy(img_path, original_masks_dir)

# Remove all images in a directory
def clear_image_directory(image_dir):
    [os.remove(image_dir + '/' + x) for x in os.listdir(image_dir)]
    print("Cleared images in " + image_dir)


def load_original_images(n=1000):
    clear_image_directory(original_images_dir)
    clear_image_directory(original_masks_dir)
    copy_to_original_img(n)
    copy_to_original_masks(n)

# Apply rotattion {90, 180, 270} degress randomly once on each image
def apply_rotation_augment():
    clear_image_directory(rotation_images_dir)
    rotation_pipeline = Augmentor.Pipeline(original_images_dir, rotation_images_dir)
    # rotation_pipeline.ground_truth(original_masks_dir)
    # rotation_pipeline.rotate_random_90(1)
    rotation_pipeline.rotate90(1)
    rotation_pipeline.process()

# Generate 1000 images with random rotation applied  
# clear_image_directory(rotation_images_dir)
# rotation_pipeline.rotate_random_90(0.75)
# rotation_pipeline.sample(5)

# Apply random elastic distortions to randomly selected images and generate 5 of such images
def apply_distortion_augment():
    clear_image_directory(distortion_images_dir)
    distortion_pipeline = Augmentor.Pipeline(original_images_dir, distortion_images_dir)
    distortion_pipeline.ground_truth(original_masks_dir)
    distortion_pipeline.random_distortion(1, 5, 5, magnitude=default_magnitude*10)
    distortion_pipeline.process()
# distortion_pipeline.sample(5)

# crop images randomly by 80%
def apply_crop_augment():
    clear_image_directory(crop_images_dir)
    crop_pipeline = Augmentor.Pipeline(original_images_dir, crop_images_dir)
    crop_pipeline.ground_truth(original_masks_dir)
    # crop_pipeline.crop_random(1, default_magnitude, False)
    crop_pipeline.zoom(1, 1.3, 1.3)
    crop_pipeline.process()
# testing: crop_pipeline.resize(1, 256, 256, )

# flip images horizontally and vertically randomly 
def apply_flip_augment():
    clear_image_directory(flip_random_images_dir)
    flip_random_pipeline = Augmentor.Pipeline(original_images_dir, flip_random_images_dir)
    # flip_random_pipeline.ground_truth(original_masks_dir)
    # flip_random_pipeline.flip_random(1)
    flip_random_pipeline.flip_left_right(1)
    flip_random_pipeline.process()

# Skew images by a random corner
def apply_skew_augment():
    clear_image_directory(skew_images_dir)
    skew_pipeline = Augmentor.Pipeline(original_images_dir, skew_images_dir)
    skew_pipeline.ground_truth(original_masks_dir)
    skew_pipeline.skew_corner(1, default_magnitude)
    skew_pipeline.process()

# Shearing tilts an image along one of its sides. The can be in the x-axis or y-axis direction
def apply_shear_augment():
    clear_image_directory(shear_images_dir)
    shear_pipeline = Augmentor.Pipeline(original_images_dir, shear_images_dir)
    shear_pipeline.ground_truth(original_masks_dir)
    shear_pipeline.shear(1, 25 * default_magnitude, 25 * default_magnitude)
    shear_pipeline.process()

# Apply random brightness augment
def apply_brightness_augment():
    clear_image_directory(brightness_images_dir)
    brightness_pipeline = Augmentor.Pipeline(original_images_dir, brightness_images_dir)
    brightness_pipeline.ground_truth(original_masks_dir)
    brightness_pipeline.random_brightness(1, 0.5, 1.5)
    brightness_pipeline.process()


# Apply random augmentations to images in a directory with certain probabilities and magnitudes
# p: probability (0 - 1)
# m: magnitude (0 - 1)
# dir_name: name of directory containing images and masks directories
def apply_random_augments(p, m, dir_name):
    image_dir_path = curr_dir + '/' + dir_name + '/images'
    masks_dir_path = curr_dir + '/' + dir_name + '/masks'

    clear_image_directory(image_dir_path)
    clear_image_directory(masks_dir_path)

    random_aug_pipeline = Augmentor.Pipeline(original_images_dir, image_dir_path)
    random_aug_pipeline.ground_truth(original_masks_dir)
    
    random_aug_pipeline.rotate_random_90(p)
    random_aug_pipeline.flip_left_right(p)
    random_aug_pipeline.flip_top_bottom(p)
    random_aug_pipeline.random_distortion(p, 5, 5, m * 10)  # Grid width and height need to be between 2 to 10, magnitude is between 1 to 10
    random_aug_pipeline.random_brightness(p, 1 - m, 1 + m)  # Need to specify min and max brightness where 1 is the original
    random_aug_pipeline.skew_corner(p, m)
    random_aug_pipeline.skew_tilt(p, m)
    random_aug_pipeline.shear(p, 25 * m, 25 * m)            # Angles more than 25 will cause unpredictable behaviour
    # random_aug_pipeline.crop_random(p, m)                   # Crop percentages
    random_aug_pipeline.zoom(p, 1 + 0.5 * m, 1 + 0.5 * m)
    random_aug_pipeline.resize(1, 256, 256)
    # random_aug_pipeline.process()
    random_aug_pipeline.sample(aug_img_num)
    separate_images_and_masks(dir_name)
    # print("selected augments: " + str(selected_augments))
    
def separate_images_and_masks(dirname):
    images_dir = curr_dir + "/" + dirname + "/images"
    masks_dir = curr_dir + "/" + dirname + "/masks"

    image_list = sorted(os.listdir(images_dir))
    # print()
    for i in image_list:
        if (i.startswith("_groundtruth_(1)_images_")):
            new_name = i.split("_groundtruth_(1)_images_")[1]
            os.rename(images_dir + "/" + i, masks_dir + "/" + new_name)
        elif (i.startswith("images_original_")):
            new_name = i.split("images_original_")[1]
            os.rename(images_dir + '/' + i, images_dir + '/' + new_name)


load_original_images(1000)
# apply_random_augments(1, 0.8, "random_augments")
apply_random_augments(0.2, 0.6, "p2m6")
apply_random_augments(0.2, 0.7, "p2m7")
apply_random_augments(0.2, 0.8, "p2m8")

apply_random_augments(0.3, 0.6, "p3m6")
apply_random_augments(0.3, 0.7, "p3m7")
apply_random_augments(0.3, 0.8, "p3m8")

apply_random_augments(0.4, 0.6, "p4m6")
apply_random_augments(0.4, 0.7, "p4m7")
apply_random_augments(0.4, 0.8, "p4m8")

# apply_random_augments(0.7, 0.8, "random_augments")
# apply_shear_augment()
# apply_brightness_augment()
# apply_flip_augment()
# apply_distortion_augment()
# apply_shear_augment()
# apply_crop_augment()
# apply_skew_augment()
# apply_random_augments(1, 0.8, "random_augments")
# clear_image_directory(original_images_dir)
# clear_image_directory(original_masks_dir)
# clear_image_directory(curr_dir + "/p2m6/images")
# clear_image_directory(curr_dir + "/p2m6/masks")