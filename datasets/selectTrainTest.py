import random
import shutil
import os

# Directory paths
project_dir = os.path.dirname(__file__)
src_images_dir = project_dir + '/datasets/segmented-images/images'
src_masks_dir = project_dir + '/datasets/segmented-images/masks'
train_images_dir = project_dir + '/datasets/train/images'
train_masks_dir = project_dir + '/datasets/train/masks'
test_images_dir = project_dir + '/datasets/test/images'
test_masks_dir = project_dir + '/datasets/test/masks'

# List of images names in order
src_images_list = sorted(os.listdir(src_images_dir))
src_masks_list = sorted(os.listdir(src_masks_dir))

# Number of images in the dataset
total_img_num = len(src_images_list)


# Select images from the dataset and split them into train and test sets according to 
# a specified ratio for the train subset
def select_train_test_images(ratio):
    clear_test_train_folders()
    train_indices, test_indices = randomize_train_test_indices(list(range(total_img_num)), ratio)
    
    [copy_to_train_img(x) for x in train_indices]
    [copy_to_train_mask(x) for x in train_indices]
    [copy_to_test_img(x) for x in test_indices]
    [copy_to_test_mask(x) for x in test_indices]


# Given a list of indices from 0 to total_img_num, randomly split them into two lists
# where the number of train images is a ratio of the dataset
def randomize_train_test_indices(num_list, ratio1):
    list1_length = round(len(num_list) * ratio1)
    list1 = random.sample(num_list, list1_length)
    list2 = [x for x in num_list if x not in (list1)]

    return list1, list2


# Copy images to the train/images folder
def copy_to_train_img(i):
    img_path = src_images_dir + '/' + src_images_list[i]
    shutil.copy(img_path, train_images_dir)


# Copy masks to the train/masks folder
def copy_to_train_mask(i):
    img_path = src_masks_dir + '/' + src_masks_list[i]
    shutil.copy(img_path, train_masks_dir)


# Copy images to the test/images folder
def copy_to_test_img(i):
    img_path = src_images_dir + '/' + src_images_list[i]
    shutil.copy(img_path, test_images_dir)


# Copy images to the train/images folder
def copy_to_test_mask(i):
    img_path = src_masks_dir + '/' + src_masks_list[i]
    shutil.copy(img_path, test_masks_dir)


# Clear all images and masks in the train and test directories
def clear_test_train_folders():
    [os.remove(train_images_dir + '/' + x) for x in os.listdir(train_images_dir)]
    [os.remove(train_masks_dir + '/' + x) for x in os.listdir(train_masks_dir)]
    [os.remove(test_images_dir + '/' + x) for x in os.listdir(test_images_dir)]
    [os.remove(test_masks_dir + '/' + x) for x in os.listdir(test_masks_dir)]
    print("Cleared train and test images and masks")


# clear_test_train_folders()
select_train_test_images(0.8)

# Verify images are split according to ratio
print("Number of images in train/images: " + str(len(os.listdir(train_images_dir))))
print("Number of images in train/masks: " + str(len(os.listdir(train_masks_dir))))
print("Number of images in test/images: " + str(len(os.listdir(test_images_dir))))
print("Number of images in test/masks: " + str(len(os.listdir(test_masks_dir))))