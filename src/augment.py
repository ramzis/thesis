import os
import cv2
import imageio
import matplotlib.image as mpimg
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap


def create_root_and_subdirs(augmentation_dir, original_paths):
    '''
    Checks if augmented data folder exists, else creates it.
    Creates subdirectories based on original image paths.

    augmentation_dir - root directory to save images
    original_paths   - list of images to augment
    '''
    print("Creating directories for augmented data...")

    for image_path in original_paths:

        directory = os.path.basename(os.path.dirname(image_path))

        new_path = os.path.join(augmentation_dir, directory)

        if not os.path.exists(new_path):
            os.makedirs(new_path)

    print("Directories created!")


def augment_data(augmentation_dir, original_paths, original_labels,
                 resize_size, n_augmentations):
    '''
    Creates and saves augmented images.
    For each original image n_augmentations images are created.
    Returns lists of augmented data paths and labels.

    augmentation_dir - root directory to save images
    original_paths   - list of images to augment
    original_labels  - list of image labels
    resize_size      - dimension to resize images
    n_augmentations  - number of images to synthesize for each original
    '''

    # List of augmented image paths
    X = []
    # List of augmented class labels
    Y = []

    # Creates folder structure to store augmented data
    create_root_and_subdirs(augmentation_dir, original_paths)

    print("Augmenting data...")

    for index, image_path in enumerate(original_paths):

        # Find paths to store augmented image
        subdirectory = os.path.basename(os.path.dirname(image_path))
        new_path = os.path.join(augmentation_dir, subdirectory)

        # Read original image
        image = mpimg.imread(image_path)[:, :, :3]  # Do not read alpha channel.

        # Resize image
        resized_image = cv2.resize(image, (resize_size, resize_size))

        for idx in range(n_augmentations):

            # Augment image and save
            seq = iaa.Sequential([
                iaa.ContrastNormalization((0.5, 1.2)),
                iaa.Multiply((0.5, 1.2)),
                iaa.Affine(
                    scale=(1.0, 1.5),
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    rotate=(-20, 20),
                    # shear=(-8, 8)
                )
                # iaa.Sometimes(0.5, iaa.AddElementwise(
                #    iap.Discretize(
                #        (iap.Beta(0.5, 0.5) * 2 - 1.0) * 64
                #    )
                # )),
            ])
            augmented_image = seq.augment_image(resized_image)
            image_name = "aug_" + str(index) + "_" + str(idx) + ".jpeg"
            augm_path = os.path.join(new_path, image_name)
            imageio.imwrite(augm_path, augmented_image)

            # Update augmented paths and labels
            X.append(augm_path)
            Y.append(original_labels[index])

    print("Done!")

    return X, Y
