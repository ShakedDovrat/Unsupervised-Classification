import os

import numpy as np
import pandas as pd
import tqdm
from PIL import Image

from utils.utils import dump_to_pickle, load_from_pickle


def crop_and_downsample(birds_200_2011_root_dir):
    birds_200_2011_input_image_dir = os.path.join(birds_200_2011_root_dir, r'CUB_200_2011\images')
    birds_200_2011_output_dir = os.path.join(birds_200_2011_root_dir, r'CUB_200_2011\parsed')
    birds_200_2011_output_image_dir = os.path.join(birds_200_2011_output_dir, 'jpgs')
    out_pickle_path = os.path.join(birds_200_2011_output_dir, 'data.pkl')

    image_out_size = 64  # [pixels] (square)

    image_dirs = os.listdir(birds_200_2011_input_image_dir)

    images_txt_path = os.path.join(birds_200_2011_root_dir, r'CUB_200_2011\images.txt')
    # # attributes_txt_path = os.path.join(birds_200_2011_root_dir, r'CUB_200_2011\attributes\image_attribute_labels.txt')
    # attributes_txt_path = os.path.join(birds_200_2011_root_dir, r'CUB_200_2011\attributes\image_attribute_labels_FIXED_6_columns.txt')
    classes_txt_path = os.path.join(birds_200_2011_root_dir, r'CUB_200_2011\image_class_labels.txt')
    split_txt_path = os.path.join(birds_200_2011_root_dir, r'CUB_200_2011\train_test_split.txt')

    images_pd = pd.read_csv(images_txt_path, delim_whitespace=True, header=None)
    # attributes_pd = pd.read_csv(attributes_txt_path, delim_whitespace=True, header=None)
    classes_pd = pd.read_csv(classes_txt_path, delim_whitespace=True, header=None)
    split_pd = pd.read_csv(split_txt_path, delim_whitespace=True, header=None)

    images = []
    ids = []
    for d in tqdm.tqdm(image_dirs, desc='Cropping & downsampling Birds 200-2011 images'):

        image_names = os.listdir(os.path.join(birds_200_2011_input_image_dir, d))
        for image_name in image_names:
            in_path = os.path.join(birds_200_2011_input_image_dir, d, image_name)
            out_path = os.path.join(birds_200_2011_output_image_dir, image_name)
            image = Image.open(in_path)

            if image.height > image.width:
                center_h = image.height // 2
                y0 = center_h - image.width // 2
                y1 = center_h + image.width // 2
                x0 = 0
                x1 = image.width
            else:
                center_w = image.width // 2
                x0 = center_w - image.height // 2
                x1 = center_w + image.height // 2
                y0 = 0
                y1 = image.height
            cropped = image.resize((image_out_size, image_out_size), box=(x0, y0, x1, y1))

            id = images_pd[images_pd[1].str.contains(image_name)].to_numpy()[0]
            assert id.shape == (2,)
            klass = classes_pd[1][classes_pd[0] == id[0]].to_numpy()[0]
            klass -= 1  # class nums should be zero-based...
            is_training_image = split_pd[1][split_pd[0] == id[0]].to_numpy()[0]
            target = np.concatenate((id, [klass, is_training_image]))

            images.append(cropped)
            ids.append(target)
            cropped.save(out_path, "JPEG")

    dump_to_pickle((images, ids), out_pickle_path)


def to_numpy_format(birds_200_2011_root_dir):
    birds_200_2011_output_dir = os.path.join(birds_200_2011_root_dir, r'CUB_200_2011\parsed')
    in_pickle_path = os.path.join(birds_200_2011_output_dir, 'data.pkl')
    out_pickle_path = os.path.join(birds_200_2011_output_dir, 'data_filtered.pkl')
    out_pickle_path2 = os.path.join(birds_200_2011_output_dir, 'data_filtered_numpy.pkl')
    out_pickle_path3 = os.path.join(birds_200_2011_output_dir, 'data_filtered_numpy_32_32_unnormed.pkl')
    out_pickle_path4_train = os.path.join(birds_200_2011_output_dir, 'data_filtered_numpy_64_64_unnormed_train.pkl')
    out_pickle_path4_val = os.path.join(birds_200_2011_output_dir, 'data_filtered_numpy_64_64_unnormed_val.pkl')

    images, targets = load_from_pickle(in_pickle_path)

    images_np = [(np.array(image), image, target) for image, target in zip(images, targets) if image.mode == 'RGB']

    images_stack = np.stack([i[0] for i in images_np])

    images_stack_normed = images_stack / 255
    rgb_mean = np.mean(images_stack_normed, axis=(0, 1, 2))
    rgb_std = np.std(images_stack_normed, axis=(0, 1, 2))
    print('RGB mean:', rgb_mean)
    print('RGB std:', rgb_std)

    dump_to_pickle([a[1:] for a in images_np], out_pickle_path)

    targets_ = [a[2] for a in images_np]
    dump_to_pickle((images_stack_normed, targets_), out_pickle_path2)

    images_stack = np.stack([np.array(a[1].resize((32, 32))) for a in images_np])

    # images_stack_normed = images_stack / 255
    # rgb_mean = np.mean(images_stack, axis=(0, 1, 2))
    # rgb_std = np.std(images_stack, axis=(0, 1, 2))
    # print('RGB mean:', rgb_mean)
    # print('RGB std:', rgb_std)

    dump_to_pickle((images_stack, targets_), out_pickle_path3)

    images_stack = np.stack([np.array(a[1].resize((64, 64))) for a in images_np])

    targets__ = np.stack(targets_)
    is_training_image = targets__[:, 3].astype(np.bool)
    train_set = images_stack[is_training_image]
    val_set = images_stack[~is_training_image]
    train_targets = targets__[is_training_image]
    val_targets = targets__[~is_training_image]

    dump_to_pickle((train_set, train_targets), out_pickle_path4_train)
    dump_to_pickle((val_set, val_targets), out_pickle_path4_val)


def main():
    birds_200_2011_root_dir = r'E:\datasets\Caltech-UCSD-Birds-200-2011'
    crop_and_downsample(birds_200_2011_root_dir)
    to_numpy_format(birds_200_2011_root_dir)


if __name__ == '__main__':
    main()
