import os

import numpy as np

from utils.utils import load_from_pickle, dump_to_pickle

birds_200_2011_root_dir = r'E:\datasets\Caltech-UCSD-Birds-200-2011'
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
