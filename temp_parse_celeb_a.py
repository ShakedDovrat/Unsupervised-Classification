import os
import csv

import tqdm
import pandas as pd
from PIL import Image

# import torchvision
#
# ds = torchvision.datasets.CelebA(r'E:\datasets\celeb-a', 'train')#, download=True)

celeb_a_root_dir = r'E:\datasets\celeb-a'
celeb_a_input_image_dir = os.path.join(celeb_a_root_dir, r'celeba\img_align_celeba_ORIG')
celeb_a_output_image_dir = os.path.join(celeb_a_root_dir, r'celeba\img_align_celeba')
celeb_a_bbox_file_path = os.path.join(celeb_a_root_dir, r'celeba\list_bbox_celeba.txt')

bboxes = pd.read_csv(celeb_a_bbox_file_path, delim_whitespace=True, skiprows=1)

image_out_size = 64  # [pixels] (square)

for index, row in tqdm.tqdm(bboxes.iterrows(), desc='Cropping & downsampling CelebA images'):
    # print(index, row)
    in_path = os.path.join(celeb_a_input_image_dir, row['image_id'])
    out_path = os.path.join(celeb_a_output_image_dir, row['image_id'])
    image = Image.open(in_path)
    # cropped = image.crop([row['x_1'], row['y_1'], row['x_1']+row['width'], row['y_1']+row['height']])

    # center = image.width // 2, image.height // 2
    # xy1 = center[0] - image_out_size // 2, center[1] - image_out_size // 2
    # xy2 = center[0] + image_out_size // 2, center[1] + image_out_size // 2
    # cropped = image.crop([xy1[0], xy1[1], xy2[0], xy2[1]])#image_out_size, image_out_size])

    assert image.height > image.width
    center_h = image.height // 2
    y0 = center_h - image.width // 2
    y1 = center_h + image.width // 2
    x0 = 0
    x1 = image.width
    # cropped = image.crop([x0, y0, x1, y1])
    cropped = image.resize((64, 64), box=(x0, y0, x1, y1))

    cropped.save(out_path, "JPEG")
    pass