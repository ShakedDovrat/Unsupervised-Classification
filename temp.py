import os

import numpy as np
import torchvision
import tqdm
from utils.utils import dump_to_pickle

base_dir = r'E:\datasets\celeb-a'
out_dir = os.path.join(base_dir, 'parsed_img_align_celeba')

part_size = 10000
for dataset_name in ['train', 'valid', 'test']:
    part_num = 0
    dataset = torchvision.datasets.CelebA(r'E:\datasets\celeb-a', dataset_name, ['identity', 'attr'])
    # dataset = TempCelebADataset(dataset)
    images = []
    identities = []
    attributes = []
    # for i, (x, y) in tqdm.tqdm(enumerate(dataset), desc=f'Reading CelebA {dataset_name} Dataset'):
    for i in tqdm.tqdm(range(len(dataset)), desc=f'Reading CelebA {dataset_name} Dataset'):
        x, y = dataset[i]
        images.append(np.array(x).transpose((2, 0, 1)))
        identities.append(y[0].numpy())
        attributes.append(y[1].numpy())

        if i > 0 and i % part_size == 0:
            images = np.stack(images)
            identities = np.stack(identities)
            attributes = np.stack(attributes)
            dump_to_pickle((images, identities, attributes), os.path.join(out_dir, f'{dataset_name}_part{part_num}.pkl'))
            # dump_to_pickle((identities, attributes), os.path.join(out_dir, f'{dataset_name}_targets_part{part_num}.pkl'))
            # np.save(os.path.join(out_dir, f'{dataset_name}_images_part{part_num}.npy'), images)
            del images
            del identities
            del attributes
            images = []
            identities = []
            attributes = []
            part_num += 1
            # dataset = torchvision.datasets.CelebA(r'E:\datasets\celeb-a', dataset_name, ['identity', 'attr'])

    images = np.stack(images)
    identities = np.stack(identities)
    attributes = np.stack(attributes)
    dump_to_pickle((images, identities, attributes), os.path.join(out_dir, f'{dataset_name}_part{part_num}.pkl'))
    # dump_to_pickle((identities, attributes), os.path.join(out_dir, f'{dataset_name}_targets_part{part_num}.pkl'))
    # np.save(os.path.join(out_dir, f'{dataset_name}_images_part{part_num}.npy'), images)
    del images
    del identities
    del attributes
