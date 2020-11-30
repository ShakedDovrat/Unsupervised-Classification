import os
import glob

import numpy as np
from torch.utils.data import Dataset
import PIL

from utils.utils import load_from_pickle


class TempCelebADataset(Dataset):
    dataset_dir = r'E:\datasets\celeb-a\parsed_img_align_celeba'

    def __init__(self, split, target_type='identity', attr_index=None, data_dir=dataset_dir, transform=None):
        super(TempCelebADataset, self).__init__()

        # self.images, identities, attributes = load_from_pickle(os.path.join(data_dir, f'{split}.pkl'))
        # files = os.listdir(os.path.join(data_dir, f'{split}_part*.pkl'))
        files = glob.glob(os.path.join(data_dir, f'{split}_part*.pkl'))
        files = [os.path.join(data_dir, f'{split}_part{i}.pkl') for i in range(len(files) // 2)]  # NOTE: TAKING HALF OF DATASET BECAUSE IT'S TOO BIG
        data = [load_from_pickle(f) for f in files]
        self.images, identities, attributes = (np.concatenate(d, axis=0) for d in zip(*data))
        self.images = self.images.transpose((0, 2, 3, 1))

        # assert target_type in ['identity', 'attr'], 'Expected argument `target_type` to be "identity" or "attr".'
        # self.targets = identities - 1 if target_type == 'identity' else attributes
        if target_type == 'identity':
            self.targets = identities - 1
        elif target_type == 'attr':
            assert attr_index is not None, 'Need to provide `attr_index` argument when `target_type`=="identity"'
            self.targets = attributes[:, attr_index]
        else:
            raise ValueError('Expected argument `target_type` to be "identity" or "attr".')
        self.transform = transform

    def __getitem__(self, index):
        img = PIL.Image.fromarray(self.images[index])
        target = int(self.targets[index])
        img_size = img.size

        if self.transform is not None:
            img = self.transform(img)
        return {'image': img, 'target': target, 'meta': {'im_size': img_size, 'index': index}}

    def get_image(self, index):
        return self.images[index]

    def __len__(self):
        return self.images.shape[0]  # 1000
