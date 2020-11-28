import torch
from torch.utils.data import Dataset
from PIL import Image
# from torchvision.transforms import ToTensor

from utils.utils import load_from_pickle


class Birds200_2011(Dataset):
    # pickle_path = r'E:\datasets\Caltech-UCSD-Birds-200-2011\CUB_200_2011\parsed\data_filtered_numpy_32_32_unnormed.pkl'
    train_pickle_path = r'E:\datasets\Caltech-UCSD-Birds-200-2011\CUB_200_2011\parsed\data_filtered_numpy_64_64_unnormed_train.pkl'
    val_pickle_path = r'E:\datasets\Caltech-UCSD-Birds-200-2011\CUB_200_2011\parsed\data_filtered_numpy_64_64_unnormed_val.pkl'
    num_attributes = 312

    def __init__(self, is_train, targets_type, train_path=train_pickle_path, val_path=val_pickle_path, transform=None):
        super(Birds200_2011, self).__init__()
        self.pickle_file_path = train_path if is_train else val_path
        self.targets_type = targets_type
        self.transform = transform

        data, targets = load_from_pickle(self.pickle_file_path)
        if self.targets_type == 'attributes':
            self.attributes_targets = self._load_attributes(targets)
        else:
            assert self.targets_type == 'class', f"Unrecognized targets type {self.targets_type}"

        self.data = data
        self.targets = targets
        # self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'image': image, 'target': index of target class, 'meta': dict}
        """
        img, target = self.data[index], self.targets[index]
        img_size = (img.shape[0], img.shape[1])
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        # if self.targets_type == 'class':
        #     out = {'image': img, 'target': target[2], 'meta': {'im_size': img_size, 'index': index, 'id': target[0], 'name': target[1]}}
        # elif self.targets_type == 'attributes':
        #     out = {'image': img, 'target': self.attributes_targets[index], 'meta': {'im_size': img_size, 'index': index, 'id': target[0], 'name': target[1]}}
        target_ = target[2] if self.targets_type == 'class' else self.attributes_targets[index]#torch.from_numpy(self.attributes_targets[index])#.cuda(non_blocking=True)
        return {'image': img, 'target': target_, 'meta': {'im_size': img_size, 'index': index, 'id': target[0], 'name': target[1]}}

    def get_image(self, index):
        img = self.data[index]
        # print('get_image, shape:', img.shape)
        return img

    def __len__(self):
        return len(self.data)

    def _load_attributes(self, targets):
        import os
        import numpy as np
        import pandas as pd
        attributes_txt_path = os.path.join(self.pickle_file_path, '..', '..', r'attributes\image_attribute_labels_FIXED_6_columns.txt')
        attributes_pd = pd.read_csv(attributes_txt_path, delim_whitespace=True, header=None)
        attributes_pd.columns = ['image_id', 'attribute_id', 'is_present', 'certainty_id', 'time']
        attributes_per_image = attributes_pd.groupby('image_id')
        attribute_targets = np.zeros((targets.shape[0], Birds200_2011.num_attributes), dtype=np.float)
        for i, id in enumerate(targets[:, 0]):
            group = attributes_per_image.get_group(id)
            # if group.shape[0] == Birds200_2011.num_attributes:
            #     attribute_targets[i, :] = group(id)['is_present']
            # else:
            #     indices = zeros
            attribute_targets[i, group['attribute_id'] - 1] = group['is_present']

        # To bill shape classes:
        bill_shapes = attribute_targets[:, 0:9]
        bill_shapes_sum = bill_shapes.sum(axis=1)
        # assert np.all((bill_shapes_sum == 0) | (bill_shapes_sum == 1))
        bill_shapes = np.concatenate((bill_shapes, np.expand_dims(bill_shapes_sum == 0, axis=1)), axis=1)
        assert np.all(bill_shapes.sum(axis=1) == 1)

        return np.where(bill_shapes)[1]#bill_shapes.astype(np.int64)
