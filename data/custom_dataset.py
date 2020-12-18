"""
Author: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy as np
import torch
from torch.utils.data import Dataset

""" 
    AugmentedDataset
    Returns an image together with an augmentation.
"""
class AugmentedDataset(Dataset):
    def __init__(self, dataset):
        super(AugmentedDataset, self).__init__()
        transform = dataset.transform
        dataset.transform = None
        self.dataset = dataset
        
        if isinstance(transform, dict):
            self.image_transform = transform['standard']
            self.augmentation_transform = transform['augment']

        else:
            self.image_transform = transform
            self.augmentation_transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset.__getitem__(index)
        image = sample['image']

        # img = self.image_transform(image)
        # if isinstance(img, tuple):
        #     sample['image'], sample['image_aug_labels'] = img
        # else:
        #     sample['image'] = img
        #
        # img = self.augmentation_transform(image)
        # if isinstance(img, tuple):
        #     sample['image_augmented'], sample['image_augmented_aug_labels'] = img
        # else:
        #     sample['image_augmented'] = img

        img = self.image_transform(image)
        if isinstance(img, tuple):
            sample['image'], aug_params1 = img
            sample['image_augmented'], aug_params2 = self.augmentation_transform(image)

            sample['aug_labels'] = [
                float(aug_params1[4] != aug_params2[4]),  # horizontal flip
                float(aug_params1[5] or aug_params2[5]),  # color jitter
                float(aug_params1[6] or aug_params2[6])  # grayscale
            ]
        else:
            sample['image'] = img
            sample['image_augmented'] = self.augmentation_transform(image)

        # Debug:
        # import torchvision
        # transforms_ = torchvision.transforms.Compose(self.image_transform.transforms[:4])
        # aug_transforms_ = torchvision.transforms.Compose(self.augmentation_transform.transforms[:4])
        # image_ = transforms_(image)
        # aug_image_ = aug_transforms_(image)

        return sample


""" 
    NeighborsDataset
    Returns an image with one of its neighbors.
"""
class NeighborsDataset(Dataset):
    def __init__(self, dataset, indices, num_neighbors=None):
        super(NeighborsDataset, self).__init__()
        transform = dataset.transform
        
        if isinstance(transform, dict):
            self.anchor_transform = transform['standard']
            self.neighbor_transform = transform['augment']
        else:
            self.anchor_transform = transform
            self.neighbor_transform = transform
       
        dataset.transform = None
        self.dataset = dataset
        self.indices = indices # Nearest neighbor indices (np.array  [len(dataset) x k])
        if num_neighbors is not None:
            self.indices = self.indices[:, :num_neighbors+1]
        assert(self.indices.shape[0] == len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        anchor = self.dataset.__getitem__(index)
        
        neighbor_index = np.random.choice(self.indices[index], 1)[0]
        neighbor = self.dataset.__getitem__(neighbor_index)

        anchor['image'] = self.anchor_transform(anchor['image'])
        neighbor['image'] = self.neighbor_transform(neighbor['image'])

        output['anchor'] = anchor['image']
        output['neighbor'] = neighbor['image'] 
        output['possible_neighbors'] = torch.from_numpy(self.indices[index])
        output['target'] = anchor['target']
        
        return output
