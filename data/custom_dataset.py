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

            # img_size = img[0].size(1)
            # assert img_size == img[0].size(2), f'Expected square image, got ({img_size},{img[0].size(2)})'
            box1 = aug_params1[:4]
            box2 = aug_params2[:4]
            iou = calc_iou(box1, box2)

            sample['aug_labels'] = [
                iou,  # crop (and resize)
                float(aug_params1[4] != aug_params2[4]),  # horizontal flip
                # (brightness_factor, contrast_factor, saturation_factor, hue_factor)
                # (max(aug_params1[5], aug_params2[5]) / min(aug_params1[5], aug_params2[5]) - 1) * 50,  # brightness_factor
                # (max(aug_params1[6], aug_params2[6]) / min(aug_params1[6], aug_params2[6]) - 1) * 50,  # contrast_factor
                # (max(aug_params1[7], aug_params2[7]) / min(aug_params1[7], aug_params2[7]) - 1) * 50,  # saturation_factor
                # max(aug_params1[8], aug_params2[8]) - min(aug_params1[8], aug_params2[8]) - 0.5,  # hue_factor
                # aug_params1[5] / aug_params2[5] - 1,  # brightness_factor
                # aug_params1[6] / aug_params2[6] - 1,  # contrast_factor
                # aug_params1[7] / aug_params2[7] - 1,  # saturation_factor
                # aug_params1[8] - aug_params2[8],  # hue_factor
                aug_params1[5] / aug_params2[5] / 2,  # brightness_factor
                aug_params1[6] / aug_params2[6] / 2,  # contrast_factor
                aug_params1[7] / aug_params2[7] / 2,  # saturation_factor
                (aug_params1[8] - aug_params2[8] + 0.3) * 2,  # hue_factor
                # float(aug_params1[5] or aug_params2[5]),  # color jitter
                float(aug_params1[9] or aug_params2[9])  # grayscale
            ]
            # Debug:
            # print(aug_params1[5] / aug_params2[5], aug_params1[6] / aug_params2[6], aug_params1[7] / aug_params2[7], aug_params1[8] - aug_params2[8])
            # print(aug_params1[5] / aug_params2[5] / 2, aug_params1[6] / aug_params2[6] / 2, aug_params1[7] / aug_params2[7] / 2, (aug_params1[8] - aug_params2[8] + 0.3) * 2)
        else:
            sample['image'] = img
            sample['image_augmented'] = self.augmentation_transform(image)

        # Debug:
        # import torchvision
        # transforms_ = torchvision.transforms.Compose(self.image_transform.transforms[:4])
        # aug_transforms_ = torchvision.transforms.Compose(self.augmentation_transform.transforms[:4])
        # image_ = transforms_(image)
        # aug_image_ = aug_transforms_(image)

        # Debug:
        # from transformations import UnNormalize
        # unNorm = UnNormalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))  # cifar 10
        # I1 = unNorm(sample['image'])
        # I2 = unNorm(sample['image_augmented'])
        # import matplotlib.pyplot as plt
        # plt.imshow(I1.numpy().transpose((1,2,0)))
        # plt.show()
        # print(sample['aug_labels'])

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


def calc_iou(box1, box2):
    intersection = (min(box1[0] + box1[2], box2[0] + box2[2]) - max(box1[0], box2[0])) * \
                   (min(box1[1] + box1[3], box2[1] + box2[3]) - max(box1[1], box2[1]))
    if intersection <= 0:
        return 0.0
    union = box1[2] * box1[3] + box2[2] * box2[3] - intersection
    iou = intersection / union
    assert 0.0 <= iou <= 1.0
    return iou
