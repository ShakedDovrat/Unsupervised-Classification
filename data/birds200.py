import numpy as np
from torch.utils.data import Dataset
from PIL import Image

from utils.utils import load_from_pickle


class Birds200_2011(Dataset):
    # pickle_path = r'E:\datasets\Caltech-UCSD-Birds-200-2011\CUB_200_2011\parsed\data_filtered_numpy_32_32_unnormed.pkl'
    train_pickle_path = r'E:\datasets\Caltech-UCSD-Birds-200-2011\CUB_200_2011\parsed\data_filtered_numpy_64_64_unnormed_train.pkl'
    val_pickle_path = r'E:\datasets\Caltech-UCSD-Birds-200-2011\CUB_200_2011\parsed\data_filtered_numpy_64_64_unnormed_val.pkl'

    def __init__(self, is_train, train_path=train_pickle_path, val_path=val_pickle_path, transform=None):
        # TODO: Separate train/test
        super(Birds200_2011, self).__init__()
        self.pickle_file_path = train_path if is_train else val_path
        self.transform = transform

        data, targets = load_from_pickle(self.pickle_file_path)

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

        out = {'image': img, 'target': target[2], 'meta': {'im_size': img_size, 'index': index, 'id': target[0], 'name': target[1]}}
        # print('__getitem__, img shape:', img.size, 'target shape:', np.array(target[2]))
        return out

    def get_image(self, index):
        img = self.data[index]
        # print('get_image, shape:', img.shape)
        return img

    def __len__(self):
        return len(self.data)