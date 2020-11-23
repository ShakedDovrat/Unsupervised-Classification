from torch.utils.data import Dataset
from PIL import Image

from utils.utils import load_from_pickle


class Birds200_2011(Dataset):
    pickle_path = r'E:\datasets\Caltech-UCSD-Birds-200-2011\CUB_200_2011\parsed\data_filtered_numpy_32_32_unnormed.pkl'

    def __init__(self, pickle_file_path=pickle_path, transform=None):
        # TODO: Separate train/test
        super(Birds200_2011, self).__init__()
        self.pickle_file_path = pickle_file_path
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

        return out

    def get_image(self, index):
        img = self.data[index]
        return img

    def __len__(self):
        return len(self.data)
