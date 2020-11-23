from torch.utils.data import Dataset


class TempCelebADataset(Dataset):
    def __init__(self, dataset, transform=None):
        # TODO: Separate train/test
        super(TempCelebADataset, self).__init__()
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.dataset.__getitem__(index)
        return {'image': img, 'target': target}

    def get_image(self, index):
        return self.dataset.__getitem__(index)[0]

    def __len__(self):
        return len(self.dataset)
