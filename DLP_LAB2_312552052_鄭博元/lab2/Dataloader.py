import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MIBCI2aDataset(Dataset):
    def _getFeatures(self, filePath):
        """
        Read all the preprocessed data from the file path, read it using np.load,
        and concatenate them into a single numpy array.
        """
        all_files = [os.path.join(filePath, f) for f in os.listdir(filePath)]
        data_list = []
        for file in all_files:
            data = np.load(file)
            data_list.append(data)
        return np.concatenate(data_list, axis=0)

    def _getLabels(self, filePath):
        """
        Read all the preprocessed labels from the file path, read it using np.load,
        and concatenate them into a single numpy array.
        """
        all_files = [os.path.join(filePath, f) for f in os.listdir(filePath)]
        label_list = []
        for file in all_files:
            labels = np.load(file)
            label_list.append(labels)
        return np.concatenate(label_list, axis=0)

    def __init__(self, mode):
        # Remember to change the file path according to different experiments
        assert mode in ['train', 'test', 'finetune']
        if mode == 'train':
            # subject dependent: ./dataset/SD_train/features/ and ./dataset/SD_train/labels/
            # leave-one-subject-out: ./dataset/LOSO_train/features/ and ./dataset/LOSO_train/labels/
            self.features = self._getFeatures(filePath='./dataset/LOSO_train/features/')
            print(self.features.shape)
            self.labels = self._getLabels(filePath='./dataset/LOSO_train/labels/')
        if mode == 'finetune':
            # finetune: ./dataset/FT/features/ and ./dataset/FT/labels/
            self.features = self._getFeatures(filePath='./dataset/FT/features/')
            self.labels = self._getLabels(filePath='./dataset/FT/labels/')
        if mode == 'test':
            # subject dependent: ./dataset/SD_test/features/ and ./dataset/SD_test/labels/
            # leave-one-subject-out and finetune: ./dataset/LOSO_test/features/ and ./dataset/LOSO_test/labels/
            self.features = self._getFeatures(filePath='./dataset/LOSO_test/features/')
            self.labels = self._getLabels(filePath='./dataset/LOSO_test/labels/')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def get_data_loader(mode, batch_size=1, shuffle=True):
    dataset = MIBCI2aDataset(mode)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

    

if __name__ == "__main__":
    
    # 測試訓練數據加載
    print("Testing SD_train data loading...")
    x = get_data_loader(mode='train')
    # 測試測試數據加載
    print("\nTesting SD_test data loading...")
    get_data_loader(mode='test')

    # 測試微調數據加載
    print("\nTesting FT data loading...")
    get_data_loader(mode='finetune')
