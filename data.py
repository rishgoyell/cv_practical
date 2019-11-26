import torch
from torch.utils.data import Dataset, DataLoader
from main import noisy_circle

def generate_data(num_samples, savepath):
        Y = torch.zeros(num_samples, 3)
        X = torch.zeros(num_samples, 200, 200)
        for i in range(num_samples):
            if not ((i + 1) % 10000):
                print(f'Completed {i+1}')
            params, img = noisy_circle(200, 50, 2)
            Y[i] = torch.tensor(params)
            X[i] = torch.tensor(img)
        torch.save([X, Y], savepath)


class circlesData(Dataset):

    def __init__(self, path, inds):
        X, Y = torch.load(path)
        self.X = X[inds]
        self.Y = Y[inds]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


if __name__ == '__main__':
    generate_data(50000, '/Users/rishabh/Documents/scale/data/data.pth')
