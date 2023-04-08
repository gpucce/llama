from torch.utils.data import Dataset


class PandasDataset(Dataset):
    def __init__(self, pd_data):
        self.df = pd_data

    def __getitem__(self, x):
        return self.df.iloc[x, :]

    def __len__(self):
        return self.df.shape[0]


def pandas_collate(x):
    return {i: [j[i] for j in x] for i in x[0].index}
