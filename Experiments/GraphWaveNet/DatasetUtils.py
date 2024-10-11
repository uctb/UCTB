
import numpy as np
from UCTB.preprocess.preprocessor import SplitData
from torch.utils.data import Dataset, DataLoader
import torch


class BaseDataset(Dataset):
    def __init__(self, xs, ys, batch_size, device, pad_with_last_sample=False):
        """
        :param xs: Input data
        :param ys: Labels
        :param pad_with_last_sample: Pad with the last sample to make the number of samples divisible by the batch size.
        """
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.xs = torch.tensor(xs, dtype=torch.float32, device=device)
        self.ys = torch.tensor(ys, dtype=torch.float32, device=device)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]


def load_dataset(uctb_data_loader, batch_size, valid_batch_size=None, test_batch_size=None, device="cpu"):
    # x_train (num_slots, time_steps, num_stations, input_dims)
    # y_train (num_slots, time_steps, num_stations, input_dims)
    data = {}

    # split data
    train_closeness, val_closeness = SplitData.split_data(uctb_data_loader.train_closeness, [0.9, 0.1])
    train_period, val_period = SplitData.split_data(uctb_data_loader.train_period, [0.9, 0.1])
    train_trend, val_trend = SplitData.split_data(uctb_data_loader.train_trend, [0.9, 0.1])
    train_y, val_y = SplitData.split_data(uctb_data_loader.train_y, [0.9, 0.1])

    # train_x = np.concatenate([train_trend, train_period, train_closeness],axis=2).transpose( # [0,3,1,2] [0,2,1,3]
    if uctb_data_loader.period_len > 0 and uctb_data_loader.trend_len > 0:
        data["x_train"] = np.concatenate([train_trend, train_period, train_closeness], axis=2).transpose([0, 3, 1, 2])
        data["x_val"] = np.concatenate([val_trend, val_period, val_closeness], axis=2).transpose([0, 3, 1, 2])
        data["x_test"] = np.concatenate(
            [uctb_data_loader.test_trend, uctb_data_loader.test_period, uctb_data_loader.test_closeness],
            axis=2).transpose([0, 3, 1, 2])
    else:
        data["x_train"] = train_closeness.transpose([0, 3, 1, 2])
        data["x_val"] = val_closeness.transpose([0, 3, 1, 2])
        data["x_test"] = uctb_data_loader.test_closeness.transpose([0, 3, 1, 2])

    data["y_train"] = train_y[:, np.newaxis]
    data["y_val"] = val_y[:, np.newaxis]
    data["y_test"] = uctb_data_loader.test_y[:, np.newaxis]

    print("x_train", data["x_train"].shape)
    print("y_train", data["y_train"].shape)
    print("x_val", data["x_val"].shape)
    print("y_val", data["y_val"].shape)
    print("x_test", data["x_test"].shape)
    print("y_test", data["y_test"].shape)


    train_loader = DataLoader(BaseDataset(data['x_train'].transpose(0, 3, 2, 1), data['y_train'].transpose(0, 3, 2, 1), batch_size, device), batch_size=batch_size)
    val_loader = DataLoader(BaseDataset(data['x_val'].transpose(0, 3, 2, 1), data['y_val'].transpose(0, 3, 2, 1), valid_batch_size, device), batch_size=valid_batch_size)
    test_loader = DataLoader(BaseDataset(data['x_test'].transpose(0, 3, 2, 1), data['y_test'].transpose(0, 3, 2, 1), test_batch_size, device), batch_size=test_batch_size)
    
    return train_loader, val_loader, test_loader