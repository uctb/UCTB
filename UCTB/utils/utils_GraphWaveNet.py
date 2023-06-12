import os
import time
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from UCTB.preprocess.preprocessor import SplitData
from UCTB.model.GraphWaveNet import *
from UCTB.train.LossFunction import masked_mape, masked_mae, masked_rmse


class trainer():
    def __init__(self, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports, gcn_bool, addaptadj, aptinit):
        self.model = gwnet(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit, in_dim=in_dim, out_dim=seq_length, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = masked_mae
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input,(1,0,0,0))
        # print("input",input.shape)
        output = self.model(input)
        output = output.transpose(1,3)
        # print("ouput",output.shape)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = output

        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = masked_mape(predict,real,0.0).item()
        rmse = masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = output
        loss = self.loss(predict, real, 0.0)
        mape = masked_mape(predict,real,0.0).item()
        rmse = masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse


def Training(args, dataloader,  device, engine):
    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    for i in range(1, args.epochs + 1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:, 0, :, :])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)
        t2 = time.time()
        train_time.append(t2 - t1)
        # validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:, 0, :, :])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),
              flush=True)
        torch.save(engine.model.state_dict(),
                   os.path.join(args.save, "epoch_" + str(i) + "_" + str(round(mvalid_loss, 2)) + ".pth"))
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    return np.argmin(his_loss), his_loss[np.argmin(his_loss)]

def Test(args,dataloader,device,engine,epoch_id, loss_id):
    # Test
    engine.model.load_state_dict(torch.load(
        os.path.join(args.save, "epoch_" + str(epoch_id + 1) + "_" + str(round(loss_id, 2)) + ".pth")))

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    print("realy", realy.shape)

    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx).transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0).cpu().numpy()
    yhat = yhat[:realy.size(0), ...]
    realy = realy.cpu().numpy()

    return yhat

class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()



def load_dataset(uctb_data_loader, batch_size, valid_batch_size=None, test_batch_size=None):
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



    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    return data