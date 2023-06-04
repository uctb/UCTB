import torch
import torch.nn as nn
import numpy as np
import os
import logging
import math
import time
import copy
from UCTB.preprocess import SplitData
from UCTB.train.LossFunction import masked_mae_loss


class Trainer(object):
    def __init__(self, model, train_loader, val_loader, test_loader,
                 args):
        super(Trainer, self).__init__()
        self.model = model
        self.optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                                          weight_decay=0, amsgrad=False)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        self.train_per_epoch = len(train_loader)
        self.lr_scheduler = None
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
        if args.loss_func == 'mask_mae':
            self.loss = masked_mae_loss(mask_value=0.0)
        elif args.loss_func == 'mae':
            self.loss = torch.nn.L1Loss().to(args.device)
        elif args.loss_func == 'mse':
            self.loss = torch.nn.MSELoss().to(args.device)
        else:
            raise ValueError
        if args.lr_decay:
            print('Applying learning rate decay.')
            lr_decay_steps = [int(i)
                              for i in list(args.lr_decay_step.split(','))]
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer,
                                                                     milestones=lr_decay_steps,
                                                                     gamma=args.lr_decay_rate)
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
        # log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(
            args.log_dir, name=args.model, debug=args.debug)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        # if not args.debug:
        # self.logger.info("Argument: %r", args)
        # for arg, value in sorted(vars(args).items()):
        #     self.logger.info("Argument %s: %r", arg, value)

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                label = target[..., :self.args.output_dim]
                output = self.model(data, target, teacher_forcing_ratio=0.)
                #loss = self.loss(output.cuda(), label)
                loss = self.loss(output, label)
                # a whole batch of Metr_LA is filtered
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info(
            '**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        return val_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            label = target[..., :self.args.output_dim]  # (..., 1)
            self.optimizer.zero_grad()

            # teacher_forcing for RNN encoder-decoder model
            # if teacher_forcing_ratio = 1: use label as input in the decoder for all steps
            if self.args.teacher_forcing:
                global_step = (epoch - 1) * self.train_per_epoch + batch_idx
                teacher_forcing_ratio = self._compute_sampling_threshold(
                    global_step, self.args.tf_decay_steps)
            else:
                teacher_forcing_ratio = 1.
            # data and target shape: B, T, N, F; output shape: B, T, N, F
            output = self.model(
                data, target, teacher_forcing_ratio=teacher_forcing_ratio)
            #loss = self.loss(output.cuda(), label)
            loss = self.loss(output, label)
            loss.backward()

            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()

            # log information
            if batch_idx % self.args.log_step == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item()))
        train_epoch_loss = total_loss / self.train_per_epoch
        self.logger.info(
            '**********Train Epoch {}: averaged Loss: {:.6f}, tf_ratio: {:.6f}'.format(epoch, train_epoch_loss,
                                                                                       teacher_forcing_ratio))

        # learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss

    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            # epoch_time = time.time()
            train_epoch_loss = self.train_epoch(epoch)
            # print(time.time()-epoch_time)
            # exit()
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)

            # print('LR:', self.optimizer.param_groups[0]['lr'])
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            # if self.val_loader == None:
            # val_epoch_loss = train_epoch_loss
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best state
            if best_state == True:
                self.logger.info(
                    '*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())

        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format(
            (training_time / 60), best_loss))

        # save the best model to file
        if not self.args.debug:
            torch.save(best_model, self.best_path)
            self.logger.info("Saving current best model to " + self.best_path)

        # test
        self.model.load_state_dict(best_model)
        # self.val_epoch(self.args.epochs, self.test_loader)
        #self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    @staticmethod
    def test(model, args, data_loader,  logger, path=None):
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                label = target[..., :args.output_dim]
                output = model(data, target, teacher_forcing_ratio=0)
                y_true.append(label)
                y_pred.append(output)
        y_pred = torch.cat(y_pred, dim=0)
        y_pred = y_pred.cpu().numpy()

        return y_pred

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))


def get_dataloader_AGCRN(data_loader, batchsize, tod=False, dow=False, weather=False, single=True):
    # split data
    train_closeness, val_closeness = SplitData.split_data(
        data_loader.train_closeness, [0.9, 0.1])
    train_period, val_period = SplitData.split_data(
        data_loader.train_period, [0.9, 0.1])
    train_trend, val_trend = SplitData.split_data(
        data_loader.train_trend, [0.9, 0.1])
    train_y, val_y = SplitData.split_data(data_loader.train_y, [0.9, 0.1])

    # T, N, D, 1 -> T, D, N, 1
    if data_loader.period_len > 0 and data_loader.trend_len > 0:
        train_x = np.concatenate(
            [train_trend, train_period, train_closeness], axis=2).transpose([0, 2, 1, 3])
        val_x = np.concatenate(
            [val_trend, val_period, val_closeness], axis=2).transpose([0, 2, 1, 3])
        test_x = np.concatenate([data_loader.test_trend, data_loader.test_period, data_loader.test_closeness],
                                axis=2).transpose([0, 2, 1, 3])
    else:
        train_x = train_closeness.transpose([0, 2, 1, 3])
        val_x = val_closeness.transpose([0, 2, 1, 3])
        test_x = data_loader.test_closeness.transpose([0, 2, 1, 3])

    train_y = train_y[:, np.newaxis]
    val_y = val_y[:, np.newaxis]
    test_y = data_loader.test_y[:, np.newaxis]

    print('Train: ', train_x.shape, train_y.shape)
    print('Val: ', val_x.shape, val_y.shape)
    print('Test: ', test_x.shape, test_y.shape)
    ############## get dataloader ######################
    train_dataloader = data_loader_torch(
        train_x, train_y, batchsize, shuffle=True, drop_last=True)
    if len(train_x) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader_torch(
            val_x, val_y, batchsize, shuffle=False, drop_last=True)
    test_dataloader = data_loader_torch(
        test_x, test_y, batchsize, shuffle=False, drop_last=False)
    return train_dataloader, val_dataloader, test_dataloader


def data_loader_torch(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader


def get_logger(root, name=None, debug=True):
    #when debug is true, show DEBUG and INFO in screen
    #when debug is false, show DEBUG in file and info in both screen&file
    #INFO will always be in screen
    # create a logger
    logger = logging.getLogger(name)
    #critical > error > warning > info > debug > notset
    logger.setLevel(logging.DEBUG)

    # define the formate
    formatter = logging.Formatter('%(asctime)s: %(message)s', "%Y-%m-%d %H:%M")
    # create another handler for output log to console
    console_handler = logging.StreamHandler()
    if debug:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)
        # create a handler for write log to file
        logfile = os.path.join(root, 'run.log')
        print('Creat Log File in: ', logfile)
        file_handler = logging.FileHandler(logfile, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    # add Handler to logger
    logger.addHandler(console_handler)
    if not debug:
        logger.addHandler(file_handler)
    return logger
