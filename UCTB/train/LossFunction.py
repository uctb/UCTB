import numpy as np
import torch
import tensorflow as tf
import mxnet as mx

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    # print(mask.sum())
    # print(mask.shape[0]*mask.shape[1]*mask.shape[2])
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels,
                                 null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def mae_loss(pred, label):
    mask = tf.not_equal(label, 0)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    # mask = tf.compat.v2.where(
    #     condition = tf.math.is_nan(mask), x = 0., y = mask)
    mask = tf.compat.v1.where(
        condition=tf.math.is_nan(mask), x=tf.zeros_like(mask), y=mask)
    loss = tf.abs(tf.subtract(pred, label))
    loss *= mask
    # loss = tf.compat.v2.where(
    #     condition = tf.math.is_nan(loss), x = 0., y = loss)
    loss = tf.compat.v1.where(
        condition=tf.math.is_nan(loss), x=tf.zeros_like(mask), y=loss)
    loss = tf.reduce_mean(loss)
    return loss

def mask_np(array, null_val):
    '''
    function for STSGCN
    '''
    if np.isnan(null_val):
        return (~np.isnan(null_val)).astype('float32')
    else:
        return np.not_equal(array, null_val).astype('float32')


def masked_mse_np(y_true, y_pred, null_val=np.nan):
    '''
    function for STSGCN
    '''
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mse = (y_true - y_pred) ** 2
    return np.mean(np.nan_to_num(mask * mse))


def masked_mae_np(y_true, y_pred, null_val=np.nan):
    '''
    function for STSGCN
    '''
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mae = np.abs(y_true - y_pred)
    return np.mean(np.nan_to_num(mask * mae))


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    '''
    function for STSGCN
    '''
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mask_np(y_true, null_val)
        mask /= mask.mean()
        mape = np.abs((y_pred - y_true) / y_true)
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


def masked_mae_loss(mask_value):
    '''
    function for AGCRN
    '''
    def loss(preds, labels):
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss

def MAE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true-pred))



def huber_loss(data, label, rho=1):
    '''
    Parameters
    ----------
    data: mx.sym.var, shape is (B, T', N)

    label: mx.sym.var, shape is (B, T', N)

    rho: float

    Returns
    ----------
    loss: mx.sym
    '''

    loss = mx.sym.abs(data - label)
    loss = mx.sym.where(loss > rho, loss - 0.5 * rho,
                        (0.5 / rho) * mx.sym.square(loss))
    loss = mx.sym.MakeLoss(loss)
    return loss
