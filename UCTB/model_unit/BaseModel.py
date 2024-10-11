import lightning
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
import re
from lightning.pytorch.callbacks import EarlyStopping

class BaseModel(lightning.LightningModule):
    def __init__(self, model: nn.Module = None, learning_rate: float = 0.001, loss_fn=nn.MSELoss(),freeze_list=None):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.validation_step_outputs = []

        self.freeze_list = freeze_list or []
        if freeze_list:
            set_trainable_parameters(self.model, self.freeze_list)
    
    def forward(self, batch):
        # the model should be a callable object
        x, y = batch
        return self.model(x) 
    
    def training_step(self, batch):
        x, y = batch
        logits = self(batch)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss, prog_bar=True, on_step=True)
        return loss
    
    def validation_step(self, batch):
        x, y = batch
        logits = self(batch)
        loss = self.loss_fn(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        
        msg = {"prediction": logits, "target": y,"loss":loss}
        self.validation_step_outputs.append(msg)
        return msg

    def on_validation_epoch_end(self):
        # record batch loss and prediction
        predictions = torch.cat([x['prediction'] for x in self.validation_step_outputs])
        targets = torch.cat([x['target'] for x in self.validation_step_outputs])
        
        self.validation_step_outputs.clear()
        
        return {"predictions": predictions, "targets": targets}

    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('test_loss', loss, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def set_trainable_parameters(model, freeze_list):
    """
    Function to freeze certain parameters and allow others to be trainable.

    Args:
    - model: The PyTorch model or nn.Module that contains the parameters.
    - freeze_list: A list of strings representing the names of the parameters (layers) to be frozen.

    Returns:
    - None
    """
    for name, param in model.named_parameters():
        # Freeze parameters if they are in the freeze_list
        if any(layer_name in name for layer_name in freeze_list):
            param.requires_grad = False
            logging.info(f'Frozen layer: {name}')
        else:
            param.requires_grad = True  # Ensure other parameters are trainable
            logging.info(f'Trainable layer: {name}')


class NaiveEarlyStopping(EarlyStopping):
    def __init__(self, log_dir, *args, **kwargs):
        super(NaiveEarlyStopping, self).__init__(*args, **kwargs)

        self._model_dir = os.path.abspath(log_dir)

    def on_train_end(self, trainer, pl_module):
        # call on_train_end
        self._log("Converged")

        detailed_info = f"Training stopped at epoch: {trainer.current_epoch}. Early stopping patience: {self.patience}. Best val_loss: {trainer.checkpoint_callback.best_model_score}"
        self._log(detailed_info)

        super().on_train_end(trainer, pl_module)

    def _log(self, text):
        save_dir_subscript = self._model_dir
        if os.path.isdir(save_dir_subscript) is False:
            os.makedirs(save_dir_subscript)
        with open(os.path.join(save_dir_subscript, 'log.txt'), 'a+', encoding='utf-8') as f:
            f.write(text + '\n')


class CheckpointManager():
    '''
    Class to manage the checkpoint files and log files.
    '''
    def __init__(self, model_dir):
        super().__init__()
        self._model_dir = model_dir
        # make sure the directory exists
        if not os.path.exists(self._model_dir):
            os.makedirs(self._model_dir)

        # parse the log file to check whether it is converged.
        self.converged = False
        log_list = self._get_log()
        for e in log_list:
            if e.lower() == 'converged':
                self.converged = True

        self.best_checkpoint_path = self._get_best_checkpoint_path()

    def _get_log(self):
        '''
        Function to read the log file and return the list of lines.
        '''
        save_dir_subscript = self._model_dir
        if os.path.isfile(os.path.join(save_dir_subscript, 'log.txt')):
            with open(os.path.join(save_dir_subscript, 'log.txt'), 'r', encoding='utf-8') as f:
                return [e.strip('\n') for e in f.readlines()]
        else:
            return []

    def _get_best_checkpoint_path(self):
        """
        Function to find the best checkpoint file based on the val_loss in the filename.
        """
        # Regular expression to extract "val_loss" from filenames like 'epoch=32-val_loss=2.4.ckpt'
        pattern = re.compile(r'epoch=\d+-val_loss=([\d\.]+)\.ckpt')

        checkpoint_dir = self._model_dir

        # make sure the directory exists
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Get list of all files that match the pattern
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]

        # Dictionary to store filenames and their corresponding val_loss values
        val_loss_map = {}

        # Iterate over the checkpoint files and extract val_loss
        for file in checkpoint_files:
            match = pattern.search(file)
            if match:
                val_loss = float(match.group(1))  # Extract val_loss as a float
                val_loss_map[file] = val_loss

        # If we found any valid checkpoint files
        if val_loss_map:
            # Find the file with the lowest val_loss
            best_checkpoint = min(val_loss_map, key=val_loss_map.get)
            best_checkpoint_path = os.path.join(checkpoint_dir, best_checkpoint)
            print(f"Best checkpoint found: {best_checkpoint_path} with val_loss={val_loss_map[best_checkpoint]}")
            return best_checkpoint_path
        else:
            print(f"No valid checkpoint files found in {checkpoint_dir}, starting from scratch...")
            return None
    
    @staticmethod
    def load_parameters_from_checkpoint(model, checkpoint_path):
        """
        Function to load the parameters from the best checkpoint file into the model.
        """
        if checkpoint_path:
            print(f"Loading model parameters from {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
        else:
            print("No best checkpoint found. Starting from scratch...")


if __name__ == "__main__":
    # Test the BaseModel class
    model = nn.Linear(10, 2)
    model = BaseModel(model)
    # Assume you have a PyTorch model called `model`
    freeze_list = ['layer1', 'layer2', 'conv1.weight']  # List of layer names or parameters you want to freeze
    set_trainable_parameters(model, freeze_list)
