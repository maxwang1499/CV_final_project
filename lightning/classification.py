import os
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader 

from models import get_model
from eval import get_loss_fn, MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from data import MultiTaskClassificationDataset, SingleTaskClassificationDataset
from util import constants as C
from .logger import TFLogger

class ClassificationTask(pl.LightningModule, TFLogger):
    """Standard interface for the trainer to interact with the model."""
    
    def __init__(self, params):
        super().__init__()
     
        self.dataset_folder = params['dataset_folder']
        self.augmentation = params['augmentation']
        self.products = params['products']
        self.image_size = params['image_size']
        self.crop_size = params['crop_size']
        self.save_dir = params['save_dir']
        self.exp_name = params['exp_name']

        if not self.products.endswith('-rgb'):
            params['model'] = params['model'] + '_' + self.products
        self.save_hyperparameters(params)
        self.model = get_model(params)
        self.loss = get_loss_fn(params)

    def forward(self, x):
        return self.model(x)

    def validation_epoch_end(self, outputs):
        """
        Aggregate and return the validation metrics
        Args:
        outputs: A list of dictionaries of metrics from `validation_step()'
        Returns: None
        Returns:
            A dictionary of loss and metrics, with:
                val_loss (required): validation_loss
                log: metrics to be logged to the TensorBoard and metrics.csv
                progress_bar: metrics to be logged to the progress bar
                              and metrics.csv
        """
        avg_loss = torch.stack(outputs).mean()
        self.log("val_loss", avg_loss)
        metrics = self.evaluator.evaluate()
        self.evaluator.reset()
        self.log_dict(metrics)
    
    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])]

 
class MultiTaskClassificationTask(ClassificationTask):
    def __init__(self, params):
        super().__init__(params)
        
        if params["loss_fn"] == "W_BCE": # Initialize weights vector if I am using weighted BCE
            df = pd.read_csv(os.path.join(self.dataset_folder, 'train_dataset.csv'))
            total_samples = df.shape[0]
            pos_weights = [0] * len(C.class_labels_list)
            for i in range(len(C.class_labels_list)):
                class_count = df[df['Type'].str.contains(C.class_labels_list[i])].shape[0]
                # Number of datapoints which contain my class (ie num of positive samples)
                pos_weights[i] = (total_samples-class_count)/class_count
                # Positive weight = num of negative samples / num of positive samples
            pos_weights = torch.FloatTensor(pos_weights)
            self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)
            
        self.evaluator = MulticlassClassificationEvaluator(threshold=None, save_dir=self.save_dir, exp_name = self.exp_name)
            
    def training_step(self, batch, batch_nb):
        """
        Returns:
            A dictionary of loss and metrics, with:
                loss(required): loss used to calculate the gradient
                log: metrics to be logged to the TensorBoard and metrics.csv
                progress_bar: metrics to be logged to the progress bar
                              and metrics.csv
        """
        _, _, y, x = batch
        # for now, just cut off the 4th channel of x
        x = x[:,:3,:,:]
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_nb):
        _, _, y, x = batch
        # for now, just cut off the 4th channel of x
        x = x[:,:3,:,:]
        logits = self.forward(x)
        loss = self.loss(logits, y)
        y_hat = (logits > 0).float()
        self.evaluator.update((torch.sigmoid(logits), y))
        return loss

    def train_dataloader(self):
        dataset = MultiTaskClassificationDataset(os.path.join(self.dataset_folder, 'train_dataset.csv'),
                                                split="train",
                                                products_to_use=self.products,
                                                augmentation=self.augmentation,
                                                image_size=self.image_size,
                                                crop_size=self.crop_size,
                                                pretrained=True)
        return DataLoader(dataset, shuffle=True,
                          batch_size=self.hparams['batch_size'], num_workers=self.hparams['num_workers'])

    def val_dataloader(self):
        dataset = MultiTaskClassificationDataset(os.path.join(self.dataset_folder, 'val_dataset.csv'),
                                                split="valid",
                                                products_to_use=self.products,
                                                augmentation="none",
                                                image_size=self.image_size,
                                                crop_size=self.crop_size,
                                                pretrained=True)
        return DataLoader(dataset, shuffle=False,
                          batch_size=self.hparams['batch_size'], num_workers=self.hparams['num_workers'])

    def test_dataloader(self):
        dataset = MultiTaskClassificationDataset(os.path.join(self.dataset_folder, 'test_dataset.csv'),
                                                split="test",
                                                products_to_use=self.products,
                                                augmentation="none",
                                                image_size=self.image_size,
                                                crop_size=self.crop_size,
                                                pretrained=True)
        return DataLoader(dataset, shuffle=False,
                          batch_size=1, num_workers=self.hparams['num_workers'])


class SingleTaskClassificationTask(ClassificationTask):
    def __init__(self, params):
        super().__init__(params)
        self.task = params['task_type']
        self.evaluator = BinaryClassificationEvaluator(threshold=None, save_dir=self.save_dir, exp_name = self.exp_name)

    def training_step(self, batch, batch_nb):
        """
        Returns:
            A dictionary of loss and metrics, with:
                loss(required): loss used to calculate the gradient
                log: metrics to be logged to the TensorBoard and metrics.csv
                progress_bar: metrics to be logged to the progress bar
                              and metrics.csv
        """
        _, _, y, x = batch
        # for now, just cut off the 4th channel of x
        x = x[:,:3,:,:]
        logits = self.forward(x)
        loss = self.loss(logits.view(-1), y)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_nb):
        _, _, y, x = batch
        # for now, just cut off the 4th channel of x
        x = x[:,:3,:,:]
        logits = self.forward(x)
        loss = self.loss(logits.view(-1), y)
        y_hat = (logits > 0).float()
        self.evaluator.update((torch.sigmoid(logits), y))
        return loss

    def train_dataloader(self):
        dataset = SingleTaskClassificationDataset(self.task, os.path.join(self.dataset_folder, 'train_dataset.csv'),
                                                split="train",
                                                augmentation=self.augmentation,
                                                image_size=500,
                                                pretrained=True)
        return DataLoader(dataset, shuffle=True,
                          batch_size=self.hparams['batch_size'], num_workers=self.hparams['num_workers'])

    def val_dataloader(self):
        dataset = SingleTaskClassificationDataset(self.task, os.path.join(self.dataset_folder, 'val_dataset.csv'),
                                                split="valid",
                                                augmentation="none",
                                                image_size=500,
                                                pretrained=True)
        return DataLoader(dataset, shuffle=False,
                          batch_size=self.hparams['batch_size'], num_workers=self.hparams['num_workers'])

    def test_dataloader(self):
        dataset = SingleTaskClassificationDataset(self.task, os.path.join(self.dataset_folder, 'test_dataset.csv'),
                                                split="test",
                                                augmentation="none",
                                                image_size=500,
                                                pretrained=True)
        return DataLoader(dataset, shuffle=False,
                          batch_size=1, num_workers=self.hparams['num_workers'])
