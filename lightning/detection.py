import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from ignite.metrics import Accuracy

from models import get_model
from eval import DetectionEvaluator
from data import ImageDetectionDemoDataset
from util import constants as C
from .logger import TFLogger


class DetectionTask(pl.LightningModule, TFLogger):
    """Standard interface for the trainer to interact with the model."""

    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)
        self.model = get_model(params)
        self.evaluator = DetectionEvaluator()

    def training_step(self, batch, batch_nb):
        losses = self.model.forward(batch)
        loss = torch.stack(list(losses.values())).mean()
        return loss

    def validation_step(self, batch, batch_nb):
        losses = self.model.forward(batch)
        loss = torch.stack(list(losses.values())).mean()
        preds = self.model.infer(batch)
        self.evaluator.process(batch, preds)
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()
        self.log("val_loss", avg_loss)
        metrics = self.evaluator.evaluate()
        self.evaluator.reset()
        self.log_dict(metrics, prog_bar=True)

    def test_step(self, batch, batch_nb):
        preds = self.model.infer(batch)
        self.evaluator.process(batch, preds)

    def test_epoch_end(self, outputs):
        metrics = self.evaluator.evaluate()
        self.log_dict(metrics)
        return metrics

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=0.02)]

    def train_dataloader(self):
        dataset = ImageDetectionDemoDataset() 
        return DataLoader(dataset, shuffle=True,
                          batch_size=2, num_workers=8,
                          collate_fn=lambda x: x)

    def val_dataloader(self):
        dataset = ImageDetectionDemoDataset() 
        return DataLoader(dataset, shuffle=False,
                batch_size=1, num_workers=8, collate_fn=lambda x: x)

    def test_dataloader(self):
        dataset = ImageDetectionDemoDataset() 
        return DataLoader(dataset, shuffle=False,
                batch_size=1, num_workers=8, collate_fn=lambda x: x)
