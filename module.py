import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from dataset import FaceIDDataset
from model import FaceIDModel


class FaceIDModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.model = FaceIDModel(self.hparams.cnn_arch, self.hparams.pretrained)
        self.loss = torch.nn.CosineEmbeddingLoss(margin=0.5)
        self.cosine = torch.nn.CosineSimilarity()
        self.threshold = 0.75

    def calculate_stat(self, embed_x0, embed_x1, y):
        cosine = self.cosine(embed_x0, embed_x1)
        same_class_cosine = cosine[y == 1]
        diff_class_cosine = cosine[y == -1]
        true_positive = (same_class_cosine > self.threshold).sum()
        true_negative = (diff_class_cosine < self.threshold).sum()
        accuracy = (true_positive + true_negative) / float(y.size(0))
        return same_class_cosine.mean(), diff_class_cosine.mean(), accuracy

    def forward(self, x0, x1):
        embed_x0 = self.model(x0)
        embed_x1 = self.model(x1)
        return embed_x0, embed_x1

    def training_step(self, batch, batch_nb):
        x0, x1, y = batch
        embed_x0, embed_x1 = self.forward(x0, x1)
        loss = self.loss(embed_x0, embed_x1, y)
        logs = {"loss/train": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_nb):
        x0, x1, y = batch
        embed_x0, embed_x1 = self.forward(x0, x1)
        loss = self.loss(embed_x0, embed_x1, y)
        same_class_cosine, diff_class_cosine, accuracy = self.calculate_stat(
            embed_x0, embed_x1, y
        )
        logs = {
            "loss/val": loss,
            "cosine/same_class": same_class_cosine,
            "cosine/diff_class": diff_class_cosine,
            "accuracy": accuracy,
        }
        return logs

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["loss/val"] for x in outputs]).mean()
        same_class_cosine = torch.stack(
            [x["cosine/same_class"] for x in outputs]
        ).mean()
        diff_class_cosine = torch.stack(
            [x["cosine/diff_class"] for x in outputs]
        ).mean()
        accuracy = torch.stack([x["accuracy"] for x in outputs]).mean()

        logs = {
            "loss/val": loss,
            "cosine/same_class": same_class_cosine,
            "cosine/diff_class": diff_class_cosine,
            "accuracy": accuracy,
        }
        error = 1 - accuracy
        return {"val_loss": error, "log": logs}

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        same_class_cosine = (
            torch.stack([x["cosine/same_class"] for x in outputs]).mean().item()
        )
        diff_class_cosine = (
            torch.stack([x["cosine/diff_class"] for x in outputs]).mean().item()
        )
        accuracy = torch.stack([x["accuracy"] for x in outputs]).mean().item()

        logs = {
            "cosine/same_class": round(same_class_cosine, 4),
            "cosine/diff_class": round(diff_class_cosine, 4),
            "accuracy": round(accuracy, 4),
        }
        return {"progress_bar": logs}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        total_step = len(self.train_dataloader()) * self.hparams.max_epochs
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_step, eta_min=1e-8,
            ),
            "interval": "step",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        dataset = FaceIDDataset(
            self.hparams.data_dir, "train", size=self.hparams.train_size
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            shuffle=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        dataset = FaceIDDataset(
            self.hparams.data_dir, "val", size=self.hparams.val_size
        )
        dataloader = DataLoader(
            dataset, batch_size=self.hparams.batch_size, num_workers=4, pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        dataset = FaceIDDataset(
            self.hparams.data_dir, "val", size=self.hparams.test_size
        )
        dataloader = DataLoader(
            dataset, batch_size=self.hparams.batch_size, num_workers=4, pin_memory=True,
        )
        return dataloader
