from faceid_dataset import FaceIDDataset
from faceid_model import FaceID_Model
from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl

class FaceIDModule(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.model = FaceID_Model()
        self.loss = torch.nn.CosineEmbeddingLoss(margin=0.5)
        self.cosine = torch.nn.CosineSimilarity()
        self.threshold = 0.85
        
    def calculate_stat(self, embed_x0, embed_x1, y):
        cosine = self.cosine(embed_x0, embed_x1)
        same_class_cosine = cosine[y==1]
        diff_class_cosine = cosine[y==-1]
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
        logs = {'loss/train': loss}
        return {'loss': loss, 'log': logs}
        
    def validation_step(self, batch, batch_nb):
        x0, x1, y = batch
        embed_x0, embed_x1 = self.forward(x0, x1)
        loss = self.loss(embed_x0, embed_x1, y)
        same_class_cosine, diff_class_cosine, accuracy = self.calculate_stat(embed_x0, embed_x1, y)
        logs = {'loss/val': loss,
                'cosine/same_class': same_class_cosine,
                'cosine/diff_class': diff_class_cosine,
                'accuracy': accuracy}
        return logs
                
    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['loss/val'] for x in outputs]).mean()
        same_class_cosine = torch.stack([x['cosine/same_class'] for x in outputs]).mean()
        diff_class_cosine = torch.stack([x['cosine/diff_class'] for x in outputs]).mean()
        accuracy = torch.stack([x['accuracy'] for x in outputs]).mean()
        
        logs = {'loss/val': loss,
                'cosine/same_class': same_class_cosine,
                'cosine/diff_class': diff_class_cosine,
                'accuracy': accuracy}
        return {'val_loss': loss, 'log': logs}    
    
    def configure_optimizers(self):
        if self.hparams.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay, momentum=0.9, nesterov=True)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.reduce_lr_per, gamma=0.1)
        return [optimizer], [scheduler]
    
    def train_dataloader(self):
        dataset = FaceIDDataset(self.hparams.data_dir)
        dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size,
                                num_workers=4, shuffle=True, drop_last=True, pin_memory=True)
        return dataloader
    
    def val_dataloader(self):
        dataset = FaceIDDataset(self.hparams.data_dir, train=False)
        dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size,
                                num_workers=4, shuffle=True, pin_memory=True)
        return dataloader