from faceid_dataset import get_dataloader
from squeezenet import SqueezeNet
import torch
import pytorch_lightning as pl

class FaceIDSystem(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.model = SqueezeNet()
        self.loss = torch.nn.CosineEmbeddingLoss(margin=0.5)
        self.cosine = torch.nn.CosineSimilarity()
        
    def calculate_cosine(self, embed_x0, embed_x1, y):
        cosine = self.cosine(embed_x0, embed_x1)
        same_class_cosine = cosine[y==1].mean()
        diff_class_cosine = cosine[y==-1].mean()
        return same_class_cosine, diff_class_cosine
                    
    def forward(self, x0, x1):
        embed_x0 = self.model(x0)
        embed_x1 = self.model(x1)
        return embed_x0, embed_x1
    
    def training_step(self, batch, batch_nb):
        x0, x1, y = batch
        embed_x0, embed_x1 = self.forward(x0, x1)
        loss = self.loss(embed_x0, embed_x1, y) 
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}
        
    def validation_step(self, batch, batch_nb):
        x0, x1, y = batch
        embed_x0, embed_x1 = self.forward(x0, x1)
        loss = self.loss(embed_x0, embed_x1, y)
        same_class_cosine, diff_class_cosine = self.calculate_cosine(embed_x0, embed_x1, y)
        logs = {'val_loss': loss,
                'same_class_cosine': same_class_cosine,
                'diff_class_cosine': diff_class_cosine}
        return logs
                
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_same_class_cosine = torch.stack([x['same_class_cosine'] for x in outputs]).mean()
        avg_diff_class_cosine = torch.stack([x['diff_class_cosine'] for x in outputs]).mean()
        
        logs = {'val_loss': avg_loss,
                'same_class_cosine': avg_same_class_cosine,
                'diff_class_cosine': avg_diff_class_cosine}
        return {'val_loss': avg_loss, 'log': logs}
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        return [optimizer], [scheduler]
    
    def train_dataloader(self):
        return get_dataloader(self.hparams.data_dir, self.hparams.batch_size)
    
    def val_dataloader(self):
        return get_dataloader(self.hparams.data_dir, self.hparams.batch_size, train=False)