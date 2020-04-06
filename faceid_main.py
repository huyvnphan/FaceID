from argparse import ArgumentParser
from pytorch_lightning import Trainer
from faceid_system import FaceIDSystem
import random
import torch

def main(hparams):
    
    # Reproducibility
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    
    model = FaceIDSystem(hparams)
    trainer = Trainer(gpus=[hparams.gpus],
                      max_epochs=hparams.epochs,
                      early_stop_callback=False,
                      check_val_every_n_epoch=5)
    trainer.fit(model)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--description', type=str, default='Default')
    parser.add_argument('--data_dir', type=str, default='/raid/data/pytorch_dataset/faceid/')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=300)
    args = parser.parse_args()

    main(args)