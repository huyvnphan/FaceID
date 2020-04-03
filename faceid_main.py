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
    trainer = Trainer(gpus=[hparams.gpus], early_stop_callback=False, max_epoch=hparams.epoch)
    trainer.fit(model)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--description', type=str, default='Default')
    parser.add_argument('--data_dir', type=str, default='/raid/data/pytorch_dataset/faceid/')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=200)
    args = parser.parse_args()

    main(args)