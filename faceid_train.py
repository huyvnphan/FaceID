from argparse import ArgumentParser
from pytorch_lightning import Trainer
from faceid_module import FaceIDModule
import random
import torch
import os

def main(hparams):
    
    # Reproducibility
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    
    # Set GPU
    torch.cuda.set_device(hparams.gpu)
    
    model = FaceIDModule(hparams)
    trainer = Trainer(gpus=[hparams.gpu],
                      max_epochs=hparams.epochs,
                      early_stop_callback=False,
                      check_val_every_n_epoch=2)
    trainer.fit(model)
    
    # Save weights from checkpoint
    checkpoint_path = os.path.join(os.getcwd(), 'lightning_logs', 'version_0', 'checkpoints')
    model = FaceIDModule.load_from_checkpoint(os.path.join(checkpoint_path, os.listdir(checkpoint_path)[0]))
    statedict_path = os.path.join(os.getcwd(), 'faceid_weights.pt')
    torch.save(model.model.state_dict(), statedict_path)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--description', type=str, default='Default')
    parser.add_argument('--data_dir', type=str, default='/raid/data/pytorch_dataset/faceid/')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'SGD'])
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--reduce_lr_per', type=int, default=50)
    args = parser.parse_args()
    main(args)