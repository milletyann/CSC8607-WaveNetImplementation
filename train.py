import argparse
import torch
from classes.WaveNet import WaveNet
from classes.Dataset import VCTKDataset, collate_fn
from classes.Trainer import Trainer
from torch.utils.data import DataLoader

from classes.Utils import quantize

def parse_args():
    parser = argparse.ArgumentParser(description="Launch WaveNet Training Experiment")
    
    # general params
    parser.add_argument('--num_layers', type=int, default=10, help="Number of layers in the WaveNet model")
    parser.add_argument('--num_blocks', type=int, default=4, help="Number of blocks in the WaveNet model")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs for training")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    
    # dataset params
    parser.add_argument('--root_dir', type=str, default='../VCTK-Corpus-0.92/VCTK-Corpus-0.92/', help="Directory containing the audio files")
    
    # log params
    parser.add_argument('--log_dir', type=str, default='logs', help="Directory to save logs")
    
    # test params
    parser.add_argument('--max_batch', type=int, default=None, help="Max batch per epoch used for training")
    parser.add_argument('--lr_range_test', action="store_true", help="Try several learning rates [boolean]")
    parser.add_argument('--validate_every', type=int, default=None, help="Validate the performance every X batches")
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    torch.cuda.empty_cache()

    dataset = VCTKDataset(root_dir=args.root_dir)
    
    model = WaveNet(num_layers=args.num_layers, num_blocks=args.num_blocks, kernel_size=2, 
                    dilation_channels=32, residual_channels=32, skip_channels=256, 
                    conditioning_channels=dataset.num_speakers)

    trainer = Trainer(model, dataset, batch_size=args.batch_size, lr=args.lr, num_epochs=args.epochs,  max_batch=args.max_batch , lr_range_test=args.lr_range_test, validate_every=args.validate_every)
    trainer.train()
    
    trainer.test()

if __name__ == "__main__":
    main()
