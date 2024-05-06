#!/usr/bin/env python

import argparse

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from src.caller import Caller
from src.datasets import HatefulMemesDataset
from src.train import TrainCLIP


def main(args):
    # Load dataset
    dataset_train = HatefulMemesDataset(image_path=args.image_path,
                                        csv_path=args.csv_path,
                                        split='train',
                                        image_size=args.image_size)
    dataset_val = HatefulMemesDataset(image_path=args.image_path,
                                      csv_path=args.csv_path,
                                      split='val',
                                      image_size=args.image_size)
    dataset_test = HatefulMemesDataset(image_path=args.image_path,
                                       csv_path=args.csv_path,
                                       split='test',
                                       image_size=args.image_size)

    # Load dataloader
    num_cpus = 1
    collator = Caller(args)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=num_cpus,
                                  collate_fn=collator)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, num_workers=num_cpus, collate_fn=collator)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=num_cpus, collate_fn=collator)

    # Create model
    seed_everything(28, workers=True)
    model = TrainCLIP(args=args)

    # Initialize Wandb logger
    wandb_logger = WandbLogger(project="meme-v2", config=args)
    num_params = {f'param_{n}': p.numel() for n, p in model.named_parameters() if p.requires_grad}
    wandb_logger.experiment.config.update(num_params)
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints', filename=wandb_logger.experiment.name + '-{epoch:02d}',
                                          monitor="val/auroc", mode='max', verbose=True, save_weights_only=True,
                                          save_top_k=1, save_last=False)

    # Initialize Trainer
    trainer = Trainer(max_epochs=args.max_epochs, max_steps=args.max_steps, gradient_clip_val=0.1,
                      logger=wandb_logger, log_every_n_steps=50, val_check_interval=1.0,
                      strategy='ddp_find_unused_parameters_true', callbacks=[checkpoint_callback],
                      limit_train_batches=1.0, limit_val_batches=1.0,
                      deterministic=True)

    # Train and test the model
    model.compute_fine_grained_metrics = True
    trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)
    trainer.test(ckpt_path='best', dataloaders=[dataloader_val, dataloader_test])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parse Training Arguments")
    parser.add_argument("--image_path", type=str, help="Path to images directory")
    parser.add_argument("--csv_path", type=str, help="Path to CSV file")
    parser.add_argument("--image_size", type=int, help="Image size", default=224)
    parser.add_argument("--clip_pretrained_model", type=str, help="Pretrained model for CLIP",
                        default="openai/clip-vit-base-patch32")
    parser.add_argument("--map_dim", type=int, help="Map dimension", default=768)
    parser.add_argument("--num_pre_output_layers", type=int, help="Number of pre-output layers", default=1)
    parser.add_argument("--drop_probs", nargs='+', type=float, help="Dropout probabilities",
                        default=[0.1, 0.4, 0.2])
    parser.add_argument("--gpus", nargs='+', type=int, help="GPUs to use", default=[0])
    parser.add_argument("--max_steps", type=int, help="Maximum number of steps", default=-1)
    parser.add_argument("--max_epochs", type=int, help="Maximum number of epochs", default=-1)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=9)
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-4)
    parser.add_argument("--weight_image_loss", type=float, help="Weight for image loss", default=1.0)
    parser.add_argument("--weight_text_loss", type=float, help="Weight for text loss", default=1.0)
    parser.add_argument("--weight_decay", type=float, help="Weight decay", default=1e-4)

    args = parser.parse_args()
    main(args)
