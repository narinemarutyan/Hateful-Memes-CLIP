import argparse

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from .caller import Caller
from .datasets import HatefulMemesDataset
from .train import TrainCLIP


def load(args, split):
    dataset = HatefulMemesDataset(image_folder='data/hateful_memes/img',
                                  csv_folder='data/hateful_memes/hateful_memes_expanded.csv',
                                  split=split,
                                  image_size=args.image_size)

    return dataset


def main(args):
    # Load dataset
    dataset_train = load(args=args, split='train')
    dataset_val = load(args=args, split='dev')
    dataset_test = load(args=args, split='test')

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

    num_params = {f'param_{n}': p.numel() for n, p in model.named_parameters() if p.requires_grad}
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints', filename='-{epoch:02d}', monitor="val/auroc",
                                          mode='max', verbose=True, save_weights_only=True, save_top_k=1,
                                          save_last=False)

    # Initialize Trainer
    trainer = Trainer(max_epochs=args.max_epochs, max_steps=args.max_steps, gradient_clip_val=0.1,
                      log_every_n_steps=50, val_check_interval=1.0,
                      callbacks=[checkpoint_callback],
                      limit_train_batches=1.0, limit_val_batches=1.0,
                      deterministic=True)

    # Train and test the model
    model.compute_fine_grained_metrics = True
    trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)
    trainer.test(ckpt_path='best', dataloaders=[dataloader_val, dataloader_test])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("-i", "--image_size", type=int, help="Image size", default=224)
    parser.add_argument("-c", "--clip_pretrained_model", type=str, help="Pretrained model for CLIP",
                        default="openai/clip-vit-base-patch32")
    parser.add_argument("-m", "--map_dim", type=int, help="Map dimension", default=768)
    parser.add_argument("-n", "--num_pre_output_layers", type=int, help="Number of pre-output layers", default=1)
    parser.add_argument("-d", "--drop_probs", nargs='+', type=float, help="Dropout probabilities",
                        default=[0.1, 0.4, 0.2])
    parser.add_argument("-g", "--gpus", nargs='+', type=int, help="GPUs to use", default=[0])
    parser.add_argument("--max_steps", type=int, help="Maximum number of steps", default=-1)
    parser.add_argument("--max_epochs", type=int, help="Maximum number of epochs", default=-1)
    parser.add_argument("-b", "--batch_size", type=int, help="Batch size", default=9)
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-4)
    parser.add_argument("--weight_image_loss", type=float, help="Weight for image loss", default=1.0)
    parser.add_argument("--weight_text_loss", type=float, help="Weight for text loss", default=1.0)
    parser.add_argument("--weight_decay", type=float, help="Weight decay", default=1e-4)

    args = parser.parse_args()
    main(args)
