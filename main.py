from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from datasets import CustomCollator, load_dataset
from engine import create_model


class Alissa:
    def __init__(self, args_dict):
        for key, value in args_dict.items():
            setattr(self, key, value)


def main(args):
    # Load dataset
    dataset_train = load_dataset(args=args, split='train')
    dataset_val = load_dataset(args=args, split='dev')
    dataset_test = load_dataset(args=args, split='test')
    # dataset_val_unseen = load_dataset(args=args, split='dev_unseen')
    # dataset_test_unseen = load_dataset(args=args, split='test_unseen')

    # Load dataloader
    num_cpus = 1
    collator = CustomCollator(args, dataset_train.fine_grained_labels)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=num_cpus,
                                  collate_fn=collator)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, num_workers=num_cpus, collate_fn=collator)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=num_cpus, collate_fn=collator)
    # dataloader_val_unseen = DataLoader(dataset_val_unseen, batch_size=args.batch_size, num_workers=num_cpus, collate_fn=collator)
    # dataloader_test_unseen = DataLoader(dataset_test_unseen, batch_size=args.batch_size, num_workers=num_cpus, collate_fn=collator)

    # Create model
    seed_everything(28, workers=True)
    model = create_model(args, dataset_train.fine_grained_labels)

    num_params = {f'param_{n}': p.numel() for n, p in model.named_parameters() if p.requires_grad}
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints', filename='-{epoch:02d}',
                                          monitor="val/auroc", mode='max', verbose=True, save_weights_only=True,
                                          save_top_k=1, save_last=False)

    # Initialize Trainer
    trainer = Trainer(max_epochs=args.max_epochs, max_steps=args.max_steps, gradient_clip_val=args.gradient_clip_val,
                      log_every_n_steps=args.log_every_n_steps,
                      val_check_interval=args.val_check_interval,
                      callbacks=[checkpoint_callback],
                      limit_train_batches=args.limit_train_batches, limit_val_batches=args.limit_val_batches,
                      deterministic=True)

    # Train and test the model
    model.compute_fine_grained_metrics = True
    trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)
    trainer.test(ckpt_path='best', dataloaders=[dataloader_val, dataloader_test])


if __name__ == '__main__':
    args_dataset = {
        "dataset": "original",
        "labels": "original",
        "image_size": 224,
        "multilingual_tokenizer_path": "none",
        "clip_pretrained_model": "openai/clip-vit-base-patch32",
        "local_pretrained_weights": "none",
        "caption_mode": "none",
        "use_pretrained_map": False,
        "num_mapping_layers": 1,
        "map_dim": 768,
        "fusion": "align",  # default value of this is clip but no case for clip  
        "num_pre_output_layers": 1,
        "drop_probs": [0.1, 0.4, 0.2],
        "image_encoder": "clip",
        "text_encoder": "clip",
        "freeze_image_encoder": True,
        "freeze_text_encoder": True,
        "remove_matches": False,
        "gpus": [0],
        "limit_train_batches": 1.0,
        "limit_val_batches": 1.0,
        "max_steps": -1,
        "max_epochs": -1,
        "log_every_n_steps": 50,
        "val_check_interval": 1.0,
        "batch_size": 9,
        "lr": 1e-4,
        "weight_image_loss": 1.0,
        "weight_text_loss": 1.0,
        "weight_fine_grained_loss": 1.0,
        "weight_super_loss": 1.0,
        "weight_decay": 1e-4,
        "gradient_clip_val": 0.1
    }
    args = Alissa(args_dataset)
    print('start')
    main(args)
