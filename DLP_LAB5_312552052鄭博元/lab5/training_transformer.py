import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn.functional as F
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData, mask_image, plot_loss, plot_accuracy
import yaml
from torch.utils.data import DataLoader


class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.args = args
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(
            device=args.device
        )
        self.optim, self.scheduler = self.configure_optimizers()
        self.prepare_training()
        if args.start_from_epoch:
            self.load_checkpoint(args.start_from_epoch)

    @staticmethod
    def prepare_training():
        os.makedirs(args.checkpoint_path, exist_ok=True)

    def train_one_epoch(self, epoch, train_loader):
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        for data in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            inputs = data.to(self.args.device)

            _, z_indices = self.model.encode_to_z(inputs)

            mask_ratio = np.random.uniform(0, 1)
            mask_rate = self.model.gamma(mask_ratio)
            masked_z_indices = mask_image(
                z_indices, self.model.mask_token_id, mask_rate
            )
            mask = masked_z_indices == self.model.mask_token_id

            logits = self.model.transformer(masked_z_indices)

            if not mask.any():
                continue

            loss = F.cross_entropy(logits[mask], z_indices[mask])
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            running_loss += loss.item()

            _, predicted = torch.max(logits[mask], 1)
            correct_predictions += (predicted == z_indices[mask]).sum().item()
            total_predictions += mask.sum().item()

        accuracy = (
            correct_predictions / total_predictions if total_predictions > 0 else 0.0
        )
        return running_loss / len(train_loader), accuracy

    def eval_one_epoch(self, epoch, val_loader):
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        with torch.no_grad():
            for data in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
                inputs = data.to(self.args.device)

                _, z_indices = self.model.encode_to_z(inputs)

                mask_ratio = np.random.uniform(0, 1)
                mask_rate = self.model.gamma(mask_ratio)
                masked_z_indices = mask_image(
                    z_indices, self.model.mask_token_id, mask_rate
                )
                mask = masked_z_indices == self.model.mask_token_id

                logits = self.model.transformer(masked_z_indices)

                if not mask.any():
                    continue

                loss = F.cross_entropy(logits[mask], z_indices[mask])
                running_loss += loss.item()

                _, predicted = torch.max(logits[mask], 1)
                correct_predictions += (predicted == z_indices[mask]).sum().item()
                total_predictions += mask.sum().item()

        accuracy = (
            correct_predictions / total_predictions if total_predictions > 0 else 0.0
        )
        return running_loss / len(train_loader), accuracy

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return optimizer, scheduler

    def save_checkpoint(self, epoch):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.transformer.state_dict(),
                "optimizer_state_dict": self.optim.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            f"transformer_checkpoints/ckpt_epoch_{epoch}.pt",
        )

    def load_checkpoint(self, epoch):
        chpt_path = f"transformer_checkpoints/ckpt_epoch_{epoch}.pt"
        ckpt = torch.load(chpt_path, weights_only=True)
        self.model.transformer.load_state_dict(ckpt["model_state_dict"])
        self.optim.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MaskGIT")
    parser.add_argument(
        "--train_d_path",
        type=str,
        default="dataset/train/",
        help="Training Dataset Path",
    )
    parser.add_argument(
        "--val_d_path",
        type=str,
        default="dataset/val/",
        help="Validation Dataset Path",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="transformer_checkpoints",
        help="Path to checkpoint.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Which device the training is on."
    )
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training."
    )
    parser.add_argument(
        "--partial",
        type=float,
        default=1.0,
        help="Percentage of the dataset to use for training.",
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of epochs to train."
    )
    parser.add_argument(
        "--val-per-epoch",
        type=int,
        default=5,
        help="Validation per ** epochs(default: 5)",
    )
    parser.add_argument(
        "--save-per-epoch",
        type=int,
        default=1,
        help="Save CKPT per ** epochs(default: 1)",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.01, help="Learning rate."
    )
    parser.add_argument(
        "--MaskGitConfig",
        type=str,
        default="config/MaskGit.yml",
        help="Configurations for TransformerVQGAN",
    )

    args = parser.parse_args()
    args.start_from_epoch = 0
    os.makedirs(args.checkpoint_path, exist_ok=True)
    ckpt_files = os.listdir(args.checkpoint_path)
    if ckpt_files:
        ckpt_files.sort()
        last_ckpt = ckpt_files[-1]
        args.start_from_epoch = int(last_ckpt.split("_")[-1].split(".")[0])

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, "r"))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root=args.train_d_path, partial=args.partial)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        shuffle=True,
    )

    val_dataset = LoadTrainData(root=args.val_d_path, partial=args.partial)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        shuffle=False,
    )

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    for epoch in range(args.start_from_epoch + 1, args.epochs + 1):
        train_loss, train_acc = train_transformer.train_one_epoch(epoch, train_loader)
        print(
            f"Epoch {epoch}: Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}"
        )
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        if epoch % args.val_per_epoch == 0:
            val_loss, val_acc = train_transformer.eval_one_epoch(epoch, val_loader)
            print(
                f"Epoch {epoch}: Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}"
            )
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)

        train_transformer.scheduler.step()

        if epoch % args.save_per_epoch == 0:
            train_transformer.save_checkpoint(epoch)

    plot_loss(train_loss_list, "Training")
    plot_accuracy(train_acc_list, "Training")
    plot_loss(val_loss_list, "Validation")
    plot_accuracy(val_acc_list, "Validation")