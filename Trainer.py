import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10

# Function to set seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Function to generate PSNR values
def Generate_PSNR(image1, image2, data_range=1.):
    """PSNR for torch tensor"""
    mse_loss = nn.functional.mse_loss(image1, image2)
    psnr_value = 20 * log10(data_range) - 10 * torch.log10(mse_loss)
    return psnr_value

# KL divergence loss criterion
def kl_criterion(mu, logvar, batch_size):
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld_loss /= batch_size  
    return kld_loss

# KL annealing class
class KLAnnealing():
    def __init__(self, args, current_epoch=0):
        self.anneal_type = args.kl_anneal_type
        self.anneal_cycle = args.kl_anneal_cycle
        self.anneal_ratio = args.kl_anneal_ratio
        self.current_epoch = current_epoch

        if self.anneal_type == 'Cyclical':
            self.beta_schedule = self.frange_cycle_linear(args.num_epoch, start=0.0, stop=1.0, n_cycle=self.anneal_cycle, ratio=self.anneal_ratio)
        elif self.anneal_type == 'Monotonic':
            self.beta_schedule = np.linspace(0, 1, args.num_epoch)
        else:
            raise ValueError("Unknown KL annealing type")

        self.beta = self.beta_schedule[self.current_epoch]
    
    def update(self):
        self.current_epoch += 1
        if self.current_epoch < len(self.beta_schedule):
            self.beta = self.beta_schedule[self.current_epoch]
    
    def get_beta(self):
        return self.beta

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0, n_cycle=1, ratio=1.0):
        beta_values = np.ones(n_iter) * stop
        period = n_iter / n_cycle
        step = (stop - start) / (period * ratio)

        for c in range(n_cycle):
            v, i = start, 0
            while v <= stop and (int(i + c * period) < n_iter):
                beta_values[int(i + c * period)] = v
                v += step
                i += 1
        return beta_values


# VAE model class
class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Initialize mode (dummy variable)
        self.mode = 1  
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optimizer      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[2, 5], gamma=0.1)
        self.kl_annealing = KLAnnealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.teacher_forcing_ratio = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        
    def forward(self, img, label):
        pass

    
    def training_stage(self):
        train_loss_plot = []
        val_loss_plot = []
        tfr_plot = []
        psnr_plot = []

        for i in range(self.args.num_epoch):
            img_count = 0
            tfr_plot.append(self.teacher_forcing_ratio)
            train_loss = 0.0
            train_loader = self.train_dataloader()
            use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                img_count += (img.size(0) * img.size(1))
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss = self.training_one_step(img, label, use_teacher_forcing)
                train_loss += loss.detach().cpu() * img.size(0)

                beta = self.kl_annealing.get_beta()
                if use_teacher_forcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {}'.format(self.teacher_forcing_ratio, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {}'.format(self.teacher_forcing_ratio, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])

            # Save model checkpoint
            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"specail={self.current_epoch}.ckpt"))

            # Evaluate after each epoch
            val_loss, psnr_value = self.eval()

            # Log the validation results
            print(f"Validation Loss: {val_loss:.6f}, PSNR: {psnr_value:.2f}")

            # Record losses and metrics for plotting
            train_loss_plot.append(train_loss / img_count)
            val_loss_plot.append(val_loss)
            psnr_plot.append(psnr_value)

            # Increment epoch and update scheduler and KL annealing
            self.current_epoch += 1
            self.scheduler.step()
            self.update_teacher_forcing_ratio()
            self.kl_annealing.update()

            # Optionally, generate and save plots
            self.generate_plots(train_loss_plot, val_loss_plot, tfr_plot, psnr_plot)


    def generate_plots(self, train_loss_plot, val_loss_plot, tfr_plot, psnr_plot, psnr_per_frame=False):
        if psnr_per_frame:
            plt.figure(figsize=(10, 6))
            plt.plot(train_loss_plot, label='PSNR')
            plt.xlabel('Frame')
            plt.ylabel('PSNR')
            plt.title(f'Validation PSNR ({self.args.kl_anneal_type})')
            plt.legend()
            plt.savefig(f'./{self.args.save_root}/{self.args.kl_anneal_type}_PSNR_per_frame.png')
            plt.close()
            return
    
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss_plot, marker='o', label='Training Loss')
        plt.plot(val_loss_plot, marker='o', label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Loss Curve({self.args.kl_anneal_type})')
        plt.legend()
        plt.savefig(f'./{self.args.save_root}/{self.args.kl_anneal_type}_loss.png')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.semilogy(train_loss_plot, marker='o', label='Training Loss')
        plt.semilogy(val_loss_plot, marker='o', label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Log Loss Curve({self.args.kl_anneal_type})')
        plt.legend()
        plt.savefig(f'./{self.args.save_root}/{self.args.kl_anneal_type}_logloss.png')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(tfr_plot, marker='o', label='Teacher Forcing Ratio')
        plt.xlabel('Epochs')
        plt.ylabel('Teacher Forcing Ratio')
        plt.title('Teacher Forcing Ratio over Epochs')
        plt.legend()
        plt.savefig(f'./{self.args.save_root}/{self.args.kl_anneal_type}_tfr.png')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(psnr_plot, marker='o', label='Validation PSNR')
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.title('Validation PSNR')
        plt.legend()
        plt.savefig(f'./{self.args.save_root}/{self.args.kl_anneal_type}_PSNR.png')
        plt.close()

            
            
    
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        val_loss = 0.0
        val_psnr = 0.0
        img_count = 0

        for (img, label) in tqdm(val_loader, ncols=120):
            img_count += (img.size(0) * img.size(1))
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss, psnr_value = self.val_one_step(img, label)
            val_loss += loss.detach().cpu()
            val_psnr += psnr_value.detach().cpu()

        return val_loss/img_count, val_psnr/img_count

            
    def training_one_step(self, img, label, use_teacher_forcing):
        img = img.permute(1, 0, 2, 3, 4)
        label = label.permute(1, 0, 2, 3, 4)
        out = img[0]

        reconstruction_loss = 0.0
        kl_loss = 0.0
        for i in range(1, self.train_vi_len):
            label_feat = self.label_transformation(label[i])
            if self.mode == 1:
                out = img[i-1] * self.teacher_forcing_ratio + out * (1 - self.teacher_forcing_ratio)
            elif use_teacher_forcing:
                out = img[i-1]
            frame_feat = self.frame_transformation(out)
            
            # Pass separate arguments to Gaussian_Predictor
            z, mu, logvar = self.Gaussian_Predictor(frame_feat, label_feat)
            
            parm = self.Decoder_Fusion(frame_feat, label_feat, z)
            out = self.Generator(parm)

            reconstruction_loss += self.mse_criterion(out, img[i])
            kl_loss += kl_criterion(mu, logvar, batch_size=self.batch_size)

        beta = torch.tensor(self.kl_annealing.get_beta()).to(self.args.device)
        loss = reconstruction_loss + beta * kl_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer_step()

        return loss

    
    def val_one_step(self, img, label):
        img = img.permute(1, 0, 2, 3, 4)
        label = label.permute(1, 0, 2, 3, 4)
        decoded_frame_list = [img[0].cpu()]
        out = img[0]
        psnr = 0.0
        psnr_per_frame = []
        reconstruction_loss = 0.0
        kl_loss = 0.0

        for i in range(1, self.val_vi_len):
            label_feat = self.label_transformation(label[i])
            frame_feat = self.frame_transformation(out)
            
            # Generate z using normal distribution
            z = torch.randn(1, self.args.N_dim, self.args.frame_H, self.args.frame_W, device=self.args.device)
            
            # Call Decoder_Fusion with correct arguments
            fusion_feat = self.Decoder_Fusion(frame_feat, label_feat, z)
            out = self.Generator(fusion_feat)
            
            decoded_frame_list.append(out.cpu())
            psnr_value = Generate_PSNR(out, img[i])
            psnr += psnr_value
            psnr_per_frame.append(psnr_value.detach().cpu())
            reconstruction_loss += self.mse_criterion(out, img[i])

        generated_frame = stack(decoded_frame_list).permute(1, 0, 2, 3, 4)
        self.make_gif(generated_frame[0], os.path.join(self.args.save_root, f'{self.args.kl_anneal_type}.gif'))
        loss = reconstruction_loss
        self.generate_plots(psnr_per_frame, [], [], [], psnr_per_frame=True)

        return loss, psnr

                
    def make_gif(self, images_list, img_name):
        image_sequence = []
        for img in images_list:
            image_sequence.append(transforms.ToPILImage()(img))
            
        image_sequence[0].save(img_name, format="GIF", append_images=image_sequence,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform_pipeline = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform_pipeline, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform_pipeline = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform_pipeline, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def update_teacher_forcing_ratio(self):
        if self.current_epoch >= self.tfr_sde and self.teacher_forcing_ratio > 0:
            self.teacher_forcing_ratio = max(0.0, self.teacher_forcing_ratio - self.tfr_d_step)
            
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.teacher_forcing_ratio,
            "last_epoch": self.current_epoch
        }, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path is not None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.teacher_forcing_ratio = checkpoint['tfr']
            
            self.optimizer      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[2, 4], gamma=0.1)
            self.kl_annealing = KLAnnealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optimizer.step()


def main(args):
    # Set random seed for reproducibility
    set_seed(42)
    
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="Initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="Enable result visualization while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="Path to save results")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=60,     help="Number of total epochs")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every set number of epochs")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Fraction of the training dataset to use")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="Validation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height of the input image to be resized")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width of the input image to be resized")
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="Initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="Epoch at which teacher forcing ratio starts to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step for teacher forcing ratio")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="Path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Fraction of data to use for fast training")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epochs to use fast train mode")
    
    # KL annealing strategy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Monotonic',       help="Type of KL annealing")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="Number of cycles for KL annealing")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=1,              help="Ratio for KL annealing")
    
    args = parser.parse_args()
    
    main(args)
