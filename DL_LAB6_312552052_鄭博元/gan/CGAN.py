import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataloader import dataloader
import matplotlib.pyplot as plt
import json
import torchvision

with open("../objects.json", "r") as f:
    onehot = json.load(f)

class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=3, output_dim=3, input_size=64, class_num=24):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim + self.class_num, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        utils.initialize_weights(self)

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.fc(x)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x

class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_size=64, class_num=24):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim + self.class_num, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        #print(self.input_size)
        x = self.conv(x)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)

        return x

def save_G_test_images(G_test, save_dir, epoch):
    # 创建保存图像的文件夹（如果不存在）
    os.makedirs(f'./test_result/epoch_{epoch}', exist_ok=True)

    # 将G_test从Tensor转换为NumPy数组并调整形状
    #G_test = G_test.data.numpy().transpose(0, 2, 3, 1)
    G_test = (G_test + 1) / 2  # 将图像从[-1, 1]范围调整到[0, 1]范围

    # 保存图像
    #for i in range(len(G_test)):
    #    plt.imshow(G_test[i, :, :, :])
    #    plt.axis('off')
    #    plt.savefig(os.path.join(save_dir, f"G_test_image_{i}.png"), dpi=100)
    #    plt.close()  # 关闭图像，释放内存

    for j in range(G_test.shape[0]):
        torchvision.utils.save_image(G_test[j], f'./test_result/epoch_{epoch}/{j}.png')


class CGAN(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.input_size = args.input_size
        self.z_dim = 50
        self.class_num = 24
        self.sample_num = self.class_num ** 2

        self.milestones = args.milestones

        self.Gen_renew_cycle = 2

        # load dataset
        self.data_loader = dataloader(self.batch_size)

        # networks init
        self.G = generator(input_dim=self.z_dim, output_dim=3, input_size=self.input_size, class_num=self.class_num)
        self.G.load_state_dict(torch.load('./CGAN_91_G.pth'))
        self.D = discriminator(input_dim=3, output_dim=1, input_size=self.input_size, class_num=self.class_num)
        self.D.load_state_dict(torch.load('./CGAN_91_D.pth'))
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        self.schedulerG = optim.lr_scheduler.MultiStepLR(self.G_optimizer, milestones=self.milestones, gamma=0.4)
        self.schedulerD = optim.lr_scheduler.MultiStepLR(self.D_optimizer, milestones=self.milestones, gamma=0.4)

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        print('-----------------------------------------------')

        # fixed noise & condition
        self.sample_z_ = torch.zeros((self.sample_num, self.z_dim))
        for i in range(self.class_num):
            self.sample_z_[i*self.class_num] = torch.rand(1, self.z_dim)
            for j in range(1, self.class_num):
                self.sample_z_[i*self.class_num + j] = self.sample_z_[i*self.class_num]

        temp = torch.zeros((self.class_num, 1))
        for i in range(self.class_num):
            temp[i, 0] = i

        temp_y = torch.zeros((self.sample_num, 1))
        for i in range(self.class_num):
            temp_y[i*self.class_num: (i+1)*self.class_num] = temp

        self.sample_y_ = torch.zeros((self.sample_num, self.class_num)).scatter_(1, temp_y.type(torch.LongTensor), 1)
        if self.gpu_mode:
            self.sample_z_, self.sample_y_ = self.sample_z_.cuda(), self.sample_y_.cuda()

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        self.D.train()
        print('training start!!')
        start_time = time.time()

        os.makedirs(f'./test_result/model', exist_ok=True)

        save_dir = './test_result/model/'

        test = [["gray cube"], ["red cube"], ["blue cube"], ["blue cube", "green cube"], ["brown cube", "purple cube"], ["purple cube", "cyan cube"], ["yellow cube", "gray sphere"], ["blue sphere", "green sphere"], ["green sphere", "gray cube"], ["brown sphere", "red cube", "red cylinder"], ["purple sphere", "brown cylinder", "blue cube"], ["cyan sphere", "purple cylinder", "green cube"], ["yellow sphere", "cyan cylinder", "brown cube"], ["gray cylinder", "yellow cylinder", "purple cube"], ["blue cylinder", "gray cube", "cyan cube"], ["blue cylinder", "red cube", "yellow cube"], ["green cylinder"], ["brown cylinder"], ["purple cylinder"], ["cyan cylinder", "purple cylinder"], ["blue cylinder", "green cylinder"], ["gray cylinder", "green cube"], ["cyan sphere", "gray cylinder"], ["brown sphere", "green sphere"], ["blue sphere", "yellow cylinder"], ["red sphere", "cyan cylinder", "cyan cube"], ["gray sphere", "purple cylinder", "blue cube"], ["yellow cube", "brown cylinder", "purple cube"], ["cyan cube", "green cylinder", "blue cube"], ["brown cube", "blue cylinder", "blue sphere"], ["green sphere", "red cylinder", "brown sphere"], ["blue cylinder", "gray cylinder", "cyan sphere"]]
        test_oh = np.zeros((len(test),len(onehot)))
        for idx, l in enumerate(test):
            for i in l:
                test_oh[idx][onehot[i]] = 1

        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()

            if epoch == 20:
                self.Gen_renew_cycle = 8  # Update the value at epoch 50

            for iter, (x_, y_) in enumerate(self.data_loader):
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break
                # print(y_.size())
                z_ = torch.rand((self.batch_size, self.z_dim))
                # y_vec_ = torch.zeros((self.batch_size, self.class_num)).scatter_(1, y_.type(torch.LongTensor), 1)
                y_fill_ = y_.unsqueeze(2).unsqueeze(3).expand(self.batch_size, self.class_num, 64, 64)
                if self.gpu_mode:
                    x_, z_, y_, y_fill_ = x_.cuda(), z_.cuda(), y_.cuda(), y_fill_.cuda()

                # print(z_.size())
                # print(y_fill_[0])
                # print(y_)

                # update D network
                self.D_optimizer.zero_grad()
                #print(y_fill_.size())
                #print(x_.size())
                D_real = self.D(x_, y_fill_)
                D_real_loss = self.BCE_loss(D_real, self.y_real_)

                G_ = self.G(z_, y_)
                #print(G_.size())
                D_fake = self.D(G_, y_fill_)
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)

                D_loss = D_real_loss + D_fake_loss
                self.train_hist['D_loss'].append(D_loss.item())

                D_loss.backward()
                self.D_optimizer.step()

                # update G network
                for rep in range(1, self.Gen_renew_cycle): 
                    self.G_optimizer.zero_grad()
                    z_ = torch.rand((self.batch_size, self.z_dim))
                    G_ = self.G(z_.cuda(), y_)
                    D_fake = self.D(G_, y_fill_)
                    G_loss = self.BCE_loss(D_fake, self.y_real_)
                    self.train_hist['G_loss'].append(G_loss.item())

                    G_loss.backward()
                    self.G_optimizer.step()

                if ((iter + 1) % 10) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_loss.item()))
            with torch.no_grad():
                z_test = torch.rand((len(test), self.z_dim))
                G_test = self.G(z_test.cuda().to(torch.float32), torch.tensor(test_oh).cuda().to(torch.float32))
            # print(test_oh)
            #print(G_test.size())
            save_G_test_images(G_test, "./test_result/", epoch)

            torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + f'_{epoch}_G.pth'))
            torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + f'_{epoch}_D.pth'))

            self.schedulerG.step()
            self.schedulerD.step()

            print(f"Epoch [{epoch+1}/{self.epoch}], Generator LR: {self.schedulerG.get_last_lr()}, Discriminator LR: {self.schedulerD.get_last_lr()}")

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            with torch.no_grad():
                self.visualize_results((epoch+1))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        # weight_path = './saved_models/CGAN' + '_Weight_History/'
        # torch.save(model.state_dict(), weight_path)


        self.save()
        utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

    def visualize_results(self, epoch, fix=True):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

        image_frame_dim = int(np.floor(np.sqrt(self.sample_num)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_, self.sample_y_)
        else:
            """ random noise """
            sample_y_ = torch.zeros(self.batch_size, self.class_num).scatter_(1, torch.randint(0, self.class_num - 1, (self.batch_size, 1)).type(torch.LongTensor), 1)
            sample_z_ = torch.rand((self.batch_size, self.z_dim))
            if self.gpu_mode:
                sample_z_, sample_y_ = sample_z_.cuda(), sample_y_.cuda()

            samples = self.G(sample_z_, sample_y_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        save_path = os.path.join(self.result_dir, self.dataset, self.model_name)
        os.makedirs(save_path, exist_ok=True)
        

        # 循环遍历样本并保存图像
        for i in range(image_frame_dim * image_frame_dim):
            plt.imshow(samples[i, :, :, :])
            plt.axis('off')
            plt.savefig(os.path.join(save_path, f"{self.model_name}_epoch{epoch:03d}_{i}.png"), dpi=100)
            plt.close()  # 关闭图像，释放内存


    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pth'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pth'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))