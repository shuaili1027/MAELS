import numpy as np
import pyrootutils
import torch
from torch.autograd import Variable
from torchvision.utils import save_image
import sys, random
from backbone import Encoder, Decoder, D_BACKBONE, D_HEAD, CODE_HEAD
import torch.nn.functional as F
sys.path.append('..')
from DataM import DataM
import os
import datetime
import logging
# import matplotlib.pyplot as plt
from CNNs import backbones as b

# Get the root directory of the whole project
root = pyrootutils.setup_root(search_from=__file__, indicator=["pyproject.toml"], pythonpath=True, dotenv=True)


class trainloop():
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.EN = Encoder(self.params)
        self.DE = Decoder(self.params)
        self.D = D_BACKBONE(self.params)
        self.D_head = D_HEAD(self.params)
        self.C_head = CODE_HEAD(self.params)

        self.EN.to(self.params.device)
        self.DE.to(self.params.device)
        self.D.to(self.params.device)
        self.D_head.to(self.params.device)
        self.C_head.to(self.params.device)

        self.acc = 0

        if self.params.stage == "stage2":
            self.EN.load_state_dict(torch.load(self.params.stage1_path)["EN"])
            self.DE.load_state_dict(torch.load(self.params.stage1_path)["DE"])

        # if self.params.device is "cuda"
        self.F = torch.cuda.FloatTensor
        self.L = torch.cuda.LongTensor

        self.acc = 0
        self.patience = 0

        if self.params.img_size == 64:
            all_model = ["mobilenetv2", "resnet18", "resnext50", "vgg16"]
        elif self.params.img_size == 224:
            all_model = ["mobilenetv2", "resnet50", "resnext50", "vgg16"]

        # White boxes
        self.target = []
        for i in range(self.params.target_num):
            target = b.CustomNet(all_model[i], self.params.database)
            target.load_state_dict(torch.load(os.path.join(root,
                self.params.log_dir, self.params.file,
                "{}-{}.pth".format(
                self.params.database,
                all_model[i]))))
            target.to(self.params.device)
            target.eval()
            self.target.append(target)

        self.adv_loss = torch.nn.BCEWithLogitsLoss()
        self.cate_loss = torch.nn.CrossEntropyLoss()
        self.cont_loss = torch.nn.MSELoss()
        self.SF_loss = torch.nn.Softmax(dim=1)

        self.setup_logging()
        logging.info("Starting Experiments")

    def all_optimizers(self):
        self.c_lr = self.params.c_lr
        self.d_lr = self.params.d_lr
        self.info_lr = self.params.info_lr
        self.en_lr = self.params.en_lr
        self.g_lr = self.params.g_lr

        c_opt = torch.optim.Adam([{'params': self.C_head.parameters()}],
                                 self.c_lr,
                                 [self.params.beta1, self.params.beta2])
        d_opt = torch.optim.Adam([{'params': self.D.parameters()},
                                  {'params': self.D_head.parameters()}],
                                 self.d_lr,
                                 [self.params.beta1, self.params.beta2])
        info_opt = torch.optim.Adam([{'params': self.DE.parameters()},
                                     {'params': self.C_head.parameters()}],
                                    self.info_lr,
                                    [self.params.beta1, self.params.beta2])
        eg_opt = torch.optim.Adam([{'params': self.DE.parameters()},
                                   {'params': self.EN.parameters()}],
                                  self.en_lr,
                                  [self.params.beta1, self.params.beta2])
        en_opt = torch.optim.Adam([{'params': self.EN.parameters()}],
                                  self.en_lr,
                                  [self.params.beta1, self.params.beta2])
        
        g_opt = torch.optim.Adam([{'params': self.DE.parameters()}],
                                  self.g_lr,
                                  [self.params.beta1, self.params.beta2])

        return c_opt, d_opt, info_opt, en_opt, eg_opt, g_opt

    def all_loader(self):
        all = DataM(self.params)
        train_loader = all.return_all_loader()
        return train_loader

    def to_categorical(self, y, num_columns):
        """Returns one-hot encoded Variable"""
        y_cat = np.zeros((y.shape[0], num_columns))
        y_cat[range(y.shape[0]), y] = 1.0
        return Variable(self.F(y_cat), requires_grad=False)

    def stage1(self, loader, current_epoch, c_opt, d_opt, info_opt, en_opt, eg_opt):
        """ First, construct a semantic-oriented manifold at stage 1 """
        for idx, (data, label) in enumerate(loader):
            x_real, logit = data.to(self.params.device), label.to(self.params.device)
            # Creating self-supervised vectors:
            # Generating input sets of fake images
            z = Variable(self.F(np.random.normal(0, 1, (x_real.shape[0], self.params.noise_dim))), requires_grad=False)
            label_nmupy = label.cpu().numpy()
            label_input = self.to_categorical(label_nmupy, num_columns=self.params.label_dim)

            code_input_F = Variable(self.F(np.random.uniform(-1, 1, (x_real.shape[0], self.params.varying_dim))),
                                    requires_grad=False)

            code_F = torch.cat((label_input, code_input_F), dim=-1)

            # Generating ground truth for real and fake images:
            self.real = Variable(self.F(x_real.shape[0], 1).fill_(1.0), requires_grad=False)
            self.fake = Variable(self.F(x_real.shape[0], 1).fill_(0.0), requires_grad=False)

            # Training Q --- 》 classifying real images
            c_opt.zero_grad()
            real_out = self.D(x_real)
            output = self.C_head(real_out)
            c_loss = self.cate_loss(output[:, :self.params.label_dim], label_input)
            c_loss.backward()
            c_opt.step()

            # # real code
            real_out = self.D(x_real)
            output = self.C_head(real_out)
            code_R = output.detach()

            # Training EN --- 》 encoding latent vectors
            en_opt.zero_grad()
            z_recon, mean, var = self.EN(x_real)
            eg_recon_loss0 = (self.loss_function(mean, var) / x_real.size(0))

            en_loss = (eg_recon_loss0)
            en_loss.backward()
            en_opt.step()

            # Training D --- 》 discriminating real and non-real images
            real_out = self.D(x_real)
            output = self.D_head(real_out)
            d_real_loss = self.adv_loss(output, self.real)

            z_recon, _, _ = self.EN(x_real)
            x_recon = self.DE(z_recon, code_R)
            recon_out = self.D(x_recon)
            output = self.D_head(recon_out)
            d_recon_loss = self.adv_loss(output, self.fake)
            
            z_recon, mean, var = self.EN(x_real)
            x_fake = self.DE(z_recon, code_F)
            fake_out = self.D(x_fake)
            output = self.D_head(fake_out)
            d_fake_loss = self.adv_loss(output, self.fake)

            d_loss = d_real_loss + 0.5 * (d_recon_loss + d_fake_loss)
            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            # Training G and E --- 》 recon images （non-real）
            for i in range(2):
                eg_opt.zero_grad()
                z_recon, _, _ = self.EN(x_real)
                x_recon = self.DE(z_recon, code_R)
                eg_recon_loss0 = (self.ssim_loss(x_recon, x_real, window_size=19) + 1.5 * self.cont_loss(x_recon, x_real))

                outputA = self.D(x_recon)
                outputB = self.D(x_real)
                eg_recon_loss1 = self.cont_loss(outputA, outputB)

                outputA = self.D(x_recon)
                output = self.D_head(outputA)
                eg_recon_loss2 = self.adv_loss(output, self.real)

                eg_loss = ((eg_recon_loss0 + eg_recon_loss2 + eg_recon_loss1))
                eg_loss.backward()
                eg_opt.step()

            # Training G and Q --- 》 obtaining mutual information
            info_opt.zero_grad()
            z_recon, mean, var = self.EN(x_real)
            x_fake = self.DE(z_recon, code_F)
            outputB = self.D(x_fake)
            C = self.D_head(outputB)
            g_loss2 = self.adv_loss(C, self.real)

            code_fake = self.C_head(outputB)
            g_loss3 = (0.5 * self.cate_loss(code_fake[:, :self.params.label_dim],
                                      code_F[:, :self.params.label_dim]) + 0.05 * self.cont_loss(
                code_fake[:, -self.params.varying_dim:], code_F[:, -self.params.varying_dim:]))

            info_loss = g_loss3 + g_loss2
            info_loss.backward()
            info_opt.step()

            logging.info(
                f"Epoch {current_epoch + 1}/{self.params.max_epoch}, batch: {idx}, C: {c_loss:.5f}, D: {d_loss:.5f}, Info: {info_loss:.5f}, Recon: {eg_loss:.5f}")

    def stage2(self, loader, current_epoch, c_opt, d_opt, info_opt, en_opt, eg_opt, g_opt):
        """ First, construct a semantic-oriented manifold at stage 1 """
        for idx, (data, label) in enumerate(loader):
            x_real, logit = data.to(self.params.device), label.to(self.params.device)
            # Creating self-supervised vectors:
            # Generating input sets of fake images
            z = Variable(self.F(np.random.normal(0, 1, (x_real.shape[0], self.params.noise_dim))), requires_grad=False)
            label_nmupy = label.cpu().numpy()
            label_input = self.to_categorical(label_nmupy, num_columns=self.params.label_dim)

            code_input_F = Variable(self.F(np.random.uniform(-1, 1, (x_real.shape[0], self.params.varying_dim))),
                                    requires_grad=False)

            # Generating ground truth for real and fake images:
            self.real = Variable(self.F(x_real.shape[0], 1).fill_(1.0), requires_grad=False)
            self.fake = Variable(self.F(x_real.shape[0], 1).fill_(0.0), requires_grad=False)

            # Training Q --- 》 classifying real images
            for i in range(1):
                c_opt.zero_grad()
                real_out = self.D(x_real)
                output = self.C_head(real_out)
                c_loss = self.cate_loss(output[:, :self.params.label_dim], label_input)
                c_loss.backward()
                c_opt.step()

            # # real code
            real_out = self.D(x_real)
            output = self.C_head(real_out)
            varying_R = output[:, -self.params.varying_dim:].detach()
            labels = torch.max(output[:, :self.params.label_dim], dim=1)[1]
            label_nmupy = labels.cpu().numpy()
            label_inputs = self.to_categorical(label_nmupy, num_columns=self.params.label_dim)
            code_R = torch.cat((label_inputs, varying_R), dim=-1)

            for i in range(code_input_F.size(0)):
                while list(code_input_F[i]) == (varying_R[i]):
                    code_input_F[i] = Variable(self.F(np.random.uniform(-1, 1, (1, self.params.varying_dim))),
                                               requires_grad=False)
            code_F = torch.cat((label_inputs, code_input_F), dim=-1)

            # Training EN --- 》 encoding latent vectors
            en_opt.zero_grad()
            z_recon, mean, var = self.EN(x_real)
            eg_recon_loss0 = (self.loss_function(mean, var) / x_real.size(0))

            en_loss = (eg_recon_loss0)
            en_loss.backward()
            en_opt.step()

            # Training D --- 》 discriminating real and non-real images
            real_out = self.D(x_real)
            output = self.D_head(real_out)
            d_real_loss = self.adv_loss(output, self.real)

            z_recon, _, _ = self.EN(x_real)
            x_recon = self.DE(z_recon, code_R)
            recon_out = self.D(x_recon)
            output = self.D_head(recon_out)
            d_recon_loss = self.adv_loss(output, self.fake)

            z_recon, mean, var = self.EN(x_real)
            x_fake = self.DE(z_recon, code_F)
            fake_out = self.D(x_fake)
            output = self.D_head(fake_out)
            d_fake_loss = self.adv_loss(output, self.fake)

            d_loss = d_real_loss + 0.5 * (d_recon_loss + d_fake_loss)
            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            # Training G and E --- 》 recon images （non-real）
            for i in range(2):
                eg_opt.zero_grad()
                z_recon, _, _ = self.EN(x_real)
                x_recon = self.DE(z_recon, code_R)
                eg_recon_loss0 = (
                        self.ssim_loss(x_recon, x_real, window_size=19) + 1.5 * self.cont_loss(x_recon, x_real))

                outputA = self.D(x_recon)
                outputB = self.D(x_real)
                eg_recon_loss1 = self.cont_loss(outputA, outputB)

                outputA = self.D(x_recon)
                output = self.D_head(outputA)
                eg_recon_loss2 = self.adv_loss(output, self.real)

                eg_loss = ((eg_recon_loss0 + eg_recon_loss2 + eg_recon_loss1))
                eg_loss.backward()
                eg_opt.step()

            # Training G and Q --- 》 obtaining mutual information
            info_opt.zero_grad()
            z_recon, mean, var = self.EN(x_real)
            x_fake = self.DE(z_recon, code_F)
            outputB = self.D(x_fake)
            C = self.D_head(outputB)
            g_loss2 = self.adv_loss(C, self.real)

            code_fake = self.C_head(outputB)
            g_loss3 = (self.cate_loss(code_fake[:, :self.params.label_dim],
                                    code_F[:, :self.params.label_dim]) + 0.05 * self.cont_loss(
                code_fake[:, -self.params.varying_dim:], code_F[:, -self.params.varying_dim:]))
            
            adv_loss = - 0.75 * torch.log(1-torch.exp(- self.cate_loss(self.target[0](x_fake), label_inputs)))

            ssim_loss = self.ssim_loss(x_fake, x_real)

            info_loss = g_loss3 + 1.25 * g_loss2 + adv_loss + ssim_loss
            info_loss.backward()
            info_opt.step()

            logging.info(
                f"Epoch {current_epoch + 1}/{self.params.max_epoch}, batch: {idx}, C: {c_loss:.5f}, D: {d_loss:.5f}, Info: {info_loss:.5f}, ADV:{adv_loss:.5f}, Non-ADV:{g_loss3 + g_loss2:.5f}, Recon: {eg_loss:.5f}")


    def setup_logging(self):
        self.current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(root, self.params.log_dir, self.params.database, self.current_time)

        if not os.path.exists(log_file):
            os.makedirs(log_file)
        param_file = os.path.join(log_file, f"{self.current_time}-hyper-paras.txt")
        log_file = os.path.join(log_file, f"{self.current_time}.txt")

        logging.basicConfig(filename=log_file,
                            level=logging.INFO,
                            format='%(asctime)s [%(levelname)s] %(message)s')

        with open(param_file, 'w') as f:
            for arg in vars(self.params):
                attr_value = getattr(self.params, arg)
                f.write(f'{arg} = {attr_value}\n')

    def ssim_loss(self, img1, img2, window_size=19, C1 = 0.01 ** 2, C2 = 0.03 ** 2): # 19
        mu1 = F.conv2d(img1, torch.ones(1, 1, window_size, window_size).to(img1.device) / (window_size ** 2),
                       padding=window_size // 2)
        mu2 = F.conv2d(img2, torch.ones(1, 1, window_size, window_size).to(img2.device) / (window_size ** 2),
                       padding=window_size // 2)

        sigma1_sq = F.conv2d(img1 ** 2, torch.ones(1, 1, window_size, window_size).to(img1.device) / (window_size ** 2),
                             padding=window_size // 2) - mu1 ** 2
        sigma2_sq = F.conv2d(img2 ** 2, torch.ones(1, 1, window_size, window_size).to(img2.device) / (window_size ** 2),
                             padding=window_size // 2) - mu2 ** 2
        sigma12 = F.conv2d(img1 * img2, torch.ones(1, 1, window_size, window_size).to(img1.device) / (window_size ** 2),
                           padding=window_size // 2) - mu1 * mu2

        luminance = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
        contrast = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        ssim = luminance * contrast
        mean_ssim = ssim.mean()
        return 1 - mean_ssim

    def loss_function(self, mean, log_std):
        var = torch.pow(torch.exp(log_std), 2)
        KLD = -0.5 * torch.sum(1 + torch.log(var) - torch.pow(mean, 2) - var)
        return KLD

    def call_and_save(self):
        if self.params.stage == "stage1":
            torch.save({"EN": self.EN.state_dict(),
                    "DE": self.DE.state_dict(),
                    "D": self.D.state_dict(),
                    "D_head": self.D_head.state_dict(),
                    "C_head": self.C_head.state_dict()},
                   os.path.join(root, self.params.log_dir, self.params.database, self.current_time, "stage1-latest.pth"))
        elif self.params.stage == "stage2":
            torch.save({"EN": self.EN.state_dict(),
                    "DE": self.DE.state_dict(),
                    "D": self.D.state_dict(),
                    "D_head": self.D_head.state_dict(),
                    "C_head": self.C_head.state_dict()},
                   os.path.join(root, self.params.log_dir, self.params.database, self.current_time, "stage2-latest.pth"))

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def to_digits(self, x):
        cls = []
        for one_hot in x:
            id = torch.nonzero(one_hot).item()
            cls.append(id)
            break
        return cls

    def best_metric(self, metric):
        self.patience += 1
        if metric >= self.acc:
            torch.save({
                    "EN": self.EN.state_dict(),
                    "DE": self.DE.state_dict(),
                    "D": self.D.state_dict(),
                    "D_head": self.D_head.state_dict(),
                    "C_head": self.C_head.state_dict()},
            os.path.join(root, self.params.log_dir, self.params.database, self.current_time, "{}-best.pth".format(self.params.stage)))
        if self.patience % 5 == 0:
            BestPATH = os.path.join(root, self.params.log_dir, self.params.database, self.current_time, "{}-best.pth".format(self.params.stage))
            self.EN.load_state_dict(torch.load(BestPATH)["EN"])
            self.DE.load_state_dict(torch.load(BestPATH)["DE"])
            self.D.load_state_dict(torch.load(BestPATH)["D"])
            self.D_head.load_state_dict(torch.load(BestPATH)["D_head"])
            self.C_head.load_state_dict(torch.load(BestPATH)["C_head"])
            self.acc = metric
    def show_images(self):

        count = 0
        correct = correct0 = correct1 = total = 0
        switch_1 = 0
        recon_loss = 0
        for idx, (data, labels) in enumerate(self.all_loader()):
            count += 1
            with torch.no_grad():
                x_real, label_real = data.to(self.params.device), labels.to(self.params.device)
                label_nmupy = label_real.cpu().numpy()
                label_input = self.to_categorical(label_nmupy, num_columns=self.params.label_dim)
                total += label_real.size(0)
                real_out = self.D(x_real)
                code_recon = self.C_head(real_out)
                z_recon, mean, var = self.EN(x_real)
                labels = torch.max(code_recon[:, :self.params.label_dim], dim=1)[1]
                label_nmupy = labels.cpu().numpy()
                label_inputs = self.to_categorical(label_nmupy, num_columns=self.params.label_dim)
                code_F = torch.cat((label_inputs, code_recon[:, -self.params.varying_dim:]), dim=-1)
                x_recon = self.DE(z_recon, code_F)
                recon_loss += self.ssim_loss(x_recon, x_real, window_size=19).item()

                correct += (torch.argmax(self.target[0](x_recon), dim=1) == label_real).sum().item()
                for i in range(x_recon.size(0)):
                    z = z_recon[i].view(1, self.params.noise_dim).repeat(self.params.c_split ** 2, 1)
                    the_same_d = Variable(self.F(np.zeros((self.params.c_split ** 2, self.params.label_dim))))
                    the_same_d_cls = the_same_d
                    for j in range(the_same_d_cls.shape[0]):
                        the_same_d_cls[j][label_real[i].item()] = 1.0
                    the_same_d_cls = self.F(the_same_d_cls)
                    different_c1 = np.linspace(-1.0, 1.0, self.params.c_split)
                    different_c1 = different_c1.reshape(len(different_c1), 1)
                    different_c2 = np.linspace(-1.0, 1.0, self.params.c_split)
                    different_c2 = different_c2.reshape(len(different_c2), 1)
                    x, y = np.meshgrid(different_c1, different_c2)
                    result = np.column_stack((x.ravel(), y.ravel()))
                    different_c = Variable(self.F(result))
                    image_sa = self.DE(z, torch.cat((the_same_d_cls, different_c), dim=-1))
                    if (i % 10) == 0:
                        save_image(self.denorm(image_sa), os.path.join(self.params.save_fig_name_stage1, "{}_stage1-varying-matrix.png".format(i)),
                                nrow=int(self.params.showing_num))

                    for k in range(self.params.c_split ** 2):
                        logit_sa_1 = self.target[0](image_sa[k].unsqueeze(0))
                        if torch.argmax(logit_sa_1, dim=1).item() != label_real[i].item():
                            switch_1 += 1
                            break
            save_image(self.denorm(
                torch.cat((x_real[:int(self.params.showing_num * 6)], x_recon[:int(self.params.showing_num * 6)],
                        ((x_recon - x_real)[:int(self.params.showing_num * 6)]).clamp_(-1, 1)),
                        dim=0)),
                os.path.join(self.params.save_fig_name_stage1, "{}_stage1-varying.png".format(count)), 
                nrow=int(self.params.showing_num * 2))

            if count // 100 == 0:
                break

        print(f"Reconstruction accuracy {(correct / total):.4f}")
        if switch_1:
            print(f"Attack success rates {((switch_1) / (total)):.4f}")
        return (correct / total), (switch_1 / total), recon_loss

    def train_epoch(self):
        random.seed(self.params.seed)
        torch.manual_seed(self.params.seed)
        for e in range(self.params.max_epoch):
            self.EN.train()
            self.DE.train()
            self.D.train()
            self.D_head.train()
            self.C_head.train()
            loader = self.all_loader()
            c_opt, d_opt, info_opt, en_opt, eg_opt, g_opt = self.all_optimizers()
            if self.params.stage == "stage1":

                self.stage1(loader, e, c_opt, d_opt, info_opt, en_opt, eg_opt)
            elif self.params.stage == "stage2":
                # self.EN.eval()
                # self.D.eval()
                self.stage2(loader, e, c_opt, d_opt, info_opt, en_opt, eg_opt, g_opt)
            self.EN.eval()
            self.DE.eval()
            self.D.eval()
            self.D_head.eval()
            self.C_head.eval()
            self.call_and_save()
            switch = self.show_images()
            if self.params.stage == "stage1":
                self.best_metric(switch[0] + 1 - switch[1])
            else:
                self.best_metric(0.6 * switch[0] + 0.4 * switch[1] + 0.01 * (1 / switch[2]))
            with open(os.path.join(root, self.params.log_dir, self.params.database, self.current_time, "RECON_ACC--ASR.txt"),"a") as f:
                f.write("{} {} {}".format(switch[0], switch[1], 0.6 * switch[0] + 0.4 * switch[1] + 0.01 * (1 / switch[2])))
                f.write("\n")
        logging.info("Finishing Experiments")
        pass