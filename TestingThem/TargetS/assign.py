import os, sys, random
sys.path.append('..')
from torchvision.utils import save_image
from backbone import Encoder, Decoder, D_BACKBONE, D_HEAD, CODE_HEAD
from DataM import DataM
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import CNNs.backbones as b
import random
from torch.utils.data import Dataset, Subset, ConcatDataset
import numpy as np
import cv2
import torch
from torch.autograd import Variable
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
import torch.nn.functional as F
# torch.backends.cudnn.enabled = False

from pytorch_grad_cam import HiResCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from PIL import Image
import matplotlib.pyplot as plt

import os
from torchvision.utils import save_image

import foolbox as fb
import torchattacks as ta
import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, indicator=["pyproject.toml"], pythonpath=True, dotenv=True)

sns.set_context({"figure.figsize": (20, 20)})
plt.rcParams['font.size'] = 52

def random_sampling_torch_dataset(dataset, num_samples_per_class):
    class_labels = torch.unique(dataset.targets)
    class_samples = []
    class_labels_sampled = []

    for class_label in class_labels:
        class_indices = torch.where(dataset.targets == class_label)[0]
        num_samples = len(class_indices)

        if num_samples_per_class < num_samples:
            sampled_indices = random.sample(class_indices.tolist(), num_samples_per_class)
        else:
            sampled_indices = class_indices.tolist()

        class_samples.extend(dataset.data[sampled_indices])
        class_labels_sampled.extend(dataset.targets[sampled_indices])

    sampled_dataset = Subset(dataset, indices=range(len(class_samples)))
    sampled_dataset.data = torch.stack(class_samples)
    sampled_dataset.targets = torch.tensor(class_labels_sampled)

    return sampled_dataset

class AssignTarget():
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.loader = self.assign()

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

        self.EN.load_state_dict(torch.load(self.params.resume_path)["EN"])
        self.DE.load_state_dict(torch.load(self.params.resume_path)["DE"])
        self.D.load_state_dict(torch.load(self.params.resume_path)["D"])
        self.D_head.load_state_dict(torch.load(self.params.resume_path)["D_head"])
        self.C_head.load_state_dict(torch.load(self.params.resume_path)["C_head"])

        self.EN.eval()
        self.DE.eval()
        self.D.eval()
        self.D_head.eval()
        self.C_head.eval()

        if self.params.using_AT_ND:
            all_model = ["mobilenetv2"]
            self.whitesAT = b.CustomNet(all_model[0], self.params.database)
            self.whitesAT.load_state_dict(torch.load(os.path.join(root,
                                                                self.params.log_dir, self.params.file,
                                                                "AT-{}-{}.pth".format(
                                                                    self.params.database,
                                                                    all_model[0]))))
            self.whitesAT.to(self.params.device)
            self.whitesAT.eval()

            self.whitesND = b.CustomNet(all_model[0], self.params.database)
            self.whitesND.load_state_dict(torch.load(os.path.join(root,
                                                                self.params.log_dir, self.params.file,
                                                                "ND-S-{}-{}.pth".format(
                                                                    self.params.database,
                                                                    all_model[0]))))
            self.whitesND.to(self.params.device)
            self.whitesND.eval()

            self.whites = b.CustomNet(all_model[0], self.params.database)
            self.whites.load_state_dict(torch.load(os.path.join(root,
                                                                self.params.log_dir, self.params.file,
                                                                "{}-{}.pth".format(
                                                                    self.params.database,
                                                                    all_model[0]))))
            self.whites.to(self.params.device)
            self.whites.eval()

        else:
            all_model = ["mobilenetv2", "resnet18", "resnext50"]

            # White boxes
            self.whites = b.CustomNet(all_model[0], self.params.database)
            self.whites.load_state_dict(torch.load(os.path.join(root,
                    self.params.log_dir, self.params.file,
                    "{}-{}.pth".format(
                    self.params.database,
                    all_model[0]))))
            self.whites.to(self.params.device)
            self.whites.eval()

            # Black boxes
            self.blacks = []
            for i in range(1,3):
                target = b.CustomNet(all_model[i], self.params.database)
                target.load_state_dict(torch.load(os.path.join(root,
                    self.params.log_dir, self.params.file,
                    "{}-{}.pth".format(
                    self.params.database,
                    all_model[i]))))
                target.to(self.params.device)
                target.eval()
                self.blacks.append(target)

        self.F = FloatTensor
        self.L = LongTensor

    def all_loader(self):
        all = DataM(self.params)
        train_dataset, test_dataset = all.return_all_dataset()
        return train_dataset, test_dataset

    def assign(self):
        train_dataset, test_dataset = self.all_loader()
        sampled_datasets = []
        if self.params.database == "surface":
            sampled_dataset = train_dataset
        else:
            for i in [train_dataset, test_dataset]:
                sampled_dataset = random_sampling_torch_dataset(i, self.params.target_images_num)
                sampled_datasets.append(sampled_dataset)
            sampled_dataset = ConcatDataset(sampled_datasets)
        print("All categories: {}, Target size: {}".format(self.params.label_dim, len(sampled_dataset)))
        self.balanced_size = len(sampled_dataset)
        self.balanced_loader = DataLoader(
            dataset=sampled_dataset,
            batch_size=1,
            num_workers=self.params.num_workers,
            pin_memory=self.params.pin_memory,
            shuffle=True
        )

    def break_loops(self):
        '''
        max_step: max division step (> 1)
        step_size: step size per division for semantic codes
        '''
        # testing attack success rate
        total = 0
        break_asr1 = break_asr2 = 0
        for idx, (x_real, label) in enumerate(self.balanced_loader):
            x_real, label = x_real.to(self.params.device), label.to(self.params.device)
            if x_real.dim() != 4:
                x_real = x_real.unsqueeze(0)
            logit = self.whites(x_real)
            if torch.argmax(logit, dim=1).item() == label.item():
                z_recon, _, _ = self.EN(x_real)
                output = self.D(x_real)
                code = self.C_head(output)

                varying_R = code[:, -self.params.varying_dim:].detach()
                labels = torch.max(code[:, :self.params.label_dim], dim=1)[1]
                label_nmupy = labels.cpu().numpy()
                label_inputs = self.to_categorical(label_nmupy, num_columns=self.params.label_dim)
                code_R = torch.cat((label_inputs, varying_R), dim=-1)
                x_recon = self.DE(z_recon, code_R)

                logit = self.whites(x_recon)
                if (torch.argmax(logit, dim=1).item() == label.item()):
                    with torch.no_grad():
                        total += 1
                        print(total)

                        A = (code[0][-self.params.varying_dim:][0]).item()
                        B = (code[0][-self.params.varying_dim:][1]).item()

                        def coding(c_centerA, c_centerB, c_split, step_size):
                            '''
                            c_center: center
                            c_split: split
                            '''
                            c1 = c_centerA + c_split * step_size
                            c2 = c_centerB + c_split * step_size
                            c3 = c_centerA - c_split * step_size
                            c4 = c_centerB - c_split * step_size

                            square_size = 2 * c_split + 1

                            different_c1 = np.linspace(c4, c2, square_size)
                            different_c1 = different_c1.reshape(len(different_c1), 1)
                            different_c2 = np.full((square_size, 1), c3)
                            result1 = np.column_stack((different_c1, different_c2))

                            different_c1 = np.linspace(c4, c2, square_size)
                            different_c1 = different_c1.reshape(len(different_c1), 1)
                            different_c2 = np.full((square_size, 1), c1)
                            result2 = np.column_stack((different_c1, different_c2))

                            different_c1 = np.linspace(c3, c1, square_size)
                            different_c1 = different_c1.reshape(len(different_c1), 1)
                            different_c2 = np.full((square_size, 1), c4)
                            result3 = np.column_stack((different_c1, different_c2))

                            different_c1 = np.linspace(c3, c1, square_size)
                            different_c1 = different_c1.reshape(len(different_c1), 1)
                            different_c2 = np.full((square_size, 1), c2)
                            result4 = np.column_stack((different_c1, different_c2))

                            result = np.vstack((result1, result2, result3, result4))
                            different_c = Variable(FloatTensor(result))
                            return different_c.clamp_(-1.0, 1.0)

                        for i in range(self.params.max_step):
                            torch.cuda.empty_cache()
                            code_diff = coding(A, B, i, self.params.step_size).clamp_(-1,1)

                            code_size = code_diff.size(0)
                            z = z_recon.repeat(code_size, 1)
                            the_same_d = Variable(self.F(np.zeros((code_size, self.params.label_dim))))
                            the_same_d_cls = the_same_d
                            for j in range(the_same_d_cls.shape[0]):
                                the_same_d_cls[j][label.item()] = 1.0
                            the_same_d_cls = self.F(the_same_d_cls)
                            images = self.DE(z, torch.cat((the_same_d_cls, code_diff), dim=-1))
                            # for i in range(0, images.size(0), 1):
                            logit = self.whites(images)
                            label_real = torch.full(torch.argmax(logit, dim=1).shape, label.item()).to(self.params.device)
                            if not (torch.argmax(logit, dim=1).tolist() == label_real.tolist()):

                                logit = self.whitesAT(images)
                                if not (torch.argmax(logit, dim=1).tolist() == label_real.tolist()):
                                    wrong_indices = torch.nonzero(torch.argmax(logit, dim=1) != label_real,
                                                                  as_tuple=False).squeeze()
                                    if not isinstance(wrong_indices, list):
                                        wrong_indices = wrong_indices.tolist()
                                    if not isinstance(wrong_indices, list):
                                        wrong_indices = [wrong_indices]
                                    for k in wrong_indices:
                                        semantic = images[k]
                                        if semantic.dim() != 4:
                                            semantic = semantic.unsqueeze(0)
                                    if ((torch.norm((semantic - x_recon), p=2)).item() <= (torch.tensor(float("inf"))).item()):
                                        break_asr1 += 1
                                        print("Yes1",break_asr1)
                                    break
                            else:
                                continue

                        for i in range(200):
                            torch.cuda.empty_cache()
                            code_diff = coding(A, B, i, self.params.step_size).clamp_(-1,1)

                            code_size = code_diff.size(0)
                            z = z_recon.repeat(code_size, 1)
                            the_same_d = Variable(self.F(np.zeros((code_size, self.params.label_dim))))
                            the_same_d_cls = the_same_d
                            for j in range(the_same_d_cls.shape[0]):
                                the_same_d_cls[j][label.item()] = 1.0
                            the_same_d_cls = self.F(the_same_d_cls)
                            images = self.DE(z, torch.cat((the_same_d_cls, code_diff), dim=-1))
                            # for i in range(0, images.size(0), 1):
                            logit = self.whites(images)
                            label_real = torch.full(torch.argmax(logit, dim=1).shape, label.item()).to(self.params.device)
                            if not (torch.argmax(logit, dim=1).tolist() == label_real.tolist()):

                                logit = self.whitesND(images)
                                if not (torch.argmax(logit, dim=1).tolist() == label_real.tolist()):
                                    wrong_indices = torch.nonzero(torch.argmax(logit, dim=1) != label_real,
                                                                  as_tuple=False).squeeze()
                                    if not isinstance(wrong_indices, list):
                                        wrong_indices = wrong_indices.tolist()
                                    if not isinstance(wrong_indices, list):
                                        wrong_indices = [wrong_indices]
                                    for k in wrong_indices:
                                        semantic = images[k]
                                        if semantic.dim() != 4:
                                            semantic = semantic.unsqueeze(0)
                                    if ((torch.norm((semantic - x_recon), p=2)).item() <= (torch.tensor(float("inf"))).item()):
                                        break_asr2 += 1
                                        print("Yes2",break_asr2)
                                    break
                            else:
                                continue

        break1 = break_asr1 / total
        break2 = break_asr2 / total
        with open(os.path.join(self.params.target_file, self.params.file, str(self.params.seed),
                               "Ours-Break.txt"),
                  "a") as f:
            f.write("Results")
            f.write("\n")
            f.write("Seed = {}".format(self.params.seed))
            f.write("\n")
            f.write(
                "Breaking AT = {}".format(break1))
            f.write("\n")
            f.write(
                "Breaking ND = {}".format(break2))
            f.write("\n" + "\n")

    def to_categorical(self, y, num_columns):
        """Returns one-hot encoded Variable"""
        y_cat = np.zeros((y.shape[0], num_columns))
        y_cat[range(y.shape[0]), y] = 1.0
        return Variable(self.F(y_cat), requires_grad=False)

    def _Attack(self, model, recon, label, attack_name, average_epsilon):
        recon = ((recon + 1) / 2).clamp_(0, 1)
        fmodel = fb.PyTorchModel(model, device=self.params.device, bounds=(-1, 1))
        if attack_name == "L2PGD":
            attack = ta.PGDL2(model, eps=average_epsilon, alpha=average_epsilon / 10., steps=12, random_start=True)
            advs = attack(recon, label)

        if attack_name == "L2APGD":
            attack = ta.APGD(model, norm='L2', eps=average_epsilon, steps=12, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False)
            advs = attack(recon, label)

        if attack_name == "L2PGDRS":
                attack = ta.PGDRSL2(model, eps=average_epsilon, alpha=average_epsilon / 10., steps=12, noise_type="guassian", noise_sd=0.5, noise_batch_size=5, batch_max=2048)
                advs = attack(recon, label)

        if attack_name == "L2CW":
            # attack = ta.CW(model, steps=10000, c=1, lr=0.1, kappa=0)
            attack = ta.CW(model, steps=10000, c=1, lr=3.0, kappa=0)
            advs = attack(recon, label)

        if attack_name == "L2DeepFool":
            attack = ta.DeepFool(model, steps=50, overshoot=0.02)
            advs = attack(recon, label)

        if attack_name == "L2Square":
            attack = ta.Square(model, norm='L2', eps=average_epsilon, n_queries=5000, n_restarts=1, p_init=.8, seed=0, verbose=False, loss='margin', resc_schedule=True)
            advs = attack(recon, label)

        if attack_name == "L2AA":
            attack = ta.AutoAttack(model, norm='L2', eps=average_epsilon, n_classes=10, seed=12345, verbose=False)
            advs = attack(recon, label)

        return advs

    def our_loops(self):
        '''
        max_step: max division step (> 1)  
        step_size: step size per division for semantic codes
        '''
        # testing attack success rate
        total = asr = 0
        label_all = []
        recon_all = []
        idx_all = []
        for idx, (x_real, label) in enumerate(self.balanced_loader):
            x_real, label = x_real.to(self.params.device), label.to(self.params.device)
            if x_real.dim() != 4:
                x_real = x_real.unsqueeze(0)
            logit = self.whites(x_real)
            if torch.argmax(logit, dim=1).item() == label.item():
                z_recon, _, _ = self.EN(x_real)
                output = self.D(x_real)
                code = self.C_head(output)

                varying_R = code[:, -self.params.varying_dim:].detach()
                labels = torch.max(code[:, :self.params.label_dim], dim=1)[1]
                label_nmupy = labels.cpu().numpy()
                label_inputs = self.to_categorical(label_nmupy, num_columns=self.params.label_dim)
                code_R = torch.cat((label_inputs, varying_R), dim=-1)
                x_recon = self.DE(z_recon, code_R)

                logit = self.whites(x_recon)
                if (torch.argmax(logit, dim=1).item() == label.item()):
                    with torch.no_grad():
                        total += 1
                        idx_all.append(idx)
                        recon_all.append(x_recon)
                        label_all.append(label)
                        self.save_these_image(self.denorm(x_real), idx, "real-image", n=1)
                        self.save_these_image(self.denorm(x_recon), idx,"recon-image", n=1)

                        A = (code[0][-self.params.varying_dim:][0]).item()
                        B = (code[0][-self.params.varying_dim:][1]).item()

                        def coding(c_centerA, c_centerB, c_split, step_size):
                            '''
                            c_center: center
                            c_split: split
                            '''
                            c1 = c_centerA + c_split * step_size
                            c2 = c_centerB + c_split * step_size
                            c3 = c_centerA - c_split * step_size
                            c4 = c_centerB - c_split * step_size

                            square_size = 2 * c_split + 1
                            # c_lu = (c4, c3), c_ru = (c2, c3), c_lb = (c4, c1), c_rb = (c2, c1)

                            different_c1 = np.linspace(c4, c2, square_size)
                            different_c1 = different_c1.reshape(len(different_c1), 1)
                            different_c2 = np.full((square_size, 1), c3)
                            result1 = np.column_stack((different_c1, different_c2))

                            different_c1 = np.linspace(c4, c2, square_size)
                            different_c1 = different_c1.reshape(len(different_c1), 1)
                            different_c2 = np.full((square_size, 1), c1)
                            result2 = np.column_stack((different_c1, different_c2))

                            different_c1 = np.linspace(c3, c1, square_size)
                            different_c1 = different_c1.reshape(len(different_c1), 1)
                            different_c2 = np.full((square_size, 1), c4)
                            result3 = np.column_stack((different_c1, different_c2))

                            different_c1 = np.linspace(c3, c1, square_size)
                            different_c1 = different_c1.reshape(len(different_c1), 1)
                            different_c2 = np.full((square_size, 1), c2)
                            result4 = np.column_stack((different_c1, different_c2))
                            # x, y = np.meshgrid(different_c1, different_c2)
                            # result4 = np.column_stack((x.ravel(), y.ravel()))

                            result = np.vstack((result1, result2, result3, result4))
                            different_c = Variable(FloatTensor(result))
                            return different_c.clamp_(-1.0, 1.0)

                        our_examples = []
                        our_norms = []
                        our_ssim = []
                        for i in range(self.params.max_step):
                            torch.cuda.empty_cache()
                            code_diff = coding(A, B, i, self.params.step_size).clamp_(-1,1)

                            code_size = code_diff.size(0)
                            z = z_recon.repeat(code_size, 1)
                            the_same_d = Variable(self.F(np.zeros((code_size, self.params.label_dim))))
                            the_same_d_cls = the_same_d
                            for j in range(the_same_d_cls.shape[0]):
                                the_same_d_cls[j][labels.item()] = 1.0
                            the_same_d_cls = self.F(the_same_d_cls)
                            images = self.DE(z, torch.cat((the_same_d_cls, code_diff), dim=-1))
                            # for i in range(0, images.size(0), 1):
                            logit = self.whites(images)

                            label_real = torch.full(torch.argmax(logit, dim=1).shape, label.item()).to(self.params.device)
                            if not (torch.argmax(logit, dim=1).tolist() == label_real.tolist()):
                                wrong_indices = torch.nonzero(torch.argmax(logit, dim=1) != label_real,
                                                              as_tuple=False).squeeze()
                                if not isinstance(wrong_indices, list):
                                    wrong_indices = wrong_indices.tolist()
                                if not isinstance(wrong_indices, list):
                                    wrong_indices = [wrong_indices]
                                asr += 1
                                norm_list = []
                                for k in wrong_indices:
                                    semantic = images[k]
                                    if semantic.dim() != 4:
                                        semantic = semantic.unsqueeze(0)
                                    norm_list.append((torch.norm((semantic-x_recon), p=2)).item())
                                sorted_list = sorted(norm_list)[:2]
                                indices = [norm_list.index(elem) for elem in sorted_list]
                                for d in indices:
                                    our_examples.append(images[d])
                                self.save_these_image(self.denorm(our_examples[0].unsqueeze(0)), idx, "our_example_Pre={}_Real={}".format(torch.argmax(self.whites(semantic), dim=1).item(), label_real[k].item()), n=1)
                                AP = float(norm_list[indices[0]])
                                print(AP)
                                SSIM = float(self.ssim_loss(our_examples[0].unsqueeze(0), x_recon))
                                our_norms.append(AP)
                                our_ssim.append(SSIM)
                                torch.cuda.empty_cache()
                                # print(our_norms)
                                break
                            else:
                                continue
            torch.cuda.empty_cache()

        recon_acc = total / self.balanced_size
        our_asr = asr / total

        average_epsilon = sum(our_norms) / len(our_norms)
        average_ssim = sum(our_ssim) / len(our_ssim)

        with open(os.path.join(self.params.target_file, self.params.file, str(self.params.seed),
                               "Ours.txt"),
                  "a") as f:
            f.write("Results")
            f.write("\n")
            f.write("Seed = {}".format(self.params.seed))
            f.write("\n")
            f.write(
                "Reconstruction Accuracy = {}".format(recon_acc))
            f.write("\n")
            f.write(
                "ASR of MobileNev2 ({}) = ({})".format("ours", our_asr))
            f.write("\n")
            f.write("Average Perturbations = {}".format(average_epsilon))
            f.write("\n")
            f.write("Average SSIM = {}".format(average_ssim))
            f.write("\n" + "\n")

    def ssim_loss(self, img1, img2, window_size=19, C1=0.01 ** 2, C2=0.03 ** 2):  # 19
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

    def our_trans_loops(self):
        '''
        max_step: max division step (> 1)
        step_size: step size per division for semantic codes
        '''
        # testing attack success rate
        total = asr = ts1 = ts2 = asr = 0
        label_all = []
        recon_all = []
        idx_all = []
        for idx, (x_real, label) in enumerate(self.balanced_loader):
            x_real, label = x_real.to(self.params.device), label.to(self.params.device)
            if x_real.dim() != 4:
                x_real = x_real.unsqueeze(0)
            logit = self.whites(x_real)
            if torch.argmax(logit, dim=1).item() == label.item():
                z_recon, _, _ = self.EN(x_real)
                output = self.D(x_real)
                code = self.C_head(output)

                varying_R = code[:, -self.params.varying_dim:].detach()
                labels = torch.max(code[:, :self.params.label_dim], dim=1)[1]
                label_nmupy = labels.cpu().numpy()
                label_inputs = self.to_categorical(label_nmupy, num_columns=self.params.label_dim)
                code_R = torch.cat((label_inputs, varying_R), dim=-1)
                x_recon = self.DE(z_recon, code_R)

                logit = self.whites(x_recon)
                if torch.argmax(logit, dim=1).item() == label.item():
                    with torch.no_grad():
                        total += 1
                        idx_all.append(idx)
                        recon_all.append(x_recon)
                        label_all.append(label)
                        self.save_these_image(self.denorm(x_real), idx, "real-image", n=1)
                        self.save_these_image(self.denorm(x_recon), idx, "recon-image", n=1)

                        A = (code[0][-self.params.varying_dim:][0]).item()
                        B = (code[0][-self.params.varying_dim:][1]).item()

                        def coding(c_centerA, c_centerB, c_split, step_size):
                            '''
                            c_center: center
                            c_split: split
                            '''
                            c1 = c_centerA + c_split * step_size
                            c2 = c_centerB + c_split * step_size
                            c3 = c_centerA - c_split * step_size
                            c4 = c_centerB - c_split * step_size

                            square_size = 2 * c_split + 1
                            # c_lu = (c4, c3), c_ru = (c2, c3), c_lb = (c4, c1), c_rb = (c2, c1)

                            different_c1 = np.linspace(c4, c2, square_size)
                            different_c1 = different_c1.reshape(len(different_c1), 1)
                            different_c2 = np.full((square_size, 1), c3)
                            result1 = np.column_stack((different_c1, different_c2))

                            different_c1 = np.linspace(c4, c2, square_size)
                            different_c1 = different_c1.reshape(len(different_c1), 1)
                            different_c2 = np.full((square_size, 1), c1)
                            result2 = np.column_stack((different_c1, different_c2))

                            different_c1 = np.linspace(c3, c1, square_size)
                            different_c1 = different_c1.reshape(len(different_c1), 1)
                            different_c2 = np.full((square_size, 1), c4)
                            result3 = np.column_stack((different_c1, different_c2))

                            different_c1 = np.linspace(c3, c1, square_size)
                            different_c1 = different_c1.reshape(len(different_c1), 1)
                            different_c2 = np.full((square_size, 1), c2)
                            result4 = np.column_stack((different_c1, different_c2))
                            # x, y = np.meshgrid(different_c1, different_c2)
                            # result4 = np.column_stack((x.ravel(), y.ravel()))

                            result = np.vstack((result1, result2, result3, result4))
                            different_c = Variable(FloatTensor(result))
                            return different_c

                        for i in range(self.params.max_step):
                            torch.cuda.empty_cache()
                            code_diff = coding(A, B, i, self.params.step_size).clamp_(-1, 1)

                            code_size = code_diff.size(0)
                            z = z_recon.repeat(code_size, 1)
                            the_same_d = Variable(self.F(np.zeros((code_size, self.params.label_dim))))
                            the_same_d_cls = the_same_d
                            for j in range(the_same_d_cls.shape[0]):
                                the_same_d_cls[j][label.item()] = 1.0
                            the_same_d_cls = self.F(the_same_d_cls)
                            images = self.DE(z, torch.cat((the_same_d_cls, code_diff), dim=-1))
                            # for i in range(0, images.size(0), 1):
                            logit = self.whites(images)

                            label_real = torch.full(torch.argmax(logit, dim=1).shape, label.item()).to(
                                self.params.device)
                            if not (torch.argmax(logit, dim=1).tolist() == label_real.tolist()):
                                asr += 1
                                break
                            else:
                                continue

                        for i in range(self.params.max_step):
                            torch.cuda.empty_cache()
                            code_diff = coding(A, B, i, self.params.step_size).clamp_(-1,1)

                            code_size = code_diff.size(0)
                            z = z_recon.repeat(code_size, 1)
                            the_same_d = Variable(self.F(np.zeros((code_size, self.params.label_dim))))
                            the_same_d_cls = the_same_d
                            for j in range(the_same_d_cls.shape[0]):
                                the_same_d_cls[j][label.item()] = 1.0
                            the_same_d_cls = self.F(the_same_d_cls)
                            images = self.DE(z, torch.cat((the_same_d_cls, code_diff), dim=-1))
                            # for i in range(0, images.size(0), 1):
                            logit = self.blacks[0](images)

                            label_real = torch.full(torch.argmax(logit, dim=1).shape, label.item()).to(self.params.device)
                            if not (torch.argmax(logit, dim=1).tolist() == label_real.tolist()):
                                ts1 += 1
                                break
                            else:
                                continue

                        for i in range(self.params.max_step):
                            torch.cuda.empty_cache()
                            code_diff = coding(A, B, i, self.params.step_size).clamp_(-1,1)

                            code_size = code_diff.size(0)
                            z = z_recon.repeat(code_size, 1)
                            the_same_d = Variable(self.F(np.zeros((code_size, self.params.label_dim))))
                            the_same_d_cls = the_same_d
                            for j in range(the_same_d_cls.shape[0]):
                                the_same_d_cls[j][label.item()] = 1.0
                            the_same_d_cls = self.F(the_same_d_cls)
                            images = self.DE(z, torch.cat((the_same_d_cls, code_diff), dim=-1))
                            # for i in range(0, images.size(0), 1):
                            logit = self.blacks[1](images)

                            label_real = torch.full(torch.argmax(logit, dim=1).shape, label.item()).to(self.params.device)
                            if not (torch.argmax(logit, dim=1).tolist() == label_real.tolist()):
                                ts2 += 1
                                break
                            else:
                                continue

        ts_1 = ts1 / total
        ts_2 = ts2 / total

        with open(os.path.join(self.params.target_file, self.params.file, str(self.params.seed),
                               "Ours-t.txt"),
                  "a") as f:
            f.write("Results")
            f.write("\n")
            f.write("Seed = {}".format(self.params.seed))
            f.write("\n")
            f.write(
                "ASR of MobileNetv2 ({}) = ({})".format("ours", asr / total))
            f.write("\n")
            f.write(
                "ASR of ResNet18 ({}) = ({})".format("ours", ts_1))
            f.write("\n")
            f.write(
                "ASR of ResNeXt50 ({}) = ({})".format("ours", ts_2))
            f.write("\n" + "\n")

    def break_compare_loops(self, average_epsilon):
        '''
        max_step: max division step (> 1)
        step_size: step size per division for semantic codes
        '''
        # testing attack success rate
        total = adv_total = adv_total1 = adv_total2 = 0
        for idx, (x_real, label) in enumerate(self.balanced_loader):
            x_real, label = x_real.to(self.params.device), label.to(self.params.device)
            if x_real.dim() != 4:
                x_real = x_real.unsqueeze(0)
            logit = self.whites(x_real)
            if torch.argmax(logit, dim=1).item() == label.item():
                z_recon, _, _ = self.EN(x_real)
                output = self.D(x_real)
                code = self.C_head(output)

                varying_R = code[:, -self.params.varying_dim:].detach()
                labels = torch.max(code[:, :self.params.label_dim], dim=1)[1]
                label_nmupy = labels.cpu().numpy()
                label_inputs = self.to_categorical(label_nmupy, num_columns=self.params.label_dim)
                code_R = torch.cat((label_inputs, varying_R), dim=-1)
                x_recon = self.DE(z_recon, code_R)

                logit = self.whites(x_recon)
                if torch.argmax(logit, dim=1).item() == label.item():

                    total += 1
                    print(total)
                    x_adv = self._Attack(self.whites, x_recon, label, self.params.attack_name,
                                         average_epsilon)

                    logit = self.whitesAT((x_adv *2-1))
                    if torch.argmax(logit, dim=1).item() != label.item():
                        print("s-at")
                        adv_total += 1
                    logit_adv1 = self.whitesND(((x_adv *2-1)))
                    if torch.argmax(logit_adv1, dim=1).item() != label.item():
                        print("s-nd")
                        adv_total1 += 1

        break_asr1 = adv_total / total
        break_asr2 = adv_total1 / total

        with open(os.path.join(self.params.target_file, self.params.file, str(self.params.seed), "{}_Break.txt".format(self.params.attack_name)),
                  "a") as f:
            f.write("Results")
            f.write("\n")
            f.write("Seed = {}".format(self.params.seed))
            f.write("\n")
            f.write(
                "Break AT {}".format(break_asr1))
            f.write("\n")
            f.write(
                "Break ND {}".format(break_asr2))
            f.write("\n + \n")

    def compare_loops(self, average_epsilon):
        '''
        max_step: max division step (> 1)
        step_size: step size per division for semantic codes
        '''
        # testing attack success rate
        per = []
        SSIM= []
        total = adv_total = adv_total1 = adv_total2 = 0
        for idx, (x_real, label) in enumerate(self.balanced_loader):
            x_real, label = x_real.to(self.params.device), label.to(self.params.device)
            if x_real.dim() != 4:
                x_real = x_real.unsqueeze(0)
            logit = self.whites(x_real)
            if torch.argmax(logit, dim=1).item() == label.item():
                z_recon, _, _ = self.EN(x_real)
                output = self.D(x_real)
                code = self.C_head(output)

                varying_R = code[:, -self.params.varying_dim:].detach()
                labels = torch.max(code[:, :self.params.label_dim], dim=1)[1]
                label_nmupy = labels.cpu().numpy()
                label_inputs = self.to_categorical(label_nmupy, num_columns=self.params.label_dim)
                code_R = torch.cat((label_inputs, varying_R), dim=-1)
                x_recon = self.DE(z_recon, code_R)

                logit = self.whites(x_recon)
                if torch.argmax(logit, dim=1).item() == label.item():

                    total += 1
                    print(total)
                    x_adv = self._Attack(self.whites, x_recon, label, self.params.attack_name,
                                         average_epsilon)
                    if self.params.attack_name in [ "L2CW", "L2DeepFool"]:
                        per.append((torch.norm((((x_adv *2-1)) - x_recon), p=2)).item())
                    self.save_these_image(x_adv, idx, f"adv_example_{average_epsilon:.3f}", n=1)
                    SSIM.append(float(self.ssim_loss((x_adv *2-1), x_recon).item()))
                    logit = self.whites((x_adv *2-1))
                    if torch.argmax(logit, dim=1).item() != label.item():
                        print("s-cw")
                        adv_total += 1
                    logit_adv1 = self.blacks[0](((x_adv *2-1)))
                    if torch.argmax(logit_adv1, dim=1).item() != label.item():
                        adv_total1 += 1
                    logit_adv2 = self.blacks[1](((x_adv *2-1)))
                    if torch.argmax(logit_adv2, dim=1).item() != label.item():
                        adv_total2 += 1

        asr = adv_total / total
        other_trans_asr1 = adv_total1 / total
        other_trans_asr2 = adv_total2 / total
        average_SSIM = sum(SSIM) / len(SSIM)

        if self.params.attack_name in [ "L2CW", "L2DeepFool"]:
            average_epsilon = sum(per) / len(per)

        with open(os.path.join(self.params.target_file, self.params.file, str(self.params.seed), "{}.txt".format(self.params.attack_name)),
                  "a") as f:
            f.write("Results")
            f.write("\n")
            f.write("Seed = {}".format(self.params.seed))
            f.write("\n")
            f.write(
                "ASR of MobileNev2 ({}) = ({})".format(self.params.attack_name, asr))
            f.write("\n")
            f.write("ASR of ResNet18 ({}) = ({})".format(self.params.attack_name, other_trans_asr1))
            f.write("\n")
            f.write("ASR of ResNeXt50 ({}) = ({})".format(self.params.attack_name, other_trans_asr2))
            f.write("\n")
            f.write("Average Perturbations = {}".format(average_epsilon))
            f.write("\n")
            f.write("Average SSIM = {}".format(average_SSIM))
            f.write("\n + \n")

    def Inputing(self):
        the_same_z = np.tile(np.random.normal(0, 1, (1, self.params.noise_dim)), (self.params.max_step ** 2, 1))
        the_same_z = Variable(FloatTensor(the_same_z))

        the_same_d = Variable(FloatTensor(np.zeros((self.params.max_step ** 2, self.params.label_dim))))
        the_same_d_cls = the_same_d
        for j in range(the_same_d_cls.shape[0]):
            the_same_d_cls[j][self.params.target_cls] = 1.0
        the_same_d_cls = FloatTensor(the_same_d_cls)

        different_c1 = np.linspace(self.params.c1_left, self.params.c1_right, self.params.max_step)
        different_c1 = different_c1.reshape(len(different_c1), 1)
        different_c2 = np.linspace(self.params.c2_left, self.params.c2_right, self.params.max_step)
        different_c2 = different_c2.reshape(len(different_c2), 1)
        x, y = np.meshgrid(different_c1, different_c2)
        result = np.column_stack((x.ravel(), y.ravel()))
        different_c = Variable(FloatTensor(result))
        return the_same_z, the_same_d_cls, different_c

    def save_these_image(self, images, idx, types, n):
        save_dir = os.path.join(self.params.target_file, self.params.file, str(self.params.seed), str(idx))
        os.makedirs(save_dir, exist_ok=True)
        save_dir = os.path.join(self.params.target_file, self.params.file, str(self.params.seed), str(idx), "{}_{}.png".format(types, idx))
        save_image(images, save_dir, nrow=n)
        save_dir = os.path.join(self.params.target_file, self.params.file, str(self.params.seed), str(idx), "{}_{}.npy".format(types, idx))
        array = images.detach().cpu().numpy()
        np.save(save_dir, array)
        print("Image saving completed.")

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)
    
    def startloop(self):
        random.seed(self.params.seed)
        torch.manual_seed(self.params.seed)
        self.our_loops()
        # self.our_trans_loops()
        # self.compare_loops(15.89)
        # self.break_loops()
        # self.break_compare_loops(15.86)

    
    def HOT(self):
        np.random.seed(self.params.seed)
        the_same_z, the_same_d_cls, different_c = self.Inputing()
        image = self.DE(the_same_z, torch.cat((the_same_d_cls, different_c), dim=-1))
        save_image(self.denorm(image.detach()), os.path.join(self.params.hot_path,
                                                        "[VaryingImage]Cls={}.png".format(self.params.target_cls)),
                   nrow=int(self.params.c_split))

        with torch.no_grad():
            for k, m in enumerate([self.whites, self.blacks[0], self.blacks[1]]):
                logit = m(image)
                probs = torch.argmax(logit, dim=1)
                probs = probs.cpu().numpy().reshape(self.params.c_split, self.params.c_split)
                conf = torch.max(F.softmax(logit, dim=1), 1)
                conf = conf.values.round_(decimals=2).detach().cpu().numpy().reshape(self.params.c_split,
                                                                                     self.params.c_split)
                # plt.figure(dpi=1000)
                color = ['#8dd3c7', '#bebada', '#fdb462', '#b3de69', '#fb8072', '#80b1d3']
                # Create colormap for labels
                # cmap = plt.cm.get_cmap('rainbow', probs.max() + 1)
                cmap = colors.ListedColormap(color)
                # Create grid for hot map
                grid = np.zeros((self.params.c_split, self.params.c_split, 4))
                # Assign colors based on labels and confidence
                for a in range(self.params.c_split):
                    for b in range(self.params.c_split):
                        label = probs[a, b]
                        confidence = conf[a, b]
                        # print(cmap(label))
                        grid[a, b] = cmap(label)
                        grid[a, b] *= confidence
                        plt.text(b, a, f"{confidence:.2f}", ha='center', va='center', color='black', fontsize=52,
                                 fontname='serif')

                # Plot hot map
                plt.imshow(grid)

                norm = plt.Normalize(vmin=0, vmax=5)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, shrink=0.8, ticks=np.arange(5 + 1))
                cbar.set_label('Class={}'.format(self.params.target_cls), fontname='serif')
                cbar.ax.tick_params(labelsize=52)

                plt.xlabel('Semantic-Code1', fontname='serif')
                plt.ylabel('Semantic-Code2', fontname='serif')
                ticks = np.arange(0, self.params.c_split)
                plt.xticks(ticks)
                plt.yticks(ticks)
                if k == 0:
                    plt.savefig(
                    os.path.join(self.params.hot_path,
                                 "{}_Net={}[VaryingHotting]Cls={}.png".format(k, "M" , self.params.target_cls)))
                elif k == 1:
                    plt.savefig(
                    os.path.join(self.params.hot_path,
                                 "{}_Net={}[VaryingHotting]Cls={}.png".format(k, "R" , self.params.target_cls)))
                elif k == 2:
                    plt.savefig(
                    os.path.join(self.params.hot_path,
                                 "{}_Net={}[VaryingHotting]Cls={}.png".format(k, "RX" , self.params.target_cls)))
                plt.close()
    
    def heatmap(self, groups):
        np.random.seed(self.params.seed)

        def coding(c_varying, c_fixing):
            '''
            c_center: center
            c_split: split
            '''
            different_varying = np.linspace(self.params.c_left, self.params.c_right, self.params.max_step)
            different_varying = different_varying.reshape(len(different_varying), 1)

            different_fixing = np.full((self.params.max_step, 1), c_fixing)
            if c_varying == "c1":
                x, y = np.meshgrid(different_varying, different_fixing)
                result = np.column_stack((x.ravel(), y.ravel()))
            else:
                x, y = np.meshgrid(different_fixing, different_varying)
                result = np.column_stack((x.ravel(), y.ravel()))

            return result

        the_same_z, the_same_d_cls, different_c = self.Inputing()
        label = torch.tensor([self.params.target_cls]).cuda()

        image_sa = self.DE(the_same_z, torch.cat((the_same_d_cls, different_c), dim=-1))

        for i in range(len(groups)):
            if i == 0:
                target_images = image_sa[groups[i]].unsqueeze(0)
            else:
                target_images = torch.cat((target_images, image_sa[groups[i]].unsqueeze(0)), dim=0)

        # save real_images
        save_image(self.denorm(target_images), os.path.join(self.params.hot_path, "Real_Cls={}.png".format(self.params.target_cls)),nrow=int(len(groups)))

        # save real predictions
        logit = self.whites(target_images)
        probs = torch.argmax(logit, dim=1)
        conf = torch.max(F.softmax(logit, dim=1), 1)
        conf = conf.values.round_(decimals=4).detach().cpu().numpy()
        
        with open( os.path.join(self.params.hot_path, "Real_Cls={}.txt".format(self.params.target_cls)),"a") as f:
            f.write("Comparing Prediction Results")
            f.write("\n")
            f.write("Predictions and Confidences of MobileNev2 = {}, {}".format(probs, conf))
            f.write("\n")
            f.write("\n")
        
        logit = self.blacks[0](target_images)
        probs = torch.argmax(logit, dim=1)
        conf = torch.max(F.softmax(logit, dim=1), 1)
        conf = conf.values.round_(decimals=4).detach().cpu().numpy()
        
        with open( os.path.join(self.params.hot_path, "Real_Cls={}.txt".format(self.params.target_cls)),"a") as f:
            f.write("Comparing Prediction Results")
            f.write("\n")
            f.write("Predictions and Confidences of ResNet18 = {}, {}".format(probs, conf))
            f.write("\n")
            f.write("\n")

        logit = self.blacks[1](target_images)
        probs = torch.argmax(logit, dim=1)
        conf = torch.max(F.softmax(logit, dim=1), 1)
        conf = conf.values.round_(decimals=4).detach().cpu().numpy()
        
        with open( os.path.join(self.params.hot_path, "Real_Cls={}.txt".format(self.params.target_cls)),"a") as f:
            f.write("Comparing Prediction Results")
            f.write("\n")
            f.write("Predictions and Confidences of ResNext50 = {}, {}".format(probs, conf))
            f.write("\n")
            f.write("\n")

        # creating HiResCAM
        white_target = self.whites.model.features[11].conv[0]
        cam = HiResCAM(model=self.whites, target_layers=white_target, use_cuda=self.params.device)
        grad_image = cam(target_images, targets=[ClassifierOutputSoftmaxTarget(self.params.target_cls)], aug_smooth=True, eigen_smooth=True)[0]
        grad_image_rgb = np.repeat(grad_image[:, :, np.newaxis], 3, axis=2)
        visualization = show_cam_on_image(np.array(self.denorm(target_images).permute((0, 2, 3, 1)).detach().cpu().numpy()),grad_image_rgb)
        
        fig, axes = plt.subplots(1, len(groups))
        for i in range(len(groups)):
            axes[i].imshow(visualization[i])
            axes[i].axis('off')
        plt.subplots_adjust(wspace=0.0)
        plt.savefig(os.path.join(self.params.hot_path,"HiResCAM_Cls={}.png".format(self.params.target_cls)))
        
            
def normalize(I):
    norm = (I-I.mean())/I.std()
    norm = norm * 0.1
    norm = norm + 0.5
    norm = norm.clip(0, 1)
    return norm


import argparse
import torch

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=device, help="using device: cuda or cpu !")
    parser.add_argument("--c_split", type=int, default=10, help="-")
    parser.add_argument("--c1_left", type=float, default=-1.0, help="-")
    parser.add_argument("--c1_right", type=float, default=1.0, help="-")
    parser.add_argument("--c2_left", type=float, default=-1.0, help="-")
    parser.add_argument("--c2_right", type=float, default=1.0, help="-")

    parser.add_argument("--target_num", type=int, default=1, help="-")
    parser.add_argument("--target_cls", type=int, default=1, help="-")

    parser.add_argument("--resume_path", type=str, default="_", help="-")
    parser.add_argument("--hot_path", type=str, default="_", help="-")
    parser.add_argument("--target_file", type=str, default="_", help="-")
    parser.add_argument("--file", type=str, default="NEU-CLS", help="-")
    parser.add_argument("--log_dir", type=str, default="Logging", help="logging")
    parser.add_argument("--database", type=str, default="surface", help="using database ? ")
    parser.add_argument("--dataset", type=str, default="surface", help="using database ? ")
    parser.add_argument("--attack_name", type=str, default="L2CW", help="-")
    parser.add_argument("--using_AT_ND", type=bool, default=None, help="-")

    parser.add_argument("--target_images_num", type=int, default=5, help="-")
    # the number of samples = 5 (target_images_num) * 10 (label_dim) * 2 (traningset + testingset)
    parser.add_argument("--num_workers", type=int, default=0, help="-")
    parser.add_argument("--pin_memory", type=bool, default=True, help="-")

    parser.add_argument("--batch_size", type=int, default=1, help="normal noise")
    parser.add_argument("--img_size", type=int, default=64, help="normal noise")
    parser.add_argument("--img_channels", type=int, default=1, help="normal noise")

    parser.add_argument("--noise_dim", type=int, default=100, help="normal noise")
    parser.add_argument("--label_dim", type=int, default=6, help="discrete label")
    parser.add_argument("--varying_dim", type=int, default=2, help="continuous code")

    parser.add_argument("--max_step", type=int, default=200, help="-")
    parser.add_argument("--step_size", type=float, default=0.01, help="-")

    parser.add_argument("--seed", type=int, default=2023920, help="random seed for reproducibility") 
    # amplings: 2023920
    opt = parser.parse_args()
    Trans_AP = AssignTarget(opt)
    Trans_AP.startloop()



