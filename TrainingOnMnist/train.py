import argparse
import torch
import os, sys
sys.path.append("..")
from trainloops import trainloop

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=device, help="using device: cuda or cpu, and GPU is recommended!")
    parser.add_argument("--num_workers", type=int, default=2, help="-")
    parser.add_argument("--pin_memory", type=bool, default=True, help="-")
    parser.add_argument("--log_dir", type=str, default="Logging", help="-")

    parser.add_argument("--database", type=str, default="mnist", help="using database")
    parser.add_argument("--dataset", type=str, default="mnist", help="using database")
    parser.add_argument("--file", type=str, default="Mnist", help="using database")
    parser.add_argument("--target_num", type=int, default=1, help="selecting the number of target models involved in training")
    parser.add_argument("--stage", type=str, default="stage1", help="selecting the stage1, stage21, or stage22")
    parser.add_argument("--resume_path", type=str, default="-", help="selecting the stage1, stage21, or stage22")
    parser.add_argument("--stage1_path", type=str, default="-", help="selecting the stage1, stage21, or stage22")

    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--img_size", type=int, default=64, help="batch size")
    parser.add_argument("--img_channels", type=int, default=1, help="batch size")
    parser.add_argument("--noise_dim", type=int, default=32, help="normal noise")
    parser.add_argument("--label_dim", type=int, default=10, help="discrete label")
    parser.add_argument("--varying_dim", type=int, default=2, help="continuous code")

    parser.add_argument("--g_lr", type=float, default=0.0002, help="-")
    parser.add_argument("--d_lr", type=float, default=0.0002, help="-")
    parser.add_argument("--c_lr", type=float, default=0.0002, help="-")
    parser.add_argument("--info_lr", type=float, default=0.0002, help="-")
    parser.add_argument("--en_lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--beta1", type=float, default=0.5, help="adam")
    parser.add_argument("--beta2", type=float, default=0.999, help="adam")

    parser.add_argument("--max_epoch", type=int, default=200, help="-")
    parser.add_argument("--showing_num", type=int, default=5, help="-")
    parser.add_argument("--c_split", type=int, default=5, help="-")
    parser.add_argument("--seed", type=int, default=2023920, help="random seed for reproducibility")
    parser.add_argument("--save_fig_name_stage1", type=str, default="-", help="-")
    parser.add_argument("--save_fig_name_stage2", type=str, default="-", help="-")
    
    opt = parser.parse_args()
    Train = trainloop(opt)
    Train.train_epoch()
