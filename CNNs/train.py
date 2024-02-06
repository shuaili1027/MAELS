import argparse
import torch, os, sys
from backbones import CustomNet
from trainloops import TrainLoop
sys.path.append("..")
from DataM import DataM
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument("--dev", type=str, default=device, help="using device: cuda or cpu !")
parser.add_argument("--database", type=str, default="-", help="-")
parser.add_argument("--dataset", type=str, default="-", help="-")
parser.add_argument("--num_workers", type=int, default=4, help="size of each image dimension")
parser.add_argument("--pin_memory", type=bool, default=True, help="-")
parser.add_argument("--net", type=str, default="resnet18", help="network used to classifiction")
parser.add_argument("--seed", type=int, default=202394, help="-")
parser.add_argument("--save_path", type=str, default=" ", help="-")

parser.add_argument("--epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")

parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="-")
parser.add_argument("--b2", type=float, default=0.999, help="-")

parser.add_argument("--using_AT", type=bool, default=True, help="-")
parser.add_argument("--epsilon", type=float, default=3.0, help="-")
parser.add_argument("--iterations", type=int, default=12, help="-")

parser.add_argument("--using_ND", type=bool, default=None, help="-")
parser.add_argument("--temp", type=float, default=100, help="-")
opt = parser.parse_args()
Train = TrainLoop(param=opt)
# Training
Train.train()

# Testing
class_loss = torch.nn.CrossEntropyLoss()
class_loss = class_loss.to("cuda")
acc = 0
total = 5
for k in range(total):
    data = DataM(opt)
    pathA = "-"
    for i in [opt.net]:
        print(opt.net)
        model = CustomNet(i, opt.dataset)
        model.load_state_dict(torch.load(pathA))
        model = model.to("cuda")
        model.eval()
        # summary(model,(3,32,32))
        acc += Train.check_accuracy(loader=data.return_test_loader(), model=model, class_loss=class_loss, device="cuda")[0]
print(f"Accuracy: {(acc / total):.4f}")


