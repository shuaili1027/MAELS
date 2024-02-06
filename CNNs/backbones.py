import torch, torchvision
from torch import nn

# VGG-16, ResNet-18, MobileNetV2, ResNeXt50

class CustomNet(torch.nn.Module):
    def __init__(self, net, database):
        super().__init__()
        self.net = net
        self.database = database

        if self.database == "mnist":
            if self.net == "vgg16":  ############# ok
                self.model = torchvision.models.vgg16(pretrained=True)
                self.model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
                self.model.classifier[6] = nn.Linear(4096, 10)

            elif self.net == "resnet18":  ############# ok
                self.model = torchvision.models.resnet18(pretrained=True)
                self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                num_features = self.model.fc.in_features
                self.model.fc = torch.nn.Linear(num_features, 10)

            elif self.net == "mobilenetv2":  ############# ok
                self.model = torchvision.models.mobilenet_v2(pretrained=True)
                self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                num_features = self.model.classifier[1].in_features
                self.model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(num_features, 10))

            elif self.net == "resnext50":  ############# ok
                self.model = torchvision.models.resnext50_32x4d(pretrained=True)
                self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.model.fc = nn.Linear(2048, 10)

        elif self.database == "surface":
            if self.net == "vgg16":  ############# ok
                self.model = torchvision.models.vgg16(pretrained=True)
                self.model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
                self.model.classifier[6] = nn.Linear(4096, 6)
            elif self.net == "resnet18":  ############# ok
                self.model = torchvision.models.resnet18(pretrained=True)
                self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                num_features = self.model.fc.in_features
                self.model.fc = torch.nn.Linear(num_features, 6)
            elif self.net == "mobilenetv2":  ############# ok
                self.model = torchvision.models.mobilenet_v2(pretrained=True)
                self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                num_features = self.model.classifier[1].in_features
                self.model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(num_features, 6))
            elif self.net == "resnext50":  ############# ok
                self.model = torchvision.models.resnext50_32x4d(pretrained=True)
                self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.model.fc = nn.Linear(2048, 6)
        
        elif self.database == "cifar10":
            if self.net == "vgg16":  ############# ok
                self.model = torchvision.models.vgg16(pretrained=True)
                self.model.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
                self.model.classifier[6] = nn.Linear(4096, 10)

            elif self.net == "resnet18":  ############# ok
                self.model = torchvision.models.resnet18(pretrained=True)
                self.model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
                num_features = self.model.fc.in_features
                self.model.fc = torch.nn.Linear(num_features, 10)

            elif self.net == "mobilenetv2":  ############# ok
                self.model = torchvision.models.mobilenet_v2(pretrained=True)
                self.model.features[0][0] = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                num_features = self.model.classifier[1].in_features
                self.model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(num_features, 10))

            elif self.net == "resnext50":  ############# ok
                self.model = torchvision.models.resnext50_32x4d(pretrained=True)
                self.model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.model.fc = nn.Linear(2048, 10)

        elif self.database == "mini-imagenet":
            if self.net == "vgg16":  ############# ok
                self.model = torchvision.models.vgg16(pretrained=True)
                num_ftrs = self.model.classifier[6].in_features
                self.model.classifier[6] = nn.Linear(num_ftrs, 100) 

            elif self.net == "resnet18":  ############# ok
                self.model = torchvision.models.resnet18(pretrained=True)
                self.model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
                num_features = self.model.fc.in_features
                self.model.fc = torch.nn.Linear(num_features, 100)

            elif self.net == "mobilenetv2":  ############# ok
                self.model = torchvision.models.mobilenet_v2(pretrained=True)
                self.model.features[0][0] = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                num_features = self.model.classifier[1].in_features
                self.model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(num_features, 100))

            elif self.net == "resnext50":  ############# ok
                self.model = torchvision.models.resnext50_32x4d(pretrained=True)
                self.model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.model.fc = nn.Linear(2048, 100)

    def forward(self, x):
        out = self.model(x)
        return out
    

class StudentNet(torch.nn.Module):
    def __init__(self, net, database):
        super().__init__()
        self.net = net
        self.database = database
        
        if self.database == "mnist":
            if self.net == "vgg16":  ############# ok
                self.model = torchvision.models.vgg16(pretrained=False)
                self.model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
                self.model.classifier[6] = nn.Linear(4096, 10)

            elif self.net == "resnet18":  ############# ok
                self.model = torchvision.models.resnet18(pretrained=False)
                self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                num_features = self.model.fc.in_features
                self.model.fc = torch.nn.Linear(num_features, 10)

            elif self.net == "mobilenetv2":  ############# ok
                self.model = torchvision.models.mobilenet_v2(pretrained=False)
                self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                num_features = self.model.classifier[1].in_features
                self.model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(num_features, 10))

            elif self.net == "resnext50":  ############# ok
                self.model = torchvision.models.resnext50_32x4d(pretrained=False)
                self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.model.fc = nn.Linear(2048, 10)

        elif self.database == "surface":
            if self.net == "vgg16":  ############# ok
                self.model = torchvision.models.vgg16(pretrained=False)
                self.model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
                self.model.classifier[6] = nn.Linear(4096, 6)
            elif self.net == "resnet18":  ############# ok
                self.model = torchvision.models.resnet18(pretrained=False)
                self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                num_features = self.model.fc.in_features
                self.model.fc = torch.nn.Linear(num_features, 6)
            elif self.net == "mobilenetv2":  ############# ok
                self.model = torchvision.models.mobilenet_v2(pretrained=False)
                self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                num_features = self.model.classifier[1].in_features
                self.model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(num_features, 6))
            elif self.net == "resnext50":  ############# ok
                self.model = torchvision.models.resnext50_32x4d(pretrained=False)
                self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.model.fc = nn.Linear(2048, 6)
        
        elif self.database == "cifar10":
            if self.net == "vgg16":  ############# ok
                self.model = torchvision.models.vgg16(pretrained=False)
                self.model.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
                self.model.classifier[6] = nn.Linear(4096, 10)

            elif self.net == "resnet18":  ############# ok
                self.model = torchvision.models.resnet18(pretrained=False)
                self.model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
                num_features = self.model.fc.in_features
                self.model.fc = torch.nn.Linear(num_features, 10)

            elif self.net == "mobilenetv2":  ############# ok
                self.model = torchvision.models.mobilenet_v2(pretrained=False)
                self.model.features[0][0] = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                num_features = self.model.classifier[1].in_features
                self.model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(num_features, 10))

            elif self.net == "resnext50":  ############# ok
                self.model = torchvision.models.resnext50_32x4d(pretrained=False)
                self.model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.model.fc = nn.Linear(2048, 10)

        elif self.database == "mini-imagenet":
            if self.net == "vgg16":  ############# ok
                self.model = torchvision.models.vgg16(pretrained=False)
                num_ftrs = self.model.classifier[6].in_features
                self.model.classifier[6] = nn.Linear(num_ftrs, 100) 

            elif self.net == "resnet18":  ############# ok
                self.model = torchvision.models.resnet18(pretrained=False)
                num_ftrs = self.model.fc.in_features
                self.model.fc = nn.Linear(num_ftrs, 100)

            elif self.net == "mobilenetv2":  ############# ok
                self.model = torchvision.models.mobilenet_v2(pretrained=False)
                num_ftrs = self.model.classifier[1].in_features
                self.model.classifier[1] = nn.Linear(num_ftrs, 100)

            elif self.net == "resnext50":  ############# ok
                self.model = torchvision.models.resnext50_32x4d(pretrained=False)
                num_ftrs = self.model.fc.in_features
                self.model.fc = nn.Linear(num_ftrs, 100)

    def forward(self, x):
        out = self.model(x)
        return out