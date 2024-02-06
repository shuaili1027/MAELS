import torch
import pyrootutils
import numpy as np
import random, time, os, sys
from backbones import CustomNet, StudentNet
sys.path.append("..")
from DataM import DataM
import torchattacks as ta
import torch.nn.functional as F
from torch.autograd import Variable


# Get the root directory of the whole project
root = pyrootutils.setup_root(search_from=__file__, indicator=["pyproject.toml"], pythonpath=True, dotenv=True)


def fix_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class EarlyStopping():

    def __init__(self, save_path, net, patience=10, verbose=False, delta=0, type=None, using_at=None, using_nd=None, teacher=True):
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.type = type
        self.net = net
        self.using_at = using_at
        self.using_nd = using_nd
        self.teacher = teacher

    def __call__(self, val_loss, model, optimizer):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model, optimizer)
        elif self.best_score - val_loss > self.delta:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model, optimizer)
            self.counter = 0
        elif self.best_score - val_loss < self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

    def save_checkpoint(self, val_loss, model, optimizer):
        if not self.using_at:
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            print("Saving STD!!")
            path = os.path.join(self.save_path, f'{self.net}-{self.type}.pth')
            torch.save(model.state_dict(), path)
            self.val_loss_min = val_loss
        else:
            if not self.using_nd:
                if self.verbose:
                    print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
                print("Saving AT!!")
                path = os.path.join(self.save_path, f'AT-{self.net}-{self.type}.pth')
                torch.save(model.state_dict(), path)
                self.val_loss_min = val_loss
            else:
                if self.verbose:
                    print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
                if self.teacher:
                    print("Saving T!!")
                    path = os.path.join(self.save_path, f'ND-T-{self.net}-{self.type}.pth')
                else:
                    print("Saving S!!")
                    path = os.path.join(self.save_path, f'ND-S-{self.net}-{self.type}.pth')                    
                torch.save(model.state_dict(), path)
                self.val_loss_min = val_loss
            


class TrainLoop():
    def __init__(self, param):
        super().__init__()
        self.param = param
        self.loader = DataM(self.param)
        self.net = self.param.net
        self.device = self.param.dev
        self.seed = self.param.seed

    def on_train_start(self):
        # print out dir
        print('Checkpoint of Output Path: {}'.format(self.param.save_path))
        print(self.param)
        print("Training is starting")
        pass
    
    def distill_step(self):
        # Ensuring reproducibility
        fix_random_seed(self.seed)

        # Introducing Teacher Model
        model_teacher = CustomNet(self.net, self.param.database)
        model_teacher = model_teacher.to(self.device)

        # Introducing Student Model
        model_student = StudentNet(self.net, self.param.database)
        model_student = model_student.to(self.device)

        # Data
        train_loader = self.loader.return_train_loader()
        val_loader = self.loader.return_val_loader()
        test_loader = self.loader.return_test_loader()

        # Loss Function
        SoftCrossEntropyLoss = torch.nn.CrossEntropyLoss()
        SoftCrossEntropyLoss = SoftCrossEntropyLoss.to(self.device)
        
        # Optimizer 
        optimizer_teacher = torch.optim.Adam(model_teacher.parameters(), lr=self.param.lr, betas=(self.param.b1, self.param.b2))
        optimizer_student = torch.optim.Adam(model_student.parameters(), lr=self.param.lr, betas=(self.param.b1, self.param.b2))

        # # Logging
        for epoch in range(self.param.epochs):
            all_time = 0
            if (epoch != 0):
                model_teacher.load_state_dict(
                    torch.load(os.path.join(
                        root,
                        self.param.save_path,
                        f"ND-T-{self.param.database}-{self.net}.pth")))
            
            model_teacher.train()
            
            start_time = time.time()
            for i, (data, labels) in enumerate(train_loader):
                data = data.to(self.device)
                labels = labels.to(self.device)

                # Forward path of teacher net
                scores = model_teacher(data)
                
                loss = SoftCrossEntropyLoss((scores / self.param.temp), labels)

                # Backward pass and optimization
                optimizer_teacher.zero_grad()
                loss.backward()
                optimizer_teacher.step()

            print('Checking Teacher Accuracy On Training Data')
            train_acc, train_loss = self.check_accuracy(train_loader, model_teacher, SoftCrossEntropyLoss)

            print('Checking Teacher Accuracy On Val Data')
            val_acc, val_loss = self.check_accuracy(val_loader, model_teacher, SoftCrossEntropyLoss)

            if epoch == (self.param.epochs - 1):
                print('Checking Teacher Accuracy On Test Data')
                test_acc, test_loss = self.check_accuracy(test_loader, model_teacher, SoftCrossEntropyLoss)

            end_time = time.time()
            all_time = (end_time - start_time)
            print('___________________________________________epoch split___________________________________________')
            sys.stdout.write(
                "\r[Epoch %d/%d] [train loss: %f] [val loss: %f] [train accuracy: %f] [val accuracy: %f]"
                % (epoch, self.param.epochs, train_loss, val_loss, train_acc, val_acc))
            sys.stdout.flush()
            print(f'[train_time]:{round(all_time, 3)}')
            
            print("A")
            self.save_model(model=model_teacher, optimizer=optimizer_teacher, val_loss=val_loss, save_path=self.param.save_path,
                            patience=5, verbose=False, delta=0.5, type0=self.net, using_nd=True, teacher=True)
            print("B")
        
        # Logging
        for epoch in range(self.param.epochs):
            all_time = 0  
            model_teacher.load_state_dict(
                    torch.load(os.path.join(
                        root,
                        self.param.save_path,
                        f"ND-T-{self.param.database}-{self.net}.pth")))   
            if (epoch != 0):   
                model_student.load_state_dict(
                    torch.load(os.path.join(
                        root,
                        self.param.save_path,
                        f"ND-S-{self.param.database}-{self.net}.pth")))

            model_student.train()
            model_teacher.eval()
            
            start_time = time.time()
            for i, (data, _) in enumerate(train_loader):
                data = data.to(self.device)
                labels = F.softmax(model_teacher(data), dim=1)

                # Forward path of student net
                scores = model_student(data)
                
                loss = SoftCrossEntropyLoss((scores / self.param.temp), labels)

                # Backward pass and optimization
                optimizer_student.zero_grad()
                loss.backward()
                optimizer_student.step()

            print('Checking Student Accuracy On Training Data')
            train_acc, train_loss = self.check_accuracy(train_loader, model_student, SoftCrossEntropyLoss)

            print('Checking Student Accuracy On Val Data')
            val_acc, val_loss = self.check_accuracy(val_loader, model_student, SoftCrossEntropyLoss)

            if epoch == (self.param.epochs - 1):
                print('Checking Student Accuracy On Test Data')
                test_acc, test_loss = self.check_accuracy(test_loader, model_student, SoftCrossEntropyLoss)

            end_time = time.time()
            all_time = (end_time - start_time)
            print('___________________________________________epoch split___________________________________________')
            sys.stdout.write(
                "\r[Epoch %d/%d] [train loss: %f] [val loss: %f] [train accuracy: %f] [val accuracy: %f]"
                % (epoch, self.param.epochs, train_loss, val_loss, train_acc, val_acc))
            sys.stdout.flush()
            print(f'[train_time]:{round(all_time, 3)}')

            print("A")
            self.save_model(model=model_student, optimizer=optimizer_student, val_loss=val_loss, save_path=self.param.save_path,
                            patience=5, verbose=False, delta=0.5, type0=self.net, using_nd=True, teacher=False)
            print("B")

    def train_step(self):
        # Ensuring reproducibility
        fix_random_seed(self.seed)

        # Introducing Model
        model = CustomNet(self.net, self.param.database)
        model = model.to(self.device)

        # Data
        train_loader = self.loader.return_train_loader()
        val_loader = self.loader.return_val_loader()
        test_loader = self.loader.return_test_loader()

        # Loss Function
        class_loss = torch.nn.CrossEntropyLoss()
        class_loss = class_loss.to(self.device)

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.param.lr, betas=(self.param.b1, self.param.b2))

        # if Starting AT
        if self.param.using_AT:
            
            Adversary = ta.PGDL2(
                model, 
                eps=self.param.epsilon, 
                alpha= 0.1 * self.param.epsilon, 
                steps=self.param.iterations, 
                random_start=True)

        # Logging
        for epoch in range(self.param.epochs):
            all_time = 0
            if (epoch != 0) and (self.param.using_AT != None):
                model.load_state_dict(
                    torch.load(os.path.join(
                        root,
                        self.param.save_path,
                        f"AT-{self.param.database}-{self.net}.pth")))
            elif (epoch != 0) and (self.param.using_AT == None):
                model.load_state_dict(
                    torch.load(os.path.join(
                        root,
                        self.param.save_path,
                        f"{self.param.database}-{self.net}.pth")))

            model.train()
            start_time = time.time()
            for i, (data, labels) in enumerate(train_loader):
                data = data.to(self.device)
                labels = labels.to(self.device)

                if not self.param.using_AT:
                    # Forward path
                    scores = model(data)
                    loss = class_loss(scores, labels)
                else:
                    adv_images = Adversary(((data+1)/2).clamp_(0, 1), labels)
                    # Forward pass
                    scores0 = model((adv_images * 2 - 1))
                    scores1 = model(data)
                    loss = (class_loss(scores0, labels) + class_loss(scores1, labels)) / 2.

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print('Checking Accuracy On Training Data')
            train_acc, train_loss = self.check_accuracy(train_loader, model, class_loss)

            print('Checking Accuracy On Val Data')
            val_acc, val_loss = self.check_accuracy(val_loader, model, class_loss)

            if epoch == (self.param.epochs - 1):
                print('Checking Accuracy On Test Data')
                test_acc, test_loss = self.check_accuracy(test_loader, model, class_loss)

            end_time = time.time()
            all_time = (end_time - start_time)
            print('___________________________________________epoch split___________________________________________')
            sys.stdout.write(
                "\r[Epoch %d/%d] [train loss: %f] [val loss: %f] [train accuracy: %f] [val accuracy: %f]"
                % (epoch, self.param.epochs, train_loss, val_loss, train_acc, val_acc))
            sys.stdout.flush()
            print(f'[train_time]:{round(all_time, 3)}')

            self.save_model(model=model, optimizer=optimizer, val_loss=val_loss, save_path=self.param.save_path,
                            patience=5, verbose=False, delta=0.5, type0=self.net, using_nd=None, teacher=None)

    def on_train_end(self):
        print("Training has finished")
        pass

    def check_accuracy(self, loader, model, class_loss, device="cuda"):
        num_correct = 0
        num_samples = 0
        total_loss = 0
        model.eval()

        with torch.no_grad():
            for data, targets in loader:
                data = data.to(device)
                targets = targets.to(device)

                scores = model(data)
                loss = class_loss(scores, targets)
                score_label = torch.argmax(scores, dim=1)
                num_correct += (score_label == targets).sum().item()
                total_loss += loss.item()
                num_samples += score_label.size(0)

        accuracy = float(num_correct) / float(num_samples) * 100
        loss = float(total_loss) / float(num_samples) * 100
        print(f'Got {num_correct}/{num_samples} with accuracy {accuracy:.4f}')
        print(f'Got {total_loss}/{num_samples} with loss {loss:.4f}')
        return accuracy, loss

    def save_model(self, model, optimizer, val_loss, save_path, patience, verbose, delta, type0, using_nd, teacher):
        save_setting = EarlyStopping(save_path, net=self.param.database, patience=patience, verbose=verbose, delta=delta, type=type0, using_at=self.param.using_AT, using_nd=using_nd, teacher =teacher)
        save_setting.save_checkpoint(val_loss, model, optimizer)
        pass

    def train(self):
        self.on_train_start()
        if self.param.using_ND:
            self.distill_step()
        else:
            self.train_step()
        self.on_train_end()
        pass
    pass
