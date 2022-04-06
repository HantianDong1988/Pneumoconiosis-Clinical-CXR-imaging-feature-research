import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchviz import make_dot
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
import time
import pandas as pd
import shufflenet
from pytorchtools import EarlyStopping

acc_loss = [[0] * 2 for row in range(30)]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# 数据预处理tensor
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    # transforms.RandomVerticalFlip(p=0.5),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.Grayscale(),  # 灰度图
    transforms.ToTensor()  # 转换Tensor
])

# 读取数据
train = 'dataset/train'
test = 'dataset/test'
train_dataset = datasets.ImageFolder(train, transform)
test_dataset = datasets.ImageFolder(test, transform)
print(len(train_dataset), len(test_dataset))
# 导入数据
Batch_size = 8
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=Batch_size, shuffle=True)
classes = train_dataset.classes
classes_index = train_dataset.class_to_idx
print(classes)
print(classes_index)
model = models.googlenet(progress=True)
for param in model.parameters():
    param.requires_grad = True
print(model)
# 构建新的全连接层
model.fc = torch.nn.Sequential(
    torch.nn.Linear(1024, 512),
    # torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(512, 3)
)
print(model)
loss_fn = nn.BCELoss()
LR = 0.0001
# 定义代价函数
entropy_loss = nn.CrossEntropyLoss()
# 定义优化器
optimizer = optim.Adam(model.parameters(), LR)
# 训练函数
model.cuda()

Init_Epoch = 0
Freeze_Epoch = 30  # 训练次数
epoch_step = len(train_dataset) // Batch_size
epoch_step_test = len(test_dataset) // Batch_size
time1 = time.time()
for epoch in range(Init_Epoch, Freeze_Epoch):
    total_loss = 0
    total_accuracy = 0
    val_loss = 0
    valid_losses = []
    early_stopping = EarlyStopping(patience=20, verbose=True)
    model.train().cuda()
    print('Start Train')
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Freeze_Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(train_loader):
            if iteration >= epoch_step:
                break
            images, targets = batch
            inputs = Variable(images.cuda())
            targets = Variable(targets.cuda())
            optimizer.zero_grad()
            outputs, aux2, aux1 = model(inputs)
            loss_value = nn.CrossEntropyLoss()(outputs, targets)
            loss_value.backward()
            optimizer.step()

            total_loss += loss_value.item()
            with torch.no_grad():
                accuracy = torch.mean(
                    (torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
                total_accuracy += accuracy.item()

            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'accuracy': total_accuracy / (iteration + 1)})
            pbar.update(1)
        acc_loss[epoch][0] = total_accuracy / (iteration + 1)

    print('\nFinish Train')

    model.eval().cuda()
    print('Start Validation')
    with tqdm(total=epoch_step_test, desc=f'Epoch {epoch + 1}/{Freeze_Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(train_loader):
            if iteration >= epoch_step_test:
                break
            images, targets = batch
            inputs = Variable(images.cuda())
            targets = Variable(targets.cuda())
            optimizer.zero_grad()
            outputs = model(inputs)
            loss_value = nn.CrossEntropyLoss()(outputs, targets)
            val_loss += loss_value.item()
            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
            pbar.update(1)
            valid_losses.append(loss_value.item())

        valid_loss = np.average(valid_losses)
        valid_losses = []
        early_stopping(valid_loss, model)
        acc_loss[epoch][1] = loss_value.item()
    if early_stopping.early_stop:
        print("Early stopping")
        break
    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Freeze_Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_test))
    print("耗时：", time.time() - time1)
    torch.save(model.state_dict(), "googlenet.pth")
    name = ["acc", 'loss']
    test = pd.DataFrame(columns=name, data=acc_loss)  # 数据有三列，列名分别为one,two,three
    test.to_csv('csv/googlenet.csv', encoding='utf-8')

# torch.save(model.state_dict(), "shufflenet.pth")
# torch.save(model.state_dict(), "shufflenet_att.pth")
