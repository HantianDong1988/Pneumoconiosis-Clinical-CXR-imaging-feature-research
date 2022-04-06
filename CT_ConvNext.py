from ConvNext import convnext_tiny
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn.functional as F
from torchsummary import summary
import time
import shufflenet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# 数据预处理tensor
transform = transforms.Compose([
    transforms.Resize((299, 299)),
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
Batch_size = 4
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=Batch_size, shuffle=True)
classes = train_dataset.classes
classes_index = train_dataset.class_to_idx
print(classes)
print(classes_index)
model = convnext_tiny(num_classes=3)
print(model)

loss_fn = nn.BCELoss()
LR = 0.0003
# 定义代价函数
entropy_loss = nn.CrossEntropyLoss()
# 定义优化器
optimizer = optim.Adam(model.parameters(), LR)
# 训练函数
model.cuda()

Init_Epoch = 0
Freeze_Epoch = 15  # 训练次数
epoch_step = len(train_dataset) // Batch_size
epoch_step_test = len(test_dataset) // Batch_size
time1 = time.time()
for epoch in range(Init_Epoch, Freeze_Epoch):
    total_loss = 0
    total_accuracy = 0
    val_loss = 0
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
            outputs = model(inputs)
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
    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Freeze_Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_test))
    print("耗时：", time.time() - time1)
torch.save(model.state_dict(), "ConvNext.pth")
