# -*- coding: utf-8 -*-
import torchvision
import torch
import torch.nn as nn
from PIL import Image
import PIL
import os
import torch.utils.data as data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import torchvision.transforms.functional as f
import elasticdeform
import random
import numpy as np
from torch.nn import functional as F
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import math
import torch.nn.functional as F
#config
class default_config(object):
    output_class = 2 #输出的类别数
    root_dir = "lucchi" 
    images_dir = "training" #图片位置
    labels_dir = "training_groundtruth" #标签位置
    output_dir = "outputs"
    split_size = 0.9 #分割训练和测试集
    writer_pos = "unet"
    sub_width = 188 #经过网络后图片大小的差距
    batch_size = 1
    learning_rate = 0.1
    momentum = 0.99
    target_size_width = 768
    target_size_height = 1024

opt = default_config()
writer = SummaryWriter(opt.writer_pos) 



#data
transform_img = torchvision.transforms.ToTensor()
transform_label = torchvision.transforms.ToTensor()

class image_data(Dataset):
    def __init__(self, root_dir,image_dir, label_dir):
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.image_dir)
        self.image_list = os.listdir(self.path)
        self.label_path = os.path.join(self.root_dir, self.label_dir)
        self.label_list = os.listdir(self.label_path)
        self.w0 = 10
        self.n_classes = 2

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_item_path = os.path.join(self.root_dir, self.image_dir, img_name)
        img = Image.open(img_item_path)
        img = transform_img(img)
        label_name = self.label_list[idx]
        label_item_path = os.path.join(self.root_dir, self.label_dir, label_name)
        label = Image.open(label_item_path)
        label = transform_label(label)
        img,label = trans_unet(img, label) 
        img = img.clone().detach().requires_grad_(True)
        label = label.clone().detach().requires_grad_(True)
        return img, label
    def __len__(self):
        return len(self.image_list)



all_data = image_data(opt.root_dir, opt.images_dir, opt.labels_dir)
train_len = (int)(len(all_data)*opt.split_size)
test_len = (int)(len(all_data)-train_len)
train_set, test_set = data.random_split(all_data,[train_len, test_len])
train_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, shuffle=True, num_workers=0, drop_last=False)
test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=True, num_workers=0, drop_last=False)

color_trans = torchvision.transforms.ColorJitter(brightness=0.4)

class GaussianNoise:
    """Apply Gaussian noise to tensor."""

    def __init__(self, mean=0., std=1., p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor):
        noise = 0
        if random.random() < self.p:
            noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise


class trans_image(object):
    def __init__(self,p=0.5,sigma=10):
        self.p = p
        self.sigma = sigma
        self.norm = torchvision.transforms.Normalize(mean=0,std=1)
    def cal_size(self,width, height):
        return (int)(width+opt.sub_width), (int)(height+opt.sub_width)
    def image_expanse(self,x):
        self.width = x.shape[1]
        self.height = x.shape[2]
        self.trans_size_width, self.trans_size_height = self.cal_size(self.width, self.height)
        self.start_pos_width = (int)((self.trans_size_width-self.width)/2)
        self.start_pos_height = (int)((self.trans_size_height-self.height)/2)
        self.end_pos_width = (int)(self.start_pos_width+self.width)
        self.end_pos_height = (int)(self.start_pos_height+self.height)

        trans_size = torchvision.transforms.Resize(size=(self.trans_size_width,self.trans_size_height))
        ans = trans_size(x)
        #中间填充原图
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    ans[i,(int)(self.start_pos_width+j),(int)(self.start_pos_height+k)] = x[i,j,k]
        #填充左右
        for i in range(self.start_pos_height):
            ans[:,self.start_pos_width:self.end_pos_width-1,self.start_pos_height-i-1] = ans[:,self.start_pos_width:self.end_pos_width-1,self.start_pos_height+i]
            ans[:,self.start_pos_width:self.end_pos_width-1,self.end_pos_height+i] = ans[:,self.start_pos_width:self.end_pos_width-1,self.end_pos_height-i-1]
        #填充上下
        for i in range(self.start_pos_width):
            ans[:,self.start_pos_width-1-i,:] = ans[:,self.start_pos_width+i,:]
            ans[:,self.end_pos_width+i,:] = ans[:,self.end_pos_width-1-i,:]
        return ans
    def __call__(self,img,label):
        #img_deform,label_deform = double_trans(img, label)
        label_deform = label
        img_deform = self.image_expanse(img)
        #img_deform = color_trans(img_deform)
        img_deform = self.norm(img_deform)
        #img_deform = GaussianNoise()(img_deform)
        '''
        p_now = random.randint(0,1)
        if p_now<0.25 :
            pass
        elif p_now<0.5:
            f.rotate(img_deform,90)
            f.rotate(label_deform,90)
        elif p_now<0.75:
            f.rotate(img_deform,180)
            f.rotate(label_deform,180)
        else:
            f.rotate(img_deform,270)
            f.rotate(label_deform,270)
            '''
        return img_deform,label_deform

trans_unet = trans_image()


#loss

class BCELoss_class_weighted(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight # 二分类中正负样本的权重，第一项为负类权重，第二项为正类权重

    def forward(self, input, target):
        input = torch.clamp(input,min=1e-7,max=1-1e-7)
        bce = - self.weight[1] * target * torch.log(input) - (1 - target) * self.weight[0] * torch.log(1 - input)
        return torch.mean(bce)

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
 
    def forward(self, logits, targets):
        bs = targets.size(0)
        smooth = 1
        probs = F.softmax(logits,dim=1,dtype=torch.float32)
        probs = probs.cuda()
        m1 = probs[:,1].view(bs, -1)
        m2 = targets.view(bs, -1)
        m1 = m1.cuda()
        m2 = m2.cuda()
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / bs
        target_img = targets.squeeze(1)
        bcescore = F.binary_cross_entropy_with_logits(probs[:,1].float(), target_img.float())
        print(bcescore.item())
        return score

        
loss_fn = SoftDiceLoss()
loss_fn = loss_fn.cuda()

#model
def get_std(input):
    return math.sqrt(2/input)

class double_conv(nn.Module):
    def __init__(self,in_channel, out_channel):
        super(double_conv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels = out_channel, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.norm1 = nn.BatchNorm2d(out_channel,track_running_stats=True)
        self.norm2 = nn.BatchNorm2d(out_channel,track_running_stats=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out= self.relu1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        return out

class down_conv(nn.Module):
    def __init__(self,in_channel, out_channel):
        super(down_conv, self).__init__()
        self.d_conv = double_conv(in_channel, out_channel)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self,x):
        right_out = self.d_conv(x)
        down_out = self.pool(right_out)
        return down_out, right_out
    
class up_conv(nn.Module):
    def __init__(self,in_channel, out_channel):
        super(up_conv, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel,kernel_size=2, stride=2)
        self.d_conv = double_conv(in_channel, out_channel)

    def forward(self, down_input, left_input):
        x = self.up_conv(down_input)
        self.crop = torchvision.transforms.CenterCrop((x.shape[2], x.shape[3]))
        left_input_trans = self.crop(left_input)
        x = torch.cat([x,left_input_trans],1)
        return self.d_conv(x)


class unet(nn.Module):
    def __init__(self):
        super(unet, self).__init__()
        self.down_conv1 = down_conv(1, 64)
        self.down_conv2 = down_conv(64, 128)
        self.down_conv3 = down_conv(128, 256)
        self.down_conv4 = down_conv(256,512)

        self.bottle_conv = double_conv(512,1024)

        self.up_conv4 = up_conv(1024,512)
        self.up_conv3 = up_conv(512,256)
        self.up_conv2 = up_conv(256,128)
        self.up_conv1 = up_conv(128,64)

        self.out_conv = nn.Conv2d(in_channels=64, out_channels=opt.output_class, kernel_size=1)
        self.crop = torchvision.transforms.CenterCrop([opt.target_size_width,opt.target_size_height])
        


    def forward(self, x):
        x, left_out1 = self.down_conv1(x)
        x, left_out2 = self.down_conv2(x)
        x, left_out3 = self.down_conv3(x)
        x, left_out4 = self.down_conv4(x)

        x = self.bottle_conv(x)

        x = self.up_conv4(x, left_out4)
        x = self.up_conv3(x, left_out3)
        x = self.up_conv2(x, left_out2)
        x = self.up_conv1(x, left_out1)
        x = self.out_conv(x)
        x = self.crop(x)
        return x
    

def init_model(layer):
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0, std=get_std(layer.weight.shape[2]))

model = unet()
model = model.cuda()
model.apply(init_model)
model.train(mode=True)


optimizer = torch.optim.SGD(model.parameters(),lr=opt.learning_rate, momentum= opt.momentum)


total_train_step = 0
total_test_step = 0
epoch = 2
min_loss = -1
min_train_loss = 100000000000000.0
for i in range(epoch):
    print("--------------The {} epoch-----------".format(i+1))
    for train_data in train_loader:
        imgs, targets = train_data
        img = imgs.squeeze(1)
        writer.add_image("input",img,total_train_step)
        target_img = targets.squeeze(1)
        writer.add_image("target", target_img, total_train_step)
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = model(imgs)
        output_img = torch.argmax(outputs, dim=1)
        writer.add_image("output",output_img,total_train_step)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        targets = torch.tensor((targets>0.5), dtype=torch.int64)
        print(targets.sum())
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step +=1
        writer.add_scalar("train_loss",loss.item(), total_train_step)
        print("train_step:{} train_loss: {}".format(total_train_step, loss.item()))
        if loss.item()<min_train_loss:
            torch.save(model, "test_3_19.pth")
            min_train_loss = loss.item()
            print("The model has been saved: {}".format(total_train_step))
        if total_train_step%10==0:
            torch.save(model,"test_{}".format(total_train_step))
            print("---------------model has been saved--------------")
    
    total_test_loss = 0
    with torch.no_grad():
        for test_data in test_loader:
            imgs, targets= test_data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = model(imgs)
            targets = torch.tensor((targets>0.5), dtype=torch.float32)
            loss = loss_fn(outputs, targets)
            total_test_loss+=loss.item()
            output_img = torch.argmax(outputs, dim=1)
            target_image = targets.squeeze(1)
            pixel = torch.tensor(output_img!=target_image).sum()
            pixel_error = pixel/(target_image.shape[1]*target_image.shape[2])
            print("pixel_error: {}".format(pixel_error))
    print("loss: {}".format(total_test_loss))
    writer.add_scalar("test_loss",total_test_loss, total_test_step)
    total_test_step+=1
    if (min_loss == -1)|(total_test_loss<min_loss):
        torch.save(model,"test.pth")
        min_loss = total_test_loss
    
writer.close()
'''
ssh -L 16006:127.0.0.1:6006 lijt@202.38.69.241 -p 22232
cd /data12T/andytbest_lijintong/unet_2015
tensorboard --logdir=unet
'''

