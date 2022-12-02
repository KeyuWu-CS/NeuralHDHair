import torch.nn as nn
import torch
from Models.BaseNetwork import BaseNetwork
class Spat_Discriminator(BaseNetwork):
    def __init__(self):
        super().__init__()

        self.d_conv1=nn.Sequential(nn.Conv3d(3,32,kernel_size=4,stride=2,padding=1),nn.LeakyReLU())
        self.d_conv2=nn.Sequential(nn.Conv3d(32,64,4,2,1),nn.LeakyReLU())
        self.d_conv3=nn.Sequential(nn.Conv3d(64,128,4,2,1),nn.LeakyReLU())
        self.d_conv4=nn.Sequential(nn.Conv3d(128,256,4,2,1),nn.LeakyReLU())

        self.outlayer=nn.Conv3d(256,1,3,1,bias=False)


    def forward(self, fake_or_real):

        caches=[]
        x=fake_or_real
        B = fake_or_real.size(0)

        x=self.d_conv1(x)
        caches.append(x)
        x=self.d_conv2(x)
        caches.append(x)
        x=self.d_conv3(x)
        caches.append(x)
        x=self.d_conv4(x)
        caches.append(x)
        out=self.outlayer(x)
        out=torch.reshape(out,[B,-1])
        scores=torch.mean(out,dim=-1)
        caches.append(scores)
        return caches