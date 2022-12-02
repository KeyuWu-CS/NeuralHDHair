import torch.nn as nn
import torch

from Tools.utils import get_spatial_points,sample,position_encoding
from torch.nn import init
from Models.base_block import ConvBlock,ResnetBlock
class Decoder(nn.Module):
    def __init__(self,in_cha,max_cha,num_out,cond=False):
        super().__init__()
        cha=max_cha
        if cond:
            num_p=9
        else:
            num_p=3
        self.layer1=ConvBlock(in_cha+num_p,cha,1,1,norm='none',activation='lrelu')
        self.layer2=ConvBlock(cha+num_p,cha//2,1,1,norm='none',activation='lrelu')
        self.layer3=ConvBlock(cha//2+num_p,cha//4,1,1,norm='none',activation='lrelu')
        self.layer4=ConvBlock(cha//4+num_p,num_out,1,1,norm='none',activation='none')

    def forward(self, x,p):

        x=self.layer1(x)

        x=self.layer2(torch.cat([x,p],dim=1))
        x=self.layer3(torch.cat([x,p],dim=1))

        x=self.layer4(torch.cat([x,p],dim=1))

        # x = self.layer2(x)
        #
        # x = self.layer3(x)
        # x = self.layer4(x)

        return x



class HairSpatDecoder(nn.Module):
    def __init__(self,min_cha,max_cha,out_cha,n_layer,latnet_d,no_use_depth=True):
        super().__init__()


        self.voxel_models=nn.ModuleList()
        self.out_models=nn.ModuleList()
        self.conv3d_transpose=nn.ModuleList()
        self.out_cha=out_cha
        C=min(max_cha,min_cha*(2**n_layer))
        mid_cha=min(max_cha,min_cha*(2**n_layer-1))
        D = latnet_d

        self.first_to_voxel = To_Voxel(C+1,mid_cha,D)
        self.first_out_layer=nn.Conv3d(mid_cha,out_cha,kernel_size=1,bias=False)


        self.n_layer = n_layer-1


        for i in reversed(range(self.n_layer)):
            D *= 2
            C = mid_cha

            mid_cha=min(max_cha,min_cha*(2**(i)))
            self.conv3d_transpose.append(nn.Sequential(nn.ConvTranspose3d(C, mid_cha, kernel_size=4, stride=2,padding=1),nn.InstanceNorm3d(min_cha),nn.LeakyReLU()))
            if i==0 and not no_use_depth:
                self.voxel_models.append(To_Voxel(2 * mid_cha + 1+1, mid_cha, depth=D))
            # if i == 0:
            #     self.voxel_models.append(To_Voxel(2 * mid_cha + 20, mid_cha, depth=D))
            else:
                self.voxel_models.append(To_Voxel(2*mid_cha+1,mid_cha,depth=D))
            # self.out_models.append(nn.Conv3d(mid_cha,out_cha,kernel_size=3,padding=1,bias=False))
            self.out_models.append(nn.Conv3d(mid_cha,out_cha,kernel_size=1,bias=False))





    def forward(self, caches,points,sample=True,depth=None):
        x,_=self.first_to_voxel(caches[-1],None,points,False)
        # v=self.first_out_layer(x)
        for i in range(self.n_layer):
            up=self.conv3d_transpose[i](x)
            x,phi = self.voxel_models[i](caches[-2 - i], up, points, last_layer=(i == self.n_layer - 1 and sample is True),depth=depth)

            # v=sample(v,2)
            # v+=self.out_models[i](x)
        v=self.out_models[i](x)
        return v[:,:,0,0,:],phi[:,:,0,0,:]





class To_Voxel(nn.Module):
    def __init__(self,in_cha,out_cha,depth):
        super().__init__()

        self.depth=depth
        # self.refine1=nn.Sequential(nn.Conv3d(in_cha,out_cha,1),nn.LeakyReLU())
        # self.refine2=nn.Sequential(nn.Conv3d(out_cha,out_cha,1),nn.LeakyReLU())


        self.refine1 = nn.Sequential(nn.Conv3d(in_cha, 4*in_cha, 1), nn.LeakyReLU())
        self.refine2 = nn.Sequential(nn.Conv3d(4*in_cha, 2*in_cha, 1), nn.LeakyReLU())
        self.refine3 = nn.Sequential(nn.Conv3d(2*in_cha, in_cha, 1), nn.LeakyReLU())
        self.refine4 = nn.Sequential(nn.Conv3d(in_cha, out_cha, 1), nn.LeakyReLU())

    def to_Voxel_with_sdf(self,x,up=None,last_layer=False):
        D=self.depth
        H,W=x.size()[-2:]
        B=x.size(0)
        # p=get_spatial_points(B,D,H,W).cuda()
        p=get_spatial_points(B,D,H,W).cuda()[:,2:3,...]
        # if last_layer:
        #     p=position_encoding(p)
        if len(x.size())<5:
            x=torch.unsqueeze(x,2)
        x=x.repeat(1,1,D,1,1)
        x=torch.cat([x,up],1) if up is not None else x
        x=torch.cat([x,p],1)

        return x

    def forward(self, x,up,points,last_layer=False,depth=None):
        feat=self.to_Voxel_with_sdf(x,up,last_layer)
        if last_layer:

            points=(points*2)-1

            points=points[...,[1,0,2]]
            points=torch.unsqueeze(points,dim=1)
            points=torch.unsqueeze(points,dim=1)

            x=torch.nn.functional.grid_sample(feat,points , mode='bilinear')
            if depth is not None:
                depth=torch.unsqueeze(depth,dim=2)
                depth=torch.unsqueeze(depth,dim=2)
                x=torch.cat([x,depth],dim=1)
        else:
            x=feat


        x=self.refine1(x)
        phi = x.clone()
        x=self.refine2(x)
        x=self.refine3(x)
        x=self.refine4(x)

        return x,phi


class UnetDecoder(nn.Module):
    def __init__(self,min_cha,max_cha,out_cha,n_layers):
        super().__init__()

        self.conv_transposed = nn.ModuleList()

        # C=min(max_cha,min_cha*(2**n_layers))
        C=min(max_cha,min_cha*(2**(n_layers-1)))
        mid_cha=min(max_cha,min_cha*(2**(n_layers-2)))
        self.conv_transposed.append(nn.ConvTranspose2d(C,mid_cha,4,2,1))

        self.n_layers = n_layers-1
        for i in reversed(range(n_layers-1)):
            C=mid_cha
            mid_cha=int(min(max_cha,min_cha*(2**(i-1))))

            self.conv_transposed.append(nn.ConvTranspose2d(C*2,max(mid_cha,min_cha),4,2,1))
        self.out=nn.Conv2d(max(mid_cha,min_cha),out_cha,3,1,1)

    def forward(self, caches,ori_amb):
        x=caches[-1]
        up=self.conv_transposed[0](x)
        for i in range(self.n_layers):
            x=caches[-2-i]
            x=torch.cat([up,x],dim=1)
            up=self.conv_transposed[i+1](x)
        # up=torch.cat([up,ori_amb],dim=1)
        out=self.out(up)
        return out


class UnetDecoder2(nn.Module):
    def __init__(self,min_cha,max_cha,out_cha,n_layers):
        super().__init__()

        self.conv_transposed = nn.ModuleList()

        # C=min(max_cha,min_cha*(2**n_layers))
        C=min(max_cha,min_cha*(2**(n_layers-1)))
        mid_cha=min(max_cha,min_cha*(2**(n_layers-2)))
        self.conv_transposed.append(nn.ConvTranspose2d(C,mid_cha,4,2,1))

        self.n_layers = n_layers-1
        for i in reversed(range(n_layers-1)):
            C=mid_cha
            mid_cha=int(min(max_cha,min_cha*(2**(i-1))))

            self.conv_transposed.append(nn.Sequential(nn.ConvTranspose2d(C*2,max(mid_cha,min_cha),4,2,1),
                                                      ResnetBlock(max(mid_cha,min_cha), padding_type='zero',
                                                                  activation=nn.LeakyReLU(True),
                                                                  norm_layer=nn.InstanceNorm2d)
                                                      ))
        self.out=nn.Conv2d(max(mid_cha,min_cha),out_cha,3,1,1)

    def forward(self, caches,ori_amb):
        x=caches[-1]
        up=self.conv_transposed[0](x)
        for i in range(self.n_layers):
            x=caches[-2-i]
            x=torch.cat([up,x],dim=1)
            up=self.conv_transposed[i+1](x)
        # up=torch.cat([up,ori_amb],dim=1)
        out=self.out(up)
        return out

class MutilLevelDecoder(nn.Module):
    def __init__(self,min_cha,max_cha,out_cha,n_layers):
        super().__init__()

        self.conv_transposed = nn.ModuleList()

        # C=min(max_cha,min_cha*(2**n_layers))
        C=min(max_cha,min_cha*(2**(n_layers-1)))
        mid_cha=min(max_cha,min_cha*(2**(n_layers-2)))
        self.conv_transposed.append(nn.ConvTranspose2d(C,mid_cha,4,2,1))

        self.n_layers = n_layers-1
        for i in reversed(range(n_layers-1)):
            C=mid_cha
            mid_cha=int(min(max_cha,min_cha*(2**(i-1))))

            self.conv_transposed.append(nn.ConvTranspose2d(C*2,max(mid_cha,min_cha),4,2,1))
        self.out=nn.Conv2d(max(mid_cha,min_cha),out_cha,3,1,1)

    def forward(self, caches,ori_amb):
        x=caches[-1]
        up=self.conv_transposed[0](x)
        for i in range(self.n_layers):
            x=caches[-2-i]
            x=torch.cat([up,x],dim=1)
            up=self.conv_transposed[i+1](x)
        # up=torch.cat([up,ori_amb],dim=1)
        out=self.out(up)
        return out





