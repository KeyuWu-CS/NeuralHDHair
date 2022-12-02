import torch.nn as nn
import torch
from Models.base_block import ConvBlock,ResnetBlock
class OriEncoder(nn.Module):
    def __init__(self,n_layers,min_cha,max_cha):
        super().__init__()
        self.models=[]
        cha=min_cha
        self.models.append(nn.Conv3d(3, cha, 3, 2))
        self.models.append(nn.LeakyReLU())
        for _ in range(n_layers):
            self.models.append(nn.Conv3d(cha, min(2 * cha, max_cha), 3, 2,1))
            cha = min(2 * cha, max_cha)
            self.models.append(nn.LeakyReLU())
        self.models.append(nn.Conv3d(cha,min(2 * cha, max_cha),1))

        self.models=nn.Sequential(*self.models)

    def forward(self,x):

        return self.models(x)



class UnetEncoder(nn.Module):
    def __init__(self,n_layers,in_cha,min_cha,max_cha,kernel_size=3,stride=2,padding=1,norm='none',activation='none'):
        super().__init__()
        self.down1=ConvBlock(in_cha,min_cha,kernel_size=kernel_size,stride=stride,padding=padding,norm='in',activation=activation)
        cha=min_cha
        self.models=nn.ModuleList()
        for i in range(n_layers-1):
            mid_cha=min(2*cha,max_cha)
            self.models.append(ConvBlock(cha,mid_cha,kernel_size=kernel_size,stride=stride,padding=padding,norm='in',activation=activation))
            cha=mid_cha


    def forward(self,x):
        assert (len(x.size())==4)
        caches=[]

        x=self.down1(x)
        caches.append(x)
        for model in self.models:
            x=model(x)
            caches.append(x)
        return caches



class UnetEncoder2(nn.Module):
    def __init__(self,n_layers,in_cha,min_cha,max_cha,kernel_size=3,stride=2,padding=1,norm='none',activation='none'):
        super().__init__()
        self.down1=ConvBlock(in_cha,min_cha,kernel_size=kernel_size,stride=stride,padding=padding,norm='in',activation=activation)
        cha=min_cha
        self.models=nn.ModuleList()
        for i in range(n_layers-1):
            mid_cha=min(2*cha,max_cha)
            self.models.append(nn.Sequential( ConvBlock(cha,mid_cha,kernel_size=kernel_size,stride=stride,padding=padding,norm='in',activation=activation),
                                              ResnetBlock(mid_cha,padding_type='zero',activation=nn.LeakyReLU(True),norm_layer=nn.InstanceNorm2d)))
            cha=mid_cha


    def forward(self,x):
        assert (len(x.size())==4)
        caches=[]

        x=self.down1(x)
        caches.append(x)
        for model in self.models:
            x=model(x)
            caches.append(x)
        return caches


class UnetEncoder1(nn.Module):
    def __init__(self, n_layers, in_cha, min_cha, max_cha, kernel_size=3, stride=2, padding=1, norm='none',
                 activation='none'):
        super().__init__()
        self.inconv1 = ConvBlock(in_cha, min_cha, kernel_size=7, stride=1, padding=3, norm='in',
                               activation=activation)
        self.inconv2=ResnetBlock(min_cha,padding_type='zero',activation=nn.LeakyReLU(True),norm_layer=nn.InstanceNorm2d)
        self.inconv3 = ResnetBlock(min_cha,padding_type='zero',activation=nn.LeakyReLU(True),norm_layer=nn.InstanceNorm2d)
        self.down1= ConvBlock(min_cha, min_cha, kernel_size=kernel_size, stride=stride, padding=padding, norm='in',
                               activation=activation)
        cha = min_cha
        self.models = nn.ModuleList()
        for i in range(n_layers - 1):
            mid_cha = min(2 * cha, max_cha)
            self.models.append(
                ConvBlock(cha, mid_cha, kernel_size=kernel_size, stride=stride, padding=padding, norm='in',
                          activation=activation))
            cha = mid_cha

    def forward(self, x):
        assert (len(x.size()) == 4)
        caches = []
        x=self.inconv1(x)
        x=self.inconv2(x)
        x=self.inconv3(x)
        x = self.down1(x)
        caches.append(x)
        for model in self.models:
            x = model(x)
            caches.append(x)
        return caches


class Autoencoder(nn.Module):
    def __init__(self,n_layers,in_cha,min_cha,max_cha,kernel_size=3,stride=2,padding=1,norm='in',activation='none'):
        super().__init__()
        self.inconv=ConvBlock(in_cha,min_cha,kernel_size=7,stride=1,padding=3,norm=norm,activation=activation)
        cha = min_cha
        self.encoder=nn.ModuleList()
        self.n_layers=n_layers
        for i in range(n_layers):
            mid_cha=min(2*cha,max_cha)
            self.encoder.append(nn.Sequential(ConvBlock(cha,mid_cha,kernel_size=kernel_size,stride=stride,padding=padding,norm=norm,activation=activation),
                                              ResnetBlock(mid_cha,padding_type='zero',norm=norm,activation='lrelu')
                                              )
                                )
            cha=mid_cha
        self.decoder=nn.ModuleList()
        for i in range(n_layers - 2):
            cha=mid_cha
            mid_cha=min(min_cha*(2**(n_layers-1-i)),max_cha)
            if i ==0:
                self.decoder.append(nn.Sequential(nn.ConvTranspose2d(cha,mid_cha,4,2,1),
                                                  ResnetBlock(mid_cha,padding_type='zero',norm='none',activation='lrelu')
                                                  )
                                    )
            else:
                self.decoder.append(nn.ConvTranspose2d(cha*2,max(mid_cha,min_cha),4,2,1))


    def forward(self, x):
        x=self.inconv(x)
        caches=[]
        for i in range(self.n_layers):
            x=self.encoder[i](x)
            caches.append(x)
        x = caches[-1]
        for i in range(self.n_layers-2):
            up=self.decoder[i](x)
            x=caches[-2-i]
            x=torch.cat([up,x],dim=1)
        return x


class GlobalEncoder(nn.Module):
    def __init__(self,input_nc,output_nc,ngf,n_downsample=3,n_blocks=9):
        super().__init__()
        model=[nn.ReflectionPad2d(3),nn.Conv2d(input_nc,ngf,kernel_size=7,padding=0),nn.InstanceNorm2d(ngf),nn.LeakyReLU()]

        ### downsample
        for i in range(n_downsample):
            mult=2**i
            model+=[ConvBlock(ngf*mult,ngf*mult*2,kernel_size=3,stride=2,padding=1,norm='in',activation='lrelu')]

        #### resnet block
        mult=2**n_downsample
        for i in range(n_blocks):
            model+=[ResnetBlock(ngf*mult,padding_type='reflect',activation=nn.LeakyReLU(True),norm_layer=nn.InstanceNorm2d)]

        ###upsample
        for i in range(n_downsample):
            mult=2**(n_downsample-i)
            model+=[nn.ConvTranspose2d(ngf*mult,int(ngf*mult/2),kernel_size=3,stride=2,padding=1,output_padding=1),nn.InstanceNorm2d(int(ngf*mult/2)),nn.LeakyReLU()]

        model+=[nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        self.model=nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# class LocalEncoder(nn.Module):
#     def __init__(self):
#         super().__init__(input_nc,output_nc,ngf)










