import torch.nn.functional as F
import torch.nn as nn
import torch
from Models.HourGlass import ConvBlock,HourGlass
class HGFilter(nn.Module):
    def __init__(self, stack, depth, in_ch, last_ch, norm='batch', down_type='conv64', use_sigmoid=True):
        super(HGFilter, self).__init__()
        self.n_stack = stack
        self.use_sigmoid = use_sigmoid
        self.depth = depth
        self.last_ch = last_ch
        self.norm = norm
        self.down_type = down_type
        self.conv1=nn.Conv2d(in_ch,64,kernel_size=7,stride=2,padding=3)

        if self.norm == 'batch':
            self.bn1 = nn.BatchNorm2d(64)
        elif self.norm == 'in':
            self.bn1 = nn.InstanceNorm2d(64)
        elif self.norm == 'group':
            self.bn1 = nn.GroupNorm(32, 64)

        if self.down_type == 'conv64':
            self.conv2 = ConvBlock(64, 64, self.norm)
            self.down_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        elif self.down_type == 'conv128':
            self.conv2 = ConvBlock(128, 128, self.norm)
            self.down_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        elif self.down_type == 'avg_pool' or self.down_type == 'no_down':
            self.conv2 = ConvBlock(64, 128, self.norm)

        self.conv3 = ConvBlock(128, 128, self.norm)
        self.conv4 = ConvBlock(128, 256, self.norm)

        for stack in range(self.n_stack):
            self.add_module('m' + str(stack), HourGlass(self.depth, 256, self.norm))
            self.add_module('top_m_' + str(stack), ConvBlock(256, 256, self.norm))
            self.add_module('conv_last' + str(stack),
                            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            if self.norm == 'batch':
                self.add_module('bn_end' + str(stack), nn.BatchNorm2d(256))
            elif self.norm == 'group':
                self.add_module('bn_end' + str(stack), nn.GroupNorm(32, 256))

            self.add_module('l' + str(stack),
                            nn.Conv2d(256, last_ch,
                                      kernel_size=1, stride=1, padding=0))

            if stack < self.n_stack - 1:
                self.add_module(
                    'bl' + str(stack), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module(
                    'al' + str(stack), nn.Conv2d(last_ch, 256, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), True)

        if self.down_type == 'avg_pool':
            x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        elif self.down_type == ['conv64', 'conv128']:
            x = self.conv2(x)
            x = self.down_conv2(x)
        elif self.down_type == 'no_down':
            x = self.conv2(x)
        else:
            raise NameError('unknown downsampling type')

        normx = x

        x = self.conv3(x)
        x = self.conv4(x)

        previous = x

        outputs = []
        for i in range(self.n_stack):
            hg = self._modules['m' + str(i)](previous)

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(self._modules['bn_end' + str(i)]
                        (self._modules['conv_last' + str(i)](ll)), True)

            tmp_out = self._modules['l' + str(i)](ll)

            if self.use_sigmoid:
                outputs.append(nn.Tanh()(tmp_out))
            else:
                outputs.append(tmp_out)

            if i < self.n_stack - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

        return outputs, normx
