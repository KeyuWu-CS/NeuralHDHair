from Models.BaseNetwork import BaseNetwork
from Models.Encoder import OriEncoder
from Models.Decoder import Decoder
import numpy as np
import torch
import torch.nn as nn
from Tools.utils import *
import time

class GrowingNet(BaseNetwork):
    @staticmethod
    def modify_options(parser):
        """Add new options and rewrite default values for existing options"""

        parser.add_argument('--local_size', type=int, default=8, help='res for the local voxel')
        parser.add_argument('--stride', type=int, default=4, help='stride between adjacent local voxels')
        parser.add_argument('--pt_per_strand', type=int, default=72, help='# of points per strand')
        parser.add_argument('--sd_per_batch', type=int, default=1000, help='# of sampled strands per batch')
        parser.add_argument('--n_step', type=int, default=10000, help='# of every iters to lengthen the sequence')
        parser.add_argument('--n_frames_max', type=int, default=24, help='# of max frames')
        return parser



    def __init__(self, opt,voxel_size=[96, 128, 128], local_size=32, stride=16, min_cha=16, max_cha=256, sample_mode="Tri"):
        super().__init__()

        self.voxel_size = torch.tensor(voxel_size,dtype=torch.int32)
        self.local_size = local_size
        self.stride = stride
        self.sample_mode = sample_mode
        self.opt=opt


        assert self.stride % 2 == 0
        assert self.local_size % 2 == 0
        # assert self.local_size / self.stride >= 2                   # for 32 // 16, just overlapping
        assert self.voxel_size[0] % self.local_size == 0            # currently only work for even number
        assert self.voxel_size[1] % self.local_size == 0
        assert self.voxel_size[2] % self.local_size == 0
        # self.latent_size = self.voxel_size // self.stride + 1       # since we pad the input
        self.latent_size = self.voxel_size // self.stride        # since we pad the input   #modify
        if self.local_size!=self.stride:
            self.latent_size+=1

        self.n_layers = np.log2(self.local_size).astype(np.int32)   # to get the bottleneck representation
        max_cha=min(min_cha*2**self.n_layers,max_cha)
        self.min_cha = min_cha
        self.max_cha = max_cha
        print("num of layers", self.n_layers, "min_cha", self.min_cha, "max_cha", self.max_cha)

        self.OriEncoder=OriEncoder(self.n_layers-1,min_cha,max_cha)

        self.Decoder_pos=Decoder(self.max_cha,self.max_cha,3,self.opt.condition)
        self.Decoder_pos_Inv=Decoder(self.max_cha,self.max_cha,3,self.opt.condition)
        self.Decoder_label=Decoder(self.max_cha,self.max_cha,2,self.opt.condition)

        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)




    def forward(self,strands,orientation,step=None,mode='nn'):
        self.total=0
        self.total1=0
        if mode=='rnn':
            return self.rnn(strands, step,orientation)
        if mode=='nn':
            return self.nn(strands,orientation)



    def get_ori_slices(self, ori):
        '''

        :param ori:输入3D Orientation map，将其分patch，参照我发的论文
        :return:每个patch的中心点及一个局部orientation 大小为6*8*8
        '''

        centers = []
        latents = []
        B,C, D, H, W = ori.size()
        d, h, w = [self.local_size] * 3
        # ori = torch.(ori, ((0, 0), (d//2, d//2), (h//2, h//2), (w//2, w//2), (0, 0)))
        if self.opt.local_size!=self.opt.stride:    ###modify
            pad=nn.ConstantPad3d(self.local_size//2,0)
            ori=pad(ori)
            start=0
        else:
            start=self.stride//2
        for z in range(self.latent_size[0]):
            for y in range(self.latent_size[1]):
                for x in range(self.latent_size[2]):

                    beg = [z * self.stride, y * self.stride, x * self.stride]
                    end = [beg[i] + self.local_size for i in range(3)]
                    local_ori=torch.unsqueeze(ori[:,:,beg[0]:end[0],beg[1]:end[1],beg[2]:end[2]],dim=1)

                    latents.append(local_ori)
                    center= [z * self.stride+start, y * self.stride+start, x * self.stride+start]    ###modify
                    centers.append(torch.tensor(center))

        centers=torch.cat(centers,dim=0)
        centers=torch.reshape(centers,(1,*self.latent_size,3))
        centers=centers.expand(B,*self.latent_size,3)
        centers=torch.flip(centers,[-1])

        centers=centers.type(torch.float)
        centers=centers.cuda()
        # centers=centers.permute(0,4,1,2,3)
        latents=torch.cat(latents,dim=1)

        latents=torch.reshape(latents,(B,*self.latent_size,3,d,h,w))

        return centers, latents

    def encoder(self, ori):
        '''

        :param ori:3D orientation map


        :return: 每个patch的中心坐标及每个patch对应的latent code  注意与上一个函数区分，
        此处是将所有6*8*8的小patch用 self.OriEncoder 提取特征得到N*C的latentcode，N代表patch数，
        C代表每个patch被转化为C维的特征向量
        '''
        # first get local oris and corresponding global centers
        centers, local_oris = self.get_ori_slices(ori)
        latents = torch.reshape(local_oris, (-1,3, self.local_size, self.local_size, self.local_size))

        latents=self.OriEncoder(latents)
        latents=torch.reshape(latents,(-1,*self.latent_size,self.max_cha))


        return centers, latents




    def decoder(self, s, wcenters, wlatents,decoder_pos,decoder_label,mode='nn',Inv=False,cond=None,cat_self=False):

        '''

        :param s: 3D点云的坐标
        :param wcenters: 每个点坐标对应其所在patch的中心坐标
        :param wlatents: 每个点坐标对应其所在patch的latent code
        :param decoder_pos: 解码器，输入为 一个latent code 及连续3个相邻点坐标的concatenate，最好问下我，输出是下一个点坐标
        :param decoder_label: 与上述相同，输出为是否在orientation 内  暂时可以不用
        :param mode: nn代表单次训练，rnn代表迭代训练 不懂问我
        :param Inv: 暂时重要
        :param cond: 是否使用condition，即是否使用连续的前几个点作为condition，如果为false则输入仅为 该点 及该点对应的latent code的concatenate
        :param cat_self: 我也忘了，暂时按默认设置
        :return:
        '''

        B,C,D,P,N=wlatents.size()

        # print(wcenters.size())
        s=s[...,None]

        # s=s.expand(B,3,D,P,8)   #B*3*D*P*8
        s=s.expand(B,3,D,P,N)   #B*3*D*P*8  #modify

        if self.opt.condition:
            if mode=='nn':
                if Inv:
                    # print('Inv')
                    # cond1=torch.cat([s[...,2:,:],s[...,-1:,:],s[...,-1:,:]],dim=-2)
                    # cond2=torch.cat([s[...,1:-1,:],s[...,-2:,:]],dim=-2)
                    cond=torch.cat([s[...,1:,:],s[...,-1:,:]],dim=-2)
                    if cat_self:

                        s=torch.cat([wcenters,s],dim=1)
                    else:

                        s=torch.cat([cond,s],dim=1)
                    # s=torch.cat([cond1,cond2,s],dim=1)
                else:
                    # cond1=torch.cat([s[...,0:1,:],s[...,0:1,:],s[...,:-2,:]],dim=-2)
                    # cond2=torch.cat([s[...,0:2,:],s[...,1:-1,:]],dim=-2)

                    cond=torch.cat([s[...,0:1,:],s[...,:-1,:]],dim=-2)
                    if cat_self:

                        s = torch.cat([wcenters, s], dim=1)
                    else:
                        s=torch.cat([cond, s], dim=1)
            elif mode=='rnn':                                  ####modify
                if cond.size(1)!=3:
                    cond=cond[...,None].repeat(1,1,1,1,N)
                    s=torch.cat([cond[:,0:6,...],s],dim=1)
                elif cond.size(1)==6:
                    s=torch.cat([cond[:,0:3,...],cond[:,0:3,...],s])
                else:
                    # s=torch.cat([wcenters,wcenters,s],dim=1)
                    s=torch.cat([s,s,s],dim=1)





            wcenters=wcenters.repeat(1,3,1,1,1)

        r = 2. / self.local_size
        p = r * (s - wcenters)          # [-1, 1],  see paper





        wcenters = torch.reshape(wcenters, (B, wcenters.size(1),D, -1))

        wlatents = torch.reshape(wlatents, (B, C,D, -1))

        p=torch.reshape(p,(B,wcenters.size(1),D,-1))
        # wlatents=torch.zeros_like(wlatents)


        x = torch.cat([wlatents, p], 1)   ####use the same variable to save memory
        # n = torch.cat([wlatents, p], 1)   ####use the same variable to save memory


        pos=decoder_pos(x,p)

        # label=decoder_label(x,p)

        return pos / r + wcenters[:,0:3,...], pos[:,0:2,...]





    def warp_feature(self, s, centers, latents,decoder_pos,decoder_label,mode='nn',Inv=False,cond=None,cat_self=False):   ### delete step
        """
        warp feature from latents, which has the shape of B * latent_size * C
        :param s: B * N * P * 3
        :param latents: latent features B * latent_size * C
        该部分比较难理解，可以直接问我
        get_voxel_value即根据所给的点坐标s（s分解为，xyz），利用该坐标找到其所在的patch 取出其对应的patch的 中心坐标及latent code  且此处为并行操作。
        linear_sample  参考论文，分patch时有许多重复的地方，重复的地方使用三线性插值
        """


        def my_sample(NoInputHere, zz, yy, xx):


            wcenters = self.get_voxel_value(centers, zz, yy, xx)

            wlatents = self.get_voxel_value(latents, zz, yy, xx)

            wlatents = wlatents.permute(0, 4, 1, 2,3)
            wcenters = wcenters.permute(0, 4, 1, 2,3)



            out=self.decoder(s, wcenters, wlatents,decoder_pos,decoder_label,mode=mode,Inv=Inv,cond=cond,cat_self=cat_self)


            return out

            # since the first center is 0, 0, 0
        ss = (s) / self.stride           # be care that the coordinate of p is x, y, z
        if self.local_size==self.stride:
            return self.linear_sample1(None, ss, my_sample, self.sample_mode,
                                 D=self.latent_size[0], H=self.latent_size[1], W=self.latent_size[2])
        else:
            return self.linear_sample(None, ss, my_sample, self.sample_mode,
                                 D=self.latent_size[0], H=self.latent_size[1], W=self.latent_size[2])

    def nn(self, strands, ori):
        # if self.opt.condition:
        #     cond1=torch.cat([strands[...,0:1],strands[...,0:1],strands[...,:-2]],dim=-1)[:,None,...]
        #     cond2=torch.cat([strands[...,0:2],strands[...,1:-1]],dim=-1)[:,None,...]
        #     forward_strands=torch.cat([cond1,cond2,torch.unsqueeze(strands,dim=1)],dim=1)
        #     cond1=torch.cat([strands[...,2:],strands[...,-1:],strands[...,-1:]],dim=-1)[:,None,...]
        #     cond2=torch.cat([strands[...,1:-1],strands[...,-2:]],dim=-1)[:,None,...]
        #     backward_strands=torch.cat([cond1,cond2,torch.unsqueeze(strands,dim=1)],dim=1)
        # else:
        #     forward_strands=strands
        #     backward_strands=strands
        # forward_strands=forward_strands.reshape(1,9,500,70)
        # backward_strands=backward_strands.reshape(1,9,500,70)
        # print(forward_strands[0,:,0,2])
        # print(backward_strands[0,:,0,2])
        # print(strands[0,:,0,:])
        if random.random()>0.5:
            cat_self =True
        else:
            cat_self=False


        centers, latents = self.encoder(ori)
        points,labels = self.warp_feature(strands, centers, latents,self.Decoder_pos,self.Decoder_label,mode='nn',Inv=False,cat_self=cat_self)
        if self.opt.Bidirectional_growth:
            points1,labels1=self.warp_feature(strands,centers,latents,self.Decoder_pos_Inv,self.Decoder_label,mode='nn',Inv=True,cat_self=cat_self)

            return points,labels,points1,labels1
        else:
            return points,labels

    def rnn(self, starting_points, steps,ori):
        '''param steps: now is a integer, represent the num of points on each strand
        '''
        # print('ori:',ori.size())

        strands = []
        labels=[]
        strands_Inv=[]
        labels_Inv=[]
        # points = starting_points[:,-3:,...]
        # points_Inv = starting_points[:,-3:,...]
        points = starting_points
        points_Inv = starting_points

        # first cut ori into slices, with or without overlapping

        centers, latents = self.encoder(ori)
        # print(centers.size())

        # prev_point=starting_points.repeat(1,2,1,1)
        # prev_point_Inv=starting_points.repeat(1,2,1,1)
        # prev_point=None
        # prev_point_Inv=None
        prev_point = starting_points
        prev_point_Inv = starting_points
        for step in range(steps):
            # start=time.time()
            points,label = self.warp_feature(points, centers, latents,self.Decoder_pos,self.Decoder_label,mode='rnn',Inv=False,cond=prev_point)
            prev_point=torch.cat([prev_point,points],dim=1)[:,-9:,...]

            # print('warp cost:', time.time()-start)
            if self.opt.Bidirectional_growth:
                points_Inv, label_Inv = self.warp_feature(points_Inv, centers, latents,self.Decoder_pos_Inv,self.Decoder_label,mode='rnn',Inv=True,cond=prev_point_Inv)
                prev_point_Inv = torch.cat([prev_point_Inv, points_Inv], dim=1)[:, -9:, ...]

                strands_Inv.append(points_Inv)
                labels_Inv.append(label_Inv)


            strands.append(points)
            labels.append(label)

        strands = torch.cat(strands, -1)
        labels = torch.cat(labels, -1)
        if  self.opt.Bidirectional_growth:
            strands_Inv = torch.cat(strands_Inv, -1)
            labels_Inv = torch.cat(labels_Inv, -1)

            return strands,labels,strands_Inv,labels_Inv
        else:
            return strands,labels


    def get_voxel_value(self, voxel, z, y, x):
        B = z.size(0)
        b = torch.arange(0, B)
        b=b.type(torch.long)

        S = list(z.size())[1:]
        for _ in S:
            b = torch.unsqueeze(b, -1)
        b = b.expand(B,*S)

        out=voxel[b, z, y, x, :]
        return out





    def linear_sample1(self,voxel, nPos, warp_fn, sample_mode,D, H, W,cal_normal=False):
        self.starter.record()

        nPos=nPos.permute(0,2,3,1)
        x,y,z=torch.chunk(nPos,3,dim=-1)

        b, d, p,_ = x.size()
        maxZ=(D-1).type(torch.int32)
        maxY=(H-1).type(torch.int32)
        maxX=(W-1).type(torch.int32)

        z0=torch.floor(z)
        y0=torch.floor(y)
        x0=torch.floor(x)

        wz = z - z0
        wy = y - y0
        wx = x - x0
        z0=z.type(torch.long)
        y0=y.type(torch.long)
        x0=x.type(torch.long)


        z0=torch.clamp(z0,0,maxZ)
        y0=torch.clamp(y0,0,maxY)
        x0=torch.clamp(x0,0,maxX)

        z1=z0+1
        y1=y0+1
        x1=x0+1
        z1=torch.clamp(z1,0,maxZ)
        y1=torch.clamp(y1,0,maxY)
        x1=torch.clamp(x1,0,maxX)




        # total_z = torch.cat([z0, z0, z0, z0, z1, z1, z1, z1], -1)  ###B*D*P*8
        # total_y = torch.cat([y0, y0, y1, y1, y0, y0, y1, y1], -1)
        # total_x = torch.cat([x0, x1, x0, x1, x0, x1, x0, x1], -1)


        # V,L=warp_fn(voxel,total_z,total_y,total_x)


        # V=torch.reshape(V,(b,3,d,p,8))
        # L=torch.reshape(L,(b,2,d,p,8))

        # V000, V001, V010, V011, V100, V101, V110, V111 = torch.chunk(V, 8, -1)
        # L000, L001, L010, L011, L100, L101, L110, L111, = torch.chunk(L, 8,-1)

        # z0=z0[...,0]
        # y0=y0[...,0]
        # x0=x0[...,0]
        # z1=z1[...,0]
        # y1=y1[...,0]
        # x1=x1[...,0]

        # out = self.decoder(s, wcenters, wlatents, decoder_pos, decoder_label)
        V000, L000 = warp_fn(voxel, z0, y0, x0)


        # V001, L001 = warp_fn(voxel, z0, y0, x1)
        # V010, L010 = warp_fn(voxel, z0, y1, x0)
        # V011, L011 = warp_fn(voxel, z0, y1, x1)
        #
        # V100, L100 = warp_fn(voxel, z1, y0, x0)
        # V101, L101 = warp_fn(voxel, z1, y0, x1)
        # V110, L110 = warp_fn(voxel, z1, y1, x0)
        # V111, L111 = warp_fn(voxel, z1, y1, x1)
        #
        #
        #
        # wz=wz.permute(0,3,1,2)
        # wy=wy.permute(0,3,1,2)
        # wx=wx.permute(0,3,1,2)


        # VO,LO=interpolation(V000[...,0], V001[...,0], V010[...,0], V011[...,0],
        #                          V100[...,0], V101[...,0], V110[...,0], V111[...,0],
        #                          wz, wy, wx, cal_normal) \
        #         , interpolation(L000[...,0], L001[...,0], L010[...,0], L011[...,0],
        #                         L100[...,0], L101[...,0], L110[...,0], L111[...,0],
        #                         wz, wy, wx, cal_normal)
        # return V000,L000
        return  V000,L000
        # return interpolation(V000, V001, V010, V011,
        #                          V100, V101, V110, V111,
        #                          wz, wy, wx, cal_normal) \
        #         , interpolation(L000, L001, L010, L011,
        #                         L100, L101, L110, L111,
        #                         wz, wy, wx, cal_normal)

    def linear_sample(self, voxel, nPos, warp_fn, sample_mode, D, H, W, cal_normal=False):

        nPos = nPos.permute(0, 2, 3, 1)

        x, y, z = torch.chunk(nPos, 3, dim=-1)

        b, d, p, _ = x.size()
        maxZ = (D - 1).type(torch.int32)
        maxY = (H - 1).type(torch.int32)
        maxX = (W - 1).type(torch.int32)

        z0 = torch.floor(z)
        y0 = torch.floor(y)
        x0 = torch.floor(x)

        wz = z - z0
        wy = y - y0
        wx = x - x0
        z0 = z.type(torch.long)
        y0 = y.type(torch.long)
        x0 = x.type(torch.long)

        z0 = torch.clamp(z0, 0, maxZ)
        y0 = torch.clamp(y0, 0, maxY)
        x0 = torch.clamp(x0, 0, maxX)

        z1 = z0 + 1
        y1 = y0 + 1
        x1 = x0 + 1
        z1 = torch.clamp(z1, 0, maxZ)
        y1 = torch.clamp(y1, 0, maxY)
        x1 = torch.clamp(x1, 0, maxX)



        total_z = torch.cat([z0, z0, z0, z0, z1, z1, z1, z1], -1)  ###B*D*P*8
        total_y = torch.cat([y0, y0, y1, y1, y0, y0, y1, y1], -1)
        total_x = torch.cat([x0, x1, x0, x1, x0, x1, x0, x1], -1)
        V, L = warp_fn(voxel, total_z, total_y, total_x)


        V = torch.reshape(V, (b, 3, d, p, 8))
        L = torch.reshape(L, (b, 2, d, p, 8))

        V000, V001, V010, V011, V100, V101, V110, V111 = torch.chunk(V, 8, -1)
        L000, L001, L010, L011, L100, L101, L110, L111, = torch.chunk(L, 8, -1)

        wz = wz.permute(0, 3, 1, 2)
        wy = wy.permute(0, 3, 1, 2)
        wx = wx.permute(0, 3, 1, 2)

        VO, LO = interpolation(V000[..., 0], V001[..., 0], V010[..., 0], V011[..., 0],
                               V100[..., 0], V101[..., 0], V110[..., 0], V111[..., 0],
                               wz, wy, wx, cal_normal) \
            , interpolation(L000[..., 0], L001[..., 0], L010[..., 0], L011[..., 0],
                            L100[..., 0], L101[..., 0], L110[..., 0], L111[..., 0],
                            wz, wy, wx, cal_normal)


        return VO, LO































