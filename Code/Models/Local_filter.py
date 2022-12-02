from Models.BaseNetwork import BaseNetwork
from Models.HGFilter import HGFilter
from Models.base_block import Conv_MLP
from Models.normalization import pixel_norm
from Loss.loss import l1_loss
import torch
class Local_Filter(BaseNetwork):

    @staticmethod
    def modify_options(parser):
        parser.add_argument('--num_stack', type=int, default=2)
        parser.add_argument('--hg_depth', type=int, default=4)
        parser.add_argument('--hg_down', type=str, default='avg_pool')
        parser.add_argument('--mlp_channels_Occ', type=list, default=[292, 512,  512,256,256, 128, 1])
        parser.add_argument('--mlp_channels_Ori', type=list, default=[292, 512,  512,256,256, 128, 3])
        # parser.add_argument('--mlp_ori_channels',type=list,default=[257,1024,512,256,128,3])
        parser.add_argument('--mlp_norm', default=None)
        parser.add_argument('--mlp_res_layers', type=list, default=[2, 3,4])
        parser.add_argument('--hg_dim', type=int, default=32)
        parser.add_argument('--hg_norm', type=str, default='group')
        parser.add_argument('--no_use_pretrain',action='store_true')
        parser.add_argument('--pretrain_path',default='HairSpatNet/2021-10-06')
        parser.add_argument('--strand_size',type=int,default=512)
        parser.add_argument('--info_mode',type=str,default='L')
        parser.add_argument('--resolution',type=list,default=[96*2,128*2,128*2])
        parser.set_defaults(use_HD=True)

        return parser


    def __init__(self,opt):
        super().__init__()
        self.num_stack=opt.num_stack
        self.hg_depth=opt.hg_depth
        self.in_cha=1
        if opt.info_mode=='amb':
            self.in_cha=2
        self.hg_dim=opt.hg_dim
        self.hg_norm=opt.hg_norm
        self.hg_down=opt.hg_down
        channels_Ori=opt.mlp_channels_Ori
        channels_Occ=opt.mlp_channels_Occ
        res_layers=opt.mlp_res_layers
        mlp_norm=opt.mlp_norm
        self.image_filter = HGFilter(self.num_stack, self.hg_depth, self.in_cha, self.hg_dim, self.hg_norm, 'no_down',False)
        self.Conv_MLP_Occ = Conv_MLP(channels=channels_Occ, merge_layer=-1, res_layers=res_layers, norm=mlp_norm)
        self.Conv_MLP_Ori = Conv_MLP(channels=channels_Ori, merge_layer=-1, res_layers=res_layers, norm=mlp_norm)


    def index(self,feat,uv):

        '''
        :param feat: [B, C, H, W] image features
        :param uv: [B, N, 2] normalized image coordinates ranged in [-1, 1]
        :return: [B, C, N] sampled pixel values
        '''
        uv=uv.unsqueeze(2)
        samples=torch.nn.functional.grid_sample(feat, uv, mode='bilinear')
        return samples[...,0]


    def query(self,points,z_feat_ori,z_feat_occ):
        xy = points[:, :, [1, 0]]
        im_feat=self.im_feat_list[-1]
        # im_feat=torch.zeros_like(im_feat)
        xy=(xy-0.5)*2

        sp_feat = points[:, :, 2:3]
        sp_feat = sp_feat.permute(0, 2, 1)
        # point_local_feat_ori = [self.index(im_feat, xy),z_feat_ori, sp_feat]
        point_local_feat_ori = [self.index(im_feat, xy),z_feat_ori]
        point_local_feat_ori = torch.cat(point_local_feat_ori, 1)
        self.pred_ori,_ = self.Conv_MLP_Ori(point_local_feat_ori)
        self.pred_ori = pixel_norm(self.pred_ori)

        # point_local_feat_occ=[self.index(im_feat, xy),z_feat_occ, sp_feat]
        point_local_feat_occ=[self.index(im_feat, xy),z_feat_occ]
        point_local_feat_occ=torch.cat(point_local_feat_occ,1)
        self.pred_occ,_ = self.Conv_MLP_Occ(point_local_feat_occ)




    def forward(self, image,strand2D,gt_occ,gt_ori,net_global,resolution=[96*4,128*4,128*4]):
        self.loss_global={}
        self.loss_local={}
        D,H,W=resolution
        self.out_ori = torch.zeros(1, 3, D, H, W).cuda()
        self.out_occ = torch.zeros(1, 1, D, H, W).cuda()
        with torch.no_grad():
            out_ori_low,out_occ_low,self.loss_global['loss_ori_low'],self.loss_global['loss_occ_low']=net_global(image,gt_occ,gt_ori,mode='generator')
        points=net_global.points
        feat_ori,feat_occ=net_global.get_phi()
        self.gt_ori=net_global.gt_ori
        self.gt_occ=net_global.gt_occ
        self.loss_weight=net_global.loss_weight


        # print(strand2D.size())
        self.im_feat_list,_ = self.image_filter(strand2D)
        self.query(points,feat_ori.detach(),feat_occ.detach())

        ori,occ=self.get_pred()

        self.loss_local['loss_ori_hd'] = l1_loss((self.gt_ori - ori * self.gt_occ)*self.loss_weight) / max(torch.sum(self.loss_weight), 1.0)
        self.loss_local['loss_occ_hd'] = l1_loss((self.gt_occ - occ) * self.loss_weight) / max(torch.sum(self.loss_weight), 1.0)

        self.point_convert_to_voxel(points, ori, mode='ori')
        self.point_convert_to_voxel(points, occ, mode='occ')
        return self.out_ori,self.out_occ,out_ori_low,out_occ_low,self.loss_local,self.loss_global
        # return out_ori_low,self.out_occ,out_ori_low,out_occ_low,self.loss_local,self.loss_global


    def test(self,image,strand2D,Ori2D,net_global,resolution,step=100000):
        D, H, W = resolution
        self.out_ori = torch.zeros(1, 3, D, H, W).cuda()
        self.out_occ = torch.zeros(1, 1, D, H, W).cuda()
        with torch.no_grad():
            out_ori_low, out_occ_low=net_global.test(image,Ori2D,resolution,step=step)
        feat_ori, feat_occ = net_global.get_phi()


        points=net_global.test_points

        self.im_feat_list,_=self.image_filter(strand2D)
        n=points.size(1)//step+1
        for i in range(n):
            start=step*i
            end=min(step*(i+1),points.size(1))
            self.query(points[:,start:end],feat_ori[...,start:end].detach(),feat_occ[...,start:end].detach())
            ori, occ = self.get_pred()
            self.point_convert_to_voxel(points[:,start:end], ori, mode='ori')
            self.point_convert_to_voxel(points[:,start:end], occ, mode='occ')

        return self.out_ori,self.out_occ,out_ori_low,out_occ_low
        # return out_ori_low,self.out_occ,out_ori_low,out_occ_low

    def point_convert_to_voxel(self, points, res, mode):
        D, H, W = self.out_ori.size()[2:]
        index = points * torch.tensor([H - 1., W - 1., D - 1.]).cuda()
        # index = points
        index = torch.round(index)
        index = index.type(torch.long)

        x, y, z = torch.chunk(index, 3, -1)
        x = torch.squeeze(x)
        y = torch.squeeze(y)
        z = torch.squeeze(z)
        if mode == 'ori':
            self.out_ori[:, :, z, x, y] = res
        elif mode == 'occ':
            self.out_occ[:, :, z, x, y] = res
    def get_pred(self):
        return self.pred_ori,self.pred_occ