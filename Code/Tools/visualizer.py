import os
import time
import numpy as np
import cv2
import random
import torch
from torchvision.utils import save_image
import imageio
class Visualizer():
    def __init__(self, opt):
        self.opt = opt

        if opt.isTrain:

            save_path = os.path.join(opt.current_path, opt.save_root)
            self.log_name = os.path.join(save_path, opt.name, 'logs','loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

    def print_current_errors(self, epoch, i, errors, t, curr_size=None):
        tt = time.asctime(time.localtime(time.time()))
        if curr_size is not None:
            message = '(epoch: %d, iters: %d, time: %.3f, size: %d) ' % (
                epoch, i, t, curr_size)
        else:
            message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        message = str(tt) + message
        for k, v in errors.items():
            # print(v)
            # if v != 0:
            v = v.mean().float()
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)





    def draw_samples(self, strands_in, strands_ot, s,width,height,depth,name="sss"):

        s_in = strands_in[s]
        s_ot = strands_ot[s]

        bbmin = np.array([0, 0, 0], dtype=np.float32)
        bbmax = np.array([width-1, height-1, depth-1], dtype=np.float32)

        s_in = np.maximum(np.minimum(np.round(s_in), bbmax), bbmin).astype(np.int32)
        s_ot = np.maximum(np.minimum(np.round(s_ot), bbmax), bbmin).astype(np.int32)

        img1 = np.zeros(shape=[height, width], dtype=np.uint8)
        img2 = np.zeros(shape=[depth, height], dtype=np.uint8)

        img1[s_in[:, 1], s_in[:, 0]] = 120
        img1[s_ot[:, 1], s_ot[:, 0]] = 250

        img2[s_in[:, 2], s_in[:, 1]] = 120
        img2[s_ot[:, 2], s_ot[:, 1]] = 250

        cv2.imwrite(f"{name}_1.jpg", img1)
        cv2.imwrite(f"{name}_2.jpg", img2)

    def draw_ori(self,image,gt_ori,out_ori,ori_occ,suffix,mul=1):
        sliceId = random.randint(24*mul,72*mul)
        gt_ori=(gt_ori+1)/2
        out_ori=(out_ori+1)/2
        ori_occ=(ori_occ+1)/2
        # image=torch.cat([image,torch.zeros_like(image)[:,0:1]],dim=1)
        if image.size()[1]==1:
            save_image(image,'image{}.png'.format(suffix))
        else:
            save_image(torch.cat([image,torch.zeros_like(image)],dim=1)[:,0:3,...],'image{}.png'.format(suffix))
        save_image(gt_ori[:,:,sliceId,...],'gt_ori{}.png'.format(suffix))
        save_image(out_ori[:,:,sliceId,...],'out_ori{}.png'.format(suffix))
        save_image(ori_occ[:,:,sliceId,...],'occ_ori{}.png'.format(suffix))

    # def draw_2D_data(self,ori_amb,weight,pred_weight):
    #     shape=ori_amb.size(2)
    #     save_image(torch.cat([(ori_amb+1)/2,torch.zeros(ori_amb.size(0),1,shape,shape).cuda()],dim=1),'Ori_amb.png')
    #     ori_amb=ori_amb.permute(0,2,3,1)
    #     weight=weight.permute(0,2,3,1)
    #     pred_weight=pred_weight.permute(0,2,3,1)
    #     gt=ori_amb*weight
    #     pred=ori_amb*pred_weight
    #     gt=gt.permute(0,3,1,2)
    #     pred=pred.permute(0,3,1,2)
    #     save_image(torch.cat([(gt+1)/2,torch.zeros(ori_amb.size(0),1,shape,shape).cuda()],dim=1),'Ori_gt.png')
    #     save_image(torch.cat([(pred+1)/2,torch.zeros(ori_amb.size(0),1,shape,shape).cuda()],dim=1),'Ori_pred.png')
    #

    def draw_2D_data(self,input,ori_amb,ori_gt,pred_ori):
        shape=ori_amb.size(2)
        save_image(torch.cat([(ori_amb+1)/2,torch.zeros(ori_amb.size(0),1,shape,shape).cuda()],dim=1),'Ori_amb.png')
        save_image(torch.cat([(pred_ori+1)/2,torch.zeros(ori_amb.size(0),1,shape,shape).cuda()],dim=1),'pred_ori.png')
        save_image(torch.cat([(ori_gt+1)/2,torch.zeros(ori_amb.size(0),1,shape,shape).cuda()],dim=1),'gt_ori.png')
        save_image(input,'input.png')

        pred_weight=torch.cosine_similarity(pred_ori,ori_amb,dim=1)[...,None]
        weight=torch.cosine_similarity(ori_gt,ori_amb,dim=1)[...,None]
        pred_weight[pred_weight<0]=-1
        pred_weight[pred_weight>0]=1
        weight[weight<0]=-1
        weight[weight>0]=1
        ori_amb=ori_amb.permute(0,2,3,1)
        gt=ori_amb*weight
        pred=ori_amb*pred_weight
        gt=gt.permute(0,3,1,2)
        pred=pred.permute(0,3,1,2)

        pred=torch.cat([(pred+1)/2,torch.zeros(ori_amb.size(0),1,shape,shape).cuda()],dim=1)
        gt=torch.cat([(gt+1)/2,torch.zeros(ori_amb.size(0),1,shape,shape).cuda()],dim=1)
        save_image(torch.cat([pred,gt],dim=0),'results.png')
        # save_image(torch.cat([(gt+1)/2,torch.zeros(ori_amb.size(0),1,shape,shape).cuda()],dim=1),'Ori_gt.png')
        # save_image(torch.cat([(pred+1)/2,torch.zeros(ori_amb.size(0),1,shape,shape).cuda()],dim=1),'Ori_pred.png')


    def draw_2D_data1(self,gt,depth,pred_depth):
        save_image(gt,'input.png')
        # print(depth.size())
        save_image(depth,'gt_depth.png')
        save_image(pred_depth,'pred_depth.png')
