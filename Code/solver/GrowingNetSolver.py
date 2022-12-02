from dataload import data_loader
from Tools.utils import *
from solver.base_solver import BaseSolver
from Models.GrowingNet import GrowingNet
import os
import torch
import time
import numpy as np
import torch.nn

class GrowingNetSolver(BaseSolver):

    @staticmethod
    def modify_options(parser):
        """Add new options and rewrite default values for existing options"""
        parser.set_defaults(save_root='checkpoints/GrowingNet')
        parser.set_defaults(data_mode='strand')
        parser=GrowingNet.modify_options(parser)

        return parser

    def initialize(self,  opt):
        super().initialize( opt)


        self.opt = opt
        self.local_size = opt.local_size
        self.stride = opt.stride

        self.pt_num = opt.pt_per_strand
        self.sd_num = opt.sd_per_batch
        self.initialize_networks(opt)
        # input = torch.rand((3757, 3, 16, 16, 16)).cuda()
        # for i in range(10):
        #     _ = self.model_on_one_gpu.OriEncoder(input)
        # pos = torch.rand((1, 3, 12 * 12, 1)).cuda()
        # feat = torch.rand((1, 128, 12 * 12, 1)).cuda()
        # input = torch.cat([feat, pos], dim=1)
        # # input=feat
        # if opt.warm_gpu:
        #     for i in range(10):
        #         start=time.time()
        #         _ = self.model_on_one_gpu.Decoder_pos(input, pos)
        #         _ = self.model_on_one_gpu.Decoder_pos_Inv(input, pos)
        #         _ = self.model_on_one_gpu.Decoder_label(input, pos)
        #         print('test:',time.time()-start)
        #     torch.cuda.synchronize()

        if self.opt.isTrain:
            self.optimizer_GN=self.create_optimizers(opt)
            self.criteria=torch.nn.CrossEntropyLoss()
            self.L1loss=torch.nn.L1Loss()

    def create_optimizers(self,opt):
        GrowingNet_params=[]
        GrowingNet_params+=list(self.GrowingNet.parameters())
        optimizer_GN=torch.optim.Adam(GrowingNet_params,lr=self.learning_rate,betas=(0.9,0.999))

        return optimizer_GN

    def preprocess_input(self,datas):
        strands = datas['strands'].type(torch.float)
        gt_orientation = datas['gt_ori'].type(torch.float)
        labels = datas['labels']
        if labels is not None:
            labels=labels.type(torch.float)


        if len(self.opt.gpu_ids)>0:
            strands = strands.cuda()
            gt_orientation = gt_orientation.cuda()
            if labels is not None:
                labels = labels.cuda()
                labels = labels.permute(0, 3, 1, 2)
        gt_orientation = gt_orientation.permute(0, 4, 1, 2, 3)
        strands = strands.permute(0, 3, 1, 2)

        return strands,gt_orientation,labels





    def initialize_networks(self, opt):
        self.GrowingNet= GrowingNet(opt,[self.depth, self.height, self.width], self.local_size, self.stride)
        self.GrowingNet.print_network()
        if len(opt.gpu_ids)>0:
            assert (torch.cuda.is_available())
            # self.GrowingNet.cuda()
            # self.model=torch.nn.DataParallel(self.GrowingNet,self.opt.gpu_ids)
            self.model=self.GrowingNet.cuda()
            # self.model_on_one_gpu=self.model.module.cuda()
            # self.GrowingNet=self.GrowingNet.module
        else:self.model=self.GrowingNet
        if opt.continue_train or opt.isTrain is False:
            path= os.path.join(opt.current_path,opt.save_root, opt.check_name,'checkpoint')
            if os.path.exists(path):
                self.GrowingNet=self.load_network(self.GrowingNet,'GrowingNet',opt.which_iter,opt)
        else:
            print(" Training from Scratch! ")
            self.GrowingNet.init_weights(opt.init_type,opt.init_variance)

    def train(self,iter_counter,dataloader,visualizer):

        for epoch in iter_counter.training_epochs():
            iter_counter.record_epoch_start(epoch)
            for i,datas in enumerate(dataloader):
                self.init_losses()
                iter_counter.record_one_iteration()

                # iter_counter.record_one_iteration()
                strands,gt_orientation,labels=self.preprocess_input(datas)

                in_points = strands[...,+1:-1]




                sta_points = strands[..., (2 * self.pt_num // 3) // 2:(2 * self.pt_num // 3) // 2 + 1]
                pre_points = strands[..., :-2]
                aft_points = strands[..., +2:]
                # print(pre_points[0,:,0,:])
                # print(aft_points[0,:,0,:])
                pre_labels = labels[..., +1:-1]
                # sum=torch.sum(pre_labels,dim=-1,keepdim=True).type(torch.long)
                # in_points[sum]=in_points[sum-1]
                # in_points[sum+1]=in_points[sum-1]
                # print(in_points.size())
                weight_labels=labels[...,+2:]
                weight_labels_Inv=labels[...,+1:-1]


                # if self.opt.Bidirectional_growth:
                #     out_points_1, labels_1, out_points_1_Inv, labels_1_Inv = self.model(in_points,gt_orientation,'nn')
                # else:
                #     out_points_1, labels_1 = self.model(in_points,gt_orientation,'nn')
                k=1
                if epoch>40 and self.opt.condition is not False:
                    k=3
                if epoch>100 and self.opt.condition is not False:
                    k=5
                if k>=2:
                    self.total_loss['rnn_p_loss']=0
                    self.total_loss['rnn_p_loss_Inv']=0
                if self.opt.Bidirectional_growth:
                    all_out_points,_,all_out_points_inv_,_=self.model(in_points,gt_orientation,k,'rnn')
                self.total_loss['p_loss']=self.L1loss(aft_points*weight_labels,all_out_points[...,0:self.pt_num-2]*weight_labels)
                self.total_loss['p_loss_Inv']=self.L1loss(pre_points*weight_labels_Inv,all_out_points_inv_[...,0:self.pt_num-2]*weight_labels_Inv)
                for i in range(1,k):
                    self.total_loss['rnn_p_loss']+=self.L1loss(aft_points[...,i:]*weight_labels[...,i:],all_out_points[...,i*(self.pt_num-2):(i+1)*(self.pt_num-2)-i]*weight_labels[...,i:])
                    self.total_loss['rnn_p_loss_Inv']+=self.L1loss(pre_points[...,:-i]*weight_labels_Inv[...,i:],all_out_points_inv_[...,i*(self.pt_num-2)+i:(i+1)*(self.pt_num-2)]*weight_labels_Inv[...,i:])

                # print('pred',out_points_1[0,:,0,0:5])
                # print('gt', aft_points[0,:, 0, 0: 5])
                # print('back:',out_points_1_Inv[0,:,0,:5])
                # print('gt:',pre_points[0,:,0,:5])


                # print('gt:',aft_points[0,:,0,0:3])
                # print('out:',out_points_1[0,:,0,0:3])
                drap_gt_points = in_points * pre_labels
                pre_labels = pre_labels.type(torch.long)[:,0,...]


                ####losss
                # self.total_loss['p_loss']=torch.mean(torch.abs(aft_points-out_points_1)*weight_labels)
                # self.total_loss['p_loss']=self.L1loss(aft_points*weight_labels,out_points_1*weight_labels)

                # if self.opt.pred_label:
                #     self.total_loss['classify_loss'] = torch.mean(self.criteria(labels_1,pre_labels)) * 0.1
                # if self.opt.Bidirectional_growth:
                #     self.total_loss['p_loss_Inv']=torch.mean(torch.abs(pre_points-out_points_1_Inv)*weight_labels_Inv)
                    # self.total_loss['p_loss_Inv']=self.L1loss(pre_points*weight_labels_Inv,out_points_1_Inv*weight_labels_Inv)

                #####backward
                self.loss_backward(self.total_loss,self.optimizer_GN)

                if iter_counter.needs_printing():
                    losses = self.get_latest_losses()
                    visualizer.print_current_errors(epoch, iter_counter.epoch_iter,losses, iter_counter.time_per_iter)
                    # print(labels_1[0,:,0,:])

                if iter_counter.needs_displaying():
                    with torch.no_grad():
                        if self.opt.Bidirectional_growth:
                            out_points_2, labels_2, out_points_2_Inv, labels_2_Inv =self.model(sta_points,gt_orientation,self.pt_num//2,'rnn')
                        else:
                            out_points_2, labels_2 = self.model(sta_points,gt_orientation,self.pt_num//2,'rnn')

                    if self.opt.pred_label:
                        drap_points = out_points_2 * torch.where(labels_2[:, 1:2, ...] > labels_2[:, 0:1, ...],torch.ones_like(labels_2[:, 0:1, ...]),torch.zeros_like(labels_2[:, 0:1, ...]))
                        if self.opt.Bidirectional_growth:
                            drap_points_Inv = out_points_2_Inv * torch.where(labels_2_Inv[:, 1:2] > labels_2_Inv[:, 0:1],torch.ones_like(labels_2_Inv[:, 0:1]),torch.zeros_like(labels_2_Inv[:, 0:1]))
                    else:
                        drap_points = out_points_2
                        if self.opt.Bidirectional_growth:
                            drap_points_Inv = out_points_2_Inv

                    N = out_points_2.size(1)
                    s = np.random.randint(0, N)
                    drap_gt_points=drap_gt_points.permute(0,2,3,1)
                    drap_points=drap_points.permute(0,2,3,1)
                    drap_points=drap_points.cpu().detach().numpy()
                    drap_gt_points=drap_gt_points.cpu().detach().numpy()
                    visualizer.draw_samples(drap_gt_points[0], drap_points[0], s,self.width,self.height,self.depth, "Forward")
                    if self.opt.Bidirectional_growth:
                        drap_points_Inv = drap_points_Inv.permute(0, 2, 3, 1)
                        drap_points_Inv = drap_points_Inv.cpu().detach().numpy()
                        visualizer.draw_samples(drap_gt_points[0], drap_points_Inv[0], s,self.width,self.height,self.depth, "Backward")


                if iter_counter.needs_saving():
                    print('saving the latest model (epoch %d, total_steps %d)' %
                          (epoch, iter_counter.total_steps_so_far))
                    self.save_network(self.GrowingNet, 'GrowingNet', iter_counter.total_steps_so_far, self.opt)
                    self.save_network(self.GrowingNet, 'GrowingNet', 'latest', self.opt)
                    iter_counter.record_current_iter()
        # self.update_learning_rate(epoch)
        iter_counter.record_epoch_end()


    def test(self,dataloader):
        if self.opt.Bidirectional_growth:
            datas=dataloader.generate_random_root()
        else:
            datas=dataloader.generate_test_data(self.opt.growInv)
        strands, gt_orientation,labels = self.preprocess_input(datas)





        with torch.no_grad():
            for i in range(2):

                if self.opt.Bidirectional_growth:
                    print('begin.....')
                    start = time.time()
                    # wcenters,wlatents=self.model_on_one_gpu.encoder(gt_orientation)
                    # wlatents=wlatents.expand(len(self.opt.gpu_ids),*wlatents.size()[1:])
                    # print(wlatents.size())
                    # wcenters=wcenters.expand(len(self.opt.gpu_ids),*wcenters.size()[1:])
                    # print('encoder cost:',time.time()-start)
                    # gt_orientation = close_voxel(gt_orientation, 5)
                    gt_orientation=gt_orientation.expand(len(self.opt.gpu_ids),*gt_orientation.size()[1:])
                    out_points_2, labels_2, out_points_2_Inv, labels_2_Inv=self.model(strands,gt_orientation,self.pt_num-1,'rnn')
                    # torch.cuda.synchronize()
                    print('grow cost:', time.time() - start)

                else:
                    out_points_2, labels_2 = self.model(strands,gt_orientation,self.pt_num-1,'rnn')


        print(out_points_2.size())
        gt_orientation=gt_orientation.permute(0,2,3,4,1)
        out_points_2=out_points_2.permute(0,2,3,1)
        labels_2=labels_2.permute(0,2,3,1)

        out_points_2 = out_points_2.reshape(1, -1,  self.pt_num-1,3)
        labels_2 = labels_2.reshape(1, -1,  self.pt_num-1,2)

        gt_orientation=gt_orientation.cpu().numpy()
        out_points_2=out_points_2.cpu().numpy()
        labels_2=labels_2.cpu().numpy()
        if self.opt.Bidirectional_growth:

            out_points_2_Inv = out_points_2_Inv.permute(0, 2, 3, 1)
            labels_2_Inv = labels_2_Inv.permute(0, 2, 3, 1)
            out_points_2_Inv=out_points_2_Inv.reshape(1,-1,self.pt_num-1,3)
            labels_2_Inv=labels_2_Inv.reshape(1,-1,self.pt_num-1,2)
            out_points_2_Inv=out_points_2_Inv.cpu().numpy()
            labels_2_Inv=labels_2_Inv.cpu().numpy()


        mask = np.linalg.norm(gt_orientation, axis=-1)

        strand_delete_by_ori, segment = delete_point_out_ori(mask, out_points_2)
        strand_delete_by_label, segment_label = delete_point_out_label(out_points_2, labels_2)
        if self.opt.Bidirectional_growth:
            strand_delete_by_ori_Inv, segment_Inv = delete_point_out_ori(mask, out_points_2_Inv)
            strand_delete_by_label_Inv, segment_label_Inv = delete_point_out_label(out_points_2_Inv, labels_2_Inv)
        else:
            segment_Inv = None
            segment_label_Inv = None
            strand_delete_by_ori_Inv = None
            strand_delete_by_label_Inv = None

        final_strand_del_by_ori, final_segment = concat_strands(strand_delete_by_ori, strand_delete_by_ori_Inv, segment,
                                                                segment_Inv, self.opt.Bidirectional_growth,gt_orientation.shape[2]//96)
        final_strand_del_by_label, final_segment_label = concat_strands(strand_delete_by_label,
                                                                        strand_delete_by_label_Inv, segment_label,
                                                                        segment_label_Inv,
                                                                        self.opt.Bidirectional_growth)
        write_strand(final_strand_del_by_ori, self.opt, final_segment, 'ori')
        # write_strand(final_strand_del_by_label, self.opt, final_segment_label, 'label')





    def loss_backward(self, losses, optimizer):
        optimizer.zero_grad()
        loss=sum(losses.values()).mean()
        loss.backward()
        optimizer.step()

    def init_losses(self):
        self.total_loss={}

    def get_latest_losses(self):
        return self.total_loss

    def update_learning_rate(self, epoch):
        if epoch%70==0 and epoch!=0:
            self.learning_rate=self.learning_rate//2

        for param_group in self.optimizer_GN:
            param_group['lr']=self.learning_rate















