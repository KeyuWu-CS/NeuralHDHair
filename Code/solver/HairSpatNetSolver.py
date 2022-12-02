from solver.base_solver import BaseSolver
from Models.HairSpatNet import HairSpatNet
from Loss.loss import lovasz_hinge,uniform_sample_loss,probability_sample_loss,binary_cross_entropy,compute_gradient_penalty
import os
import torch
import torch.nn
import torch.nn.functional as F
from Tools.utils import *
from Models.Discriminator import Spat_Discriminator
import torch.autograd
class HairSpatNetSolver(BaseSolver):

    @staticmethod
    def modify_options(parser):
        parser.set_defaults(save_root='checkpoints/HairSpatNet')
        parser.set_defaults(data_mode='image')
        parser=HairSpatNet.modify_options(parser)
        parser.add_argument('--close_gt',default=False)
        parser.add_argument('--use_gan',action='store_true')
        parser.add_argument('--use_gt_Ori',action='store_true')
        return parser

    def initialize(self, opt):
        super().initialize(opt)
        self.opt = opt
        self.Spat_min_cha=opt.Spat_min_cha
        self.Spat_max_cha=opt.Spat_max_cha

        self.initialize_networks(opt)

        if self.opt.isTrain:
            self.classification_weight=1.0
            self.classification_sparse_weight=0.1
            self.optimizer,self.D_optimizer=self.create_optimizers()
            self.criteria=torch.nn.CrossEntropyLoss()
            self.L1loss=torch.nn.L1Loss()
            self.L1loss_cont=torch.nn.L1Loss()


    def initialize_networks(self,opt):
        self.net=HairSpatNet(opt,in_cha=opt.input_nc,min_cha=self.Spat_min_cha,max_cha=self.Spat_max_cha)
        self.net.print_network()
        if opt.continue_train or opt.isTrain is False:
            path = os.path.join(opt.current_path, opt.save_root, opt.check_name, 'checkpoint')
            if os.path.exists(path):
                self.net = self.load_network(self.net, 'HairSpatNet', opt.which_iter, opt)
        else:
            print(" Training from Scratch! ")
            self.net.init_weights(opt.init_type, opt.init_variance)


        if opt.isTrain:
            self.Discriminator = Spat_Discriminator()
            self.Discriminator.print_network()
        if len(opt.gpu_ids) > 0:
            assert (torch.cuda.is_available())
            self.model=self.net
            self.model=self.model.cuda()
            if opt.isTrain:
                self.Discriminator=self.Discriminator.cuda()



            # self.GrowingNet.cuda()
        #     self.model = torch.nn.DataParallel(self.net, self.opt.gpu_ids)
        #     self.model_on_one_gpu = self.model.module.cuda()
        #     # self.GrowingNet=self.GrowingNet.module
        # else:
        #     self.model = self.net

    def create_optimizers(self):
        params = []
        params += list(self.model.parameters())
        optimizer = torch.optim.Adam(params, lr=self.learning_rate, betas=(0.9, 0.999))
        d_params=self.Discriminator.parameters()
        D_optimizer=torch.optim.Adam(d_params,lr=self.learning_rate*3,betas=(0.9, 0.999))

        return optimizer,D_optimizer


    def preprocess_input(self,datas):
        image = datas['image'].type(torch.float)
        gt_orientation = datas['gt_ori'].type(torch.float)
        gt_occ=datas['gt_occ']
        Ori2D = datas['Ori2D'].type(torch.float)
        depth=datas['depth'].type(torch.float)
        if self.use_gpu():
            image = image.cuda()
            gt_orientation = gt_orientation.cuda()
            gt_occ=gt_occ.cuda()
            Ori2D = Ori2D.cuda()
            depth=depth.cuda()
        # save_image(torch.cat([image,torch.zeros(1,1,256,256).cuda()],dim=1),'test_image.png')
        # save_image(depth,'test_depth.png')
        return image,gt_orientation,gt_occ,Ori2D,depth



    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0


    def train(self,iter_counter,dataloader,visualizer):
        for epoch in iter_counter.training_epochs():
            if epoch>60:
                self.opt.use_gt_Ori=False
            iter_counter.record_epoch_start(epoch)
            for i, datas in enumerate(dataloader):
                self.init_losses()
                iter_counter.record_one_iteration()
                image,gt_orientation,gt_occ,Ori2D,depth= self.preprocess_input(datas)
                if self.opt.close_gt:
                    close_gt=close_voxel(gt_occ,5)
                else:
                    close_gt=gt_occ

                if torch.sum(depth)==0:
                    depth=None
                depth=None
                out_ori, out_occ,self.G_loss['ori_loss'],self.G_loss['occ_loss'] = self.model(image,gt_occ,gt_orientation,depth_map=depth,no_use_depth=self.opt.no_use_depth)
                # out_ori, _,self.G_loss['ori_loss'],_ = self.model(image,gt_occ,gt_orientation,depth_map=depth,no_use_depth=self.opt.no_use_depth)

                if self.opt.use_gan:
                    # alpha=torch.rand(size=[self.batch_size,1,1,1,1]).cuda()

                    feature_fake=self.Discriminator(out_ori*gt_occ)
                    feature_real=self.Discriminator(gt_orientation)
                    self.D_loss['gradient_penalty'] = compute_gradient_penalty(self.Discriminator, gt_orientation.data,
                                                                               (out_ori * gt_occ).data)

                    self.G_loss['content'] = self.L1loss_cont(feature_fake[2], feature_real[2]) * 0.01
                    scores_for_fake=torch.mean(feature_fake[-1])
                    scores_for_real=torch.mean(feature_real[-1])

                    # self.G_loss['G_loss']=-scores_for_fake

                    self.D_loss['D_loss']=scores_for_fake-scores_for_real
                    self.D_loss['D_score_loss']=torch.mean(feature_real[-1]**2)*1e-3

                    # grandient=torch.autograd.grad(outputs=feature_delta[-1],inputs=delta_orientation,grad_outputs=torch.ones(feature_delta[-1].size()).cuda(),create_graph=False,retain_graph=False)[0]


                if self.opt.use_gan:
                    self.loss_backward(self.D_loss, self.D_optimizer,True)
                self.loss_backward(self.G_loss, self.optimizer,False)



                if iter_counter.needs_printing():

                    losses = self.get_latest_losses()
                    visualizer.print_current_errors(epoch, iter_counter.epoch_iter, losses, iter_counter.time_per_iter)
                if iter_counter.needs_displaying():
                    # positive=torch.sigmoid(out_occ)>0.65
                    out_occ[out_occ>=0.2]=1
                    out_occ[out_occ<0.2]=0
                    # out_occ=torch.where(positive,torch.ones_like(positive),torch.zeros_like(positive))
                    visualizer.draw_ori(image,gt_orientation,out_ori*gt_occ,out_occ*out_ori,suffix='')

                if iter_counter.needs_saving():
                    print('saving the latest model (epoch %d, total_steps %d)' %
                          (epoch, iter_counter.total_steps_so_far))
                    self.save_network(self.model, 'HairSpatNet', iter_counter.total_steps_so_far, self.opt)
                    self.save_network(self.model, 'HairSpatNet', 'latest', self.opt)
                    self.save_network(self.Discriminator, 'Discriminator', iter_counter.total_steps_so_far, self.opt)
                    self.save_network(self.Discriminator,'Discriminator','latest',self.opt)

                    iter_counter.record_current_iter()
            self.update_learning_rate(epoch)
            iter_counter.record_epoch_end()

    def test(self,dataloader):
        with torch.no_grad():
            datas = dataloader.generate_test_data()
            image, gt_orientation, gt_occ = self.preprocess_input(datas)
            out_ori, out_occ = self.model(image)

            pred_ori=out_ori*gt_occ
            pred_ori=pred_ori.permute(0,2,3,4,1)
            pred_ori=pred_ori.cpu().numpy()
            save_ori_as_mat(pred_ori,self.opt)







    def loss_backward(self, losses, optimizer,retain=False):
        optimizer.zero_grad()
        loss = sum(losses.values()).mean()
        loss.backward(retain_graph=retain)
        optimizer.step()

    def init_losses(self):
        self.total_loss = {}
        self.D_loss={}
        self.G_loss={}

    def get_latest_losses(self):
        self.total_loss={**self.D_loss,**self.G_loss}
        return self.total_loss

    def update_learning_rate(self, epoch):
        if epoch % 6 == 0 and epoch != 0:
            self.learning_rate = self.learning_rate // 2

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate

        for param_group_d in self.D_optimizer.param_groups:
            param_group_d['lr'] = self.learning_rate
