from solver.base_solver import BaseSolver
from Models.HairSpatNet import HairSpatNet
import torch.nn

from Tools.utils import *
import torch.autograd
from Models.Local_filter import Local_Filter
class HairModelingHDSolver(BaseSolver):

    @staticmethod
    def modify_options(parser):
        parser.set_defaults(save_root='checkpoints/HairModelingHD')
        parser.set_defaults(data_mode='image')
        parser=HairSpatNet.modify_options(parser)
        parser=Local_Filter.modify_options(parser)
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
            self.optimizer_global,self.optimizer_local=self.create_optimizers()
            self.criteria=torch.nn.CrossEntropyLoss()
            self.L1loss=torch.nn.L1Loss()
            self.L1loss_cont=torch.nn.L1Loss()


    def initialize_networks(self,opt):
        self.net_global=HairSpatNet(opt,in_cha=opt.input_nc,min_cha=self.Spat_min_cha,max_cha=self.Spat_max_cha)
        self.net_local=Local_Filter(opt)

        self.net_global.print_network()
        self.net_local.print_network()
        if not opt.no_use_pretrain:
            path=os.path.join(opt.current_path,'checkpoints', opt.pretrain_path, 'checkpoint')
            self.net_global=self.load_network(self.net_global,'HairSpatNet',opt.which_iter,opt,specify_path=path)
        else:
            self.net_global.init_weights(opt.init_type, opt.init_variance)
        if opt.continue_train or opt.isTrain is False:
            path = os.path.join(opt.current_path, opt.save_root, opt.check_name, 'checkpoint')
            if os.path.exists(path):
                self.net_global = self.load_network(self.net_global, 'HairModelingGlobal', opt.which_iter, opt)
                self.net_local = self.load_network(self.net_local, 'HairModelingLocal', opt.which_iter, opt)
        else:
            print(" Training from Scratch! ")
            self.net_local.init_weights(opt.init_type, opt.init_variance)

        if len(opt.gpu_ids) > 0:
            assert (torch.cuda.is_available())

            self.net_global=self.net_global.cuda()
            self.net_local=self.net_local.cuda()



    def create_optimizers(self):
        params_global = []
        params_global += list(self.net_global.parameters())
        optimizer_global = torch.optim.Adam(params_global, lr=self.learning_rate, betas=(0.9, 0.999))
        params_local=list(self.net_local.parameters())
        optimizer_local = torch.optim.Adam(params_local, lr=self.learning_rate, betas=(0.9, 0.999))


        return optimizer_global,optimizer_local


    def preprocess_input(self,datas):
        image = datas['image'].type(torch.float)
        gt_orientation = datas['gt_ori'].type(torch.float)
        gt_occ=datas['gt_occ']
        Ori2D = datas['Ori2D'].type(torch.float)
        add_info=datas['add_info'].type(torch.float)
        if self.use_gpu():
            image = image.cuda()
            gt_orientation = gt_orientation.cuda()
            gt_occ=gt_occ.cuda()
            Ori2D = Ori2D.cuda()
            add_info=add_info.cuda()
        return image,gt_orientation,gt_occ,Ori2D,add_info



    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0


    def train(self,iter_counter,dataloader,visualizer):
        for epoch in iter_counter.training_epochs():
            iter_counter.record_epoch_start(epoch)
            if epoch > 20:
                self.opt.use_gt_Ori = False
            for i, datas in enumerate(dataloader):
                self.init_losses()
                iter_counter.record_one_iteration()

                image,gt_orientation,gt_occ,Ori2D,add_info= self.preprocess_input(datas)
                # save_image(add_info,'test.png')
                unsample = torch.nn.Upsample(scale_factor=self.opt.resolution[0]//96, mode='trilinear')
                gt_occ_low=gt_occ.clone()
                gt_orientation = unsample(gt_orientation)
                gt_occ=unsample(gt_occ)


                out_ori_hd, out_occ_hd, out_ori_low, out_occ_low, self.loss_local,self.loss_global=self.net_local(image,add_info, gt_occ, gt_orientation, self.net_global, resolution=self.opt.resolution)

                # self.loss_backward(self.loss_global,self.optimizer_global)
                # self.loss_backward(self.loss_global,self.optimizer_local)
                self.loss_backward(self.loss_local,self.optimizer_local)



                if iter_counter.needs_printing():

                    losses = self.get_latest_losses()
                    visualizer.print_current_errors(epoch, iter_counter.epoch_iter, losses, iter_counter.time_per_iter)
                if iter_counter.needs_displaying():
                    # positive=torch.sigmoid(out_occ)>0.65
                    out_occ_hd[out_occ_hd>=0.2]=1
                    out_occ_hd[out_occ_hd<0.2]=0
                    out_occ_low[out_occ_low>=0.2]=1
                    out_occ_low[out_occ_low<0.2]=0
                    # out_occ=torch.where(positive,torch.ones_like(positive),torch.zeros_like(positive))
                    visualizer.draw_ori(add_info,gt_orientation,out_ori_hd*gt_occ,out_occ_hd*out_ori_hd,'_hd_L',mul=2)
                    visualizer.draw_ori(image,gt_orientation,out_ori_low*gt_occ,out_occ_low*out_ori_low,'_low_L',mul=1)

                if iter_counter.needs_saving():
                    print('saving the latest model (epoch %d, total_steps %d)' %
                          (epoch, iter_counter.total_steps_so_far))
                    self.save_network(self.net_local, 'HairModelingLocal', iter_counter.total_steps_so_far, self.opt)
                    self.save_network(self.net_local, 'HairModelingLocal', 'latest', self.opt)
                    self.save_network(self.net_global, 'HairModelingGlobal', iter_counter.total_steps_so_far, self.opt)
                    self.save_network(self.net_global, 'HairModelingGlobal', 'latest', self.opt)

                    iter_counter.record_current_iter()
            self.update_learning_rate(epoch)
            iter_counter.record_epoch_end()

    def test(self,dataloader):
        with torch.no_grad():
            datas = dataloader.generate_test_data()
            image, gt_orientation, gt_occ = self.preprocess_input(datas)
            out_ori, out_occ = self.model(image)

            pred_ori=out_ori*gt_occ
            # pred_ori=(pred_ori+1)/2
            # gt_orientation=(gt_orientation+1)/2
            # save_image(pred_ori[:,:,45,:,:],'test1.png')
            # save_image(gt_orientation[:,:,45,:,:],'test2.png')
            pred_ori=pred_ori.permute(0,2,3,4,1)
            # pred_ori=torch.reshape(pred_ori,(128,128,96*3))
            pred_ori=pred_ori.cpu().numpy()
            save_ori_as_mat(pred_ori,self.opt)







    def loss_backward(self, losses, optimizer,retain=False):
        optimizer.zero_grad()
        loss = sum(losses.values()).mean()
        loss.backward(retain_graph=retain)
        optimizer.step()

    def init_losses(self):
        self.total_loss = {}
        self.loss_global={}
        self.loss_local={}
    def get_latest_losses(self):
        self.total_loss={**self.loss_global,**self.loss_local}
        return self.total_loss

    def update_learning_rate(self, epoch):
        if epoch % 30 == 0 and epoch != 0:
            self.learning_rate = self.learning_rate // 2

        for param_group_local in self.optimizer_local.param_groups:
            param_group_local['lr'] = self.learning_rate

        for param_group_global in self.optimizer_global.param_groups:
            param_group_global['lr'] = self.learning_rate
