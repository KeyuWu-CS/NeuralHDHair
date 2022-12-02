import os
from Tools.utils import *
import torch
import torch.nn as nn
class BaseSolver(nn.Module):

    @staticmethod
    def modify_options(parser):
        pass

    @staticmethod
    def parse_opt(opt):
        pass

    def initialize(self, opt):



        self.batch_size = opt.batch_size
        # self.depth = opt.voxel_depth
        # self.width = opt.voxel_width
        # self.height = opt.voxel_height
        self.depth,self.width,self.height=opt.voxel_size.split(',')
        self.depth=int(self.depth)
        self.width=int(self.width)
        self.height=int(self.height)
        self.image_size = opt.image_size
        self.min_channels = opt.min_channels
        self.max_channels = opt.max_channels




        save_path=os.path.join(opt.current_path,opt.save_root)


        self.train_log_dir = os.path.join(save_path,opt.name, 'logs', 'train')
        self.val_log_dir = os.path.join(save_path, opt.name,'logs', 'val')
        self.checkpoint_dir = os.path.join(save_path,opt.name, 'checkpoint')

        mkdirs([save_path, self.train_log_dir, self.val_log_dir, self.checkpoint_dir])

        self.isTrain = opt.isTrain
        if self.isTrain:
            self.save_iter = opt.save_latest_freq
            self.iterations = opt.niter
            self.display_iter = opt.display_freq
            self.learning_rate = opt.lr
            self.continue_train = opt.continue_train

            # the train data dir & vai data dir are shared for each sub class

        #     self.train_data_dir = opt.train_data_dir
        #     self.val_data_dir = opt.val_data_dir
        # else:
        #     self.test_data_dir=opt.test_data_dir

    def load_network(self,net, label, epoch, opt,specify_path=None):
        if specify_path is not None:
            save_path=specify_path
            save_path=os.path.join(save_path,label+'_'+opt.which_iter+'.pth')
        else:
            save_filename = '%s_%s.pth' % (label, epoch)
            save_dir = os.path.join(opt.current_path,opt.save_root, opt.check_name,'checkpoint')
            save_path = os.path.join(save_dir, save_filename)
        print('load weights from::', save_path)
        weights = torch.load(save_path)
        self.load_weights(net, weights)
        return net

    def save_network(self,net, label, epoch, opt):
        save_filename = '%s_%s.pth' % (label, epoch)
        save_path = os.path.join(opt.current_path,opt.save_root, opt.name,'checkpoint')
        if int(torch.__version__[2]) >= 6:
            torch.save(net.cpu().state_dict(), save_path + '/' + save_filename,_use_new_zipfile_serialization=False)
        else:
            torch.save(net.cpu().state_dict(), save_path+'/'+save_filename)
        if len(opt.gpu_ids) and torch.cuda.is_available():
            net.cuda()

    def load_weights(self,cnn_model, weights):
        """
        argus:
        :param cnn_model: the cnn networks need to load weights
        :param weights: the pretrained weigths
        :return: no return
        """
        from torch.nn.parameter import Parameter
        pre_dict = cnn_model.state_dict()

        for key, val in weights.items():
            # print(key)

            if key[0:7] == 'module.':  # the pretrained networks was trained on multi-GPU
                key = key[7:]  # remove 'module.' from the key
            if key in pre_dict.keys():
                # print('test: ',key)
                if isinstance(val, Parameter):
                    val = val.data
                pre_dict[key].copy_(val)
        cnn_model.load_state_dict(pre_dict)

    # def load(self, saver, checkpoint_dir):
    #     print(" [*] Reading latest checkpoint from folder %s." % (checkpoint_dir))
    #     ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    #
    #     if ckpt and ckpt.model_checkpoint_path:
    #         ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    #         print(ckpt_name)
    #         # ckpt_name='model.ckpt-19000'
    #         saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
    #         print("Load form {}".format(os.path.join(checkpoint_dir, ckpt_name)))
    #         return True
    #     else:
    #         print("Load Failure!")
    #         return False