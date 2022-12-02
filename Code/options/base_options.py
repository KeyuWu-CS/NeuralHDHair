import os
import argparse
from solver.GrowingNetSolver import GrowingNetSolver
from solver.HairSpatNetSolver import HairSpatNetSolver
from  solver import get_option_setter
import pickle
from Tools import utils
class BaseOptions():
    def __init__(self):
        self.initialized=False

    def initialize(self,parser):

        parser.add_argument('--name', type=str, default='2021-10-13_bust',
                            help='name of the experiment. It decides where to store samples and models',)
        parser.add_argument('--check_name',type=str,default='2021-05-30-0')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--load_from_opt_file', action='store_true',
                            help='load the options from checkpoints and use that as default')
        parser.add_argument('--save_root', type=str, default='checkpoints/',
                                 help='name of the dir to save all the experiments')
        parser.add_argument('--current_path', type=str, default='/')
        parser.add_argument('--strand_dir', type=str, default='data/Train_input/')
        parser.add_argument('--video_dir', type=str, default='E:\wukeyu\hair\HairData\Growing')
        parser.add_argument('--growInv', action='store_true')
        parser.add_argument('--Bidirectional_growth', default=True)
        parser.add_argument('--continue_train', action='store_true')




        parser.add_argument('--init_type', type=str, default='xavier',
                            help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_variance', type=float, default=0.02,
                            help='variance of the initialization distribution')

        ####data
        parser.add_argument('--data_mode',type=str,default='image')
        parser.add_argument('--image_size', type=int, default=256, help='')
        parser.add_argument('--nThreads', default=4, type=int, help='# threads for loading data')

        ####set voxel
        parser.add_argument('--voxel_depth', type=int,default=192, help='the depth of voxel')
        parser.add_argument('--voxel_width', type=int, default=256, help='the width of voxel')
        parser.add_argument('--voxel_height', type=int, default=256, help='the height of voxel')
        parser.add_argument('--voxel_size', type=str, default="96,128,128", help='the height of voxel')

        #####model
        parser.add_argument('--model_name',type=str,default='OriCorrectNet')

        parser.add_argument('--min_channels', type=int, default=16, help='min channels in networks')
        parser.add_argument('--max_channels', type=int, default=64, help='max channels in networks')
        parser.add_argument('--condition',action='store_true')

        parser.add_argument('--warm_gpu',action='store_true')
        parser.add_argument('--use_HD',type=bool,default=False)




        self.initialized=True



        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)


        # parser=GrowingNetSolver.modify_options(parser)
        opt, unknown = parser.parse_known_args()
        parser=get_option_setter(opt.model_name)(parser)
        # parser=HairSpatNetSolver.modify_options(parser)



        current= os.path.join(os.getcwd())
        parser.set_defaults(current_path=current)


        # if there is opt_file, load it.
        # The previous default options will be overwritten
        if opt.load_from_opt_file:
            parser = self.update_options_from_file(parser, opt)

        opt = parser.parse_args()
        self.parser = parser
        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def option_file_path(self, opt, makedir=False):

        expr_dir = os.path.join(opt.current_path, opt.save_root,opt.name,'logs')
        if makedir:
            utils.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt')
        return file_name

    def save_options(self, opt):
        file_name = self.option_file_path(opt, makedir=True)
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(opt, opt_file)

    def update_options_from_file(self, parser, opt):
        new_opt = self.load_options(opt)
        for k, v in sorted(vars(opt).items()):
            if hasattr(new_opt, k) and v != getattr(new_opt, k):
                new_val = getattr(new_opt, k)
                parser.set_defaults(**{k: new_val})
        return parser

    def load_options(self, opt):
        file_name = self.option_file_path(opt, makedir=False)
        new_opt = pickle.load(open(file_name + '.pkl', 'rb'))
        return new_opt

    def parse(self, save=False):

        opt = self.gather_options()

        opt.isTrain = self.isTrain  # train or test

        self.print_options(opt)
        if opt.isTrain:
            self.save_options(opt)

        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)

        self.opt = opt
        return self.opt
