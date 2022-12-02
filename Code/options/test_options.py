from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self,parser):
        BaseOptions.initialize(self, parser)
        parser.set_defaults(pt_per_strand=128)
        parser.set_defaults(batch_size=1)
        # parser.set_defaults(warm_gpu=True)
        parser.add_argument('--test_data_dir',type=str,default='E:\wukeyu\hair\DataSet\Train_input',help='')
        parser.add_argument('--test_file',default=None)
        parser.add_argument('--use_pred_ori',action='store_true')
        parser.add_argument('--pred_ori', action='store_true')
        parser.add_argument('--num_root',type=int,default=15000)
        parser.add_argument('--which_iter', type=str, default='latest')
        parser.add_argument('--data_dir',type=str,default='/home/wky/data/Test_input')
        parser.add_argument('--save_dir',type=str,default='results')
        parser.add_argument('--add_face_width',type=float,default=10)
        parser.add_argument('--add_align_height',type=int,default=0)
        parser.add_argument('--add_offset_x',type=int,default=0)
        parser.add_argument('--translate_ori',type=int,default=3)

        self.isTrain = False

        return parser