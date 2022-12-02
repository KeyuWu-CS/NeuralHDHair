from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self,parser):
        BaseOptions.initialize(self,parser)

        # data_loader
        parser.add_argument('--train_data_dir', type=str, default='E:\wukeyu\hair\HairData\Growing')
        parser.add_argument('--val_data_dir', type=str, default='D:/HairData/Val')
        parser.add_argument('--num_of_val', type=int, default=1)
        parser.add_argument('--print_freq', type=int, default=10,help='frequency of showing training results on console')
        parser.add_argument('--display_freq', type=int, default=20,help='frequency of showing training results on screen')
        parser.add_argument('--save_epoch_freq', type=int, default=30,help='frequency of saving checkpoints at the end of epochs')
        # learning rate and loss weight
        parser.add_argument('--niter', type=int, default=150000, help='training iterations')
        # parser.add_argument('--niter', type=int, default=50,
        #                     help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
        parser.add_argument('--niter_decay', type=int, default=0,
                            help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')

        # display the results
        parser.add_argument('--save_latest_freq', type=int, default=5000,
                                 help='frequency of saving the latest results')

        # train which network, the networks are trained separatly currently, to do: train together
        parser.add_argument('--train_flow', action='store_true')

        parser.add_argument('--which_iter',type=str,default='latest')

        parser.add_argument('--pred_label',action='store_true')

        parser.set_defaults(train_flow=False)

        self.isTrain=True

        return parser