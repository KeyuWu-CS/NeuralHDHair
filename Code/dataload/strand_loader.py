from dataload.base_loader import base_loader
from Tools.utils import *
import torch
class strand_loader(base_loader):

    def initialize(self,opt):

        self.opt = opt
        self.batch_size = opt.batch_size
        self.sd_num = opt.sd_per_batch
        self.pt_num = opt.pt_per_strand
        self.image_size = opt.image_size
        self.isTrain = opt.isTrain
        self.parent_dir = os.path.dirname(os.getcwd())
        self.root = os.path.join(self.parent_dir,opt.strand_dir)
        if self.isTrain:
            self.num_of_val = opt.num_of_val
            self.train_corpus = []
            self.train_nums = 0
            self.generate_corpus()
        else:
            path = os.path.join(self.root, opt.test_file)
            self.gt_orientation = get_ground_truth_3D_ori(path, False, opt.growInv)[None]
            if self.opt.Bidirectional_growth:
                self.num_strands=self.opt.num_root

    def generate_corpus(self):
        # exclude the tail, to do improve this
        self.all_data = get_all_the_data(self.root)
        self.train_corpus=self.all_data[:-self.num_of_val]

        self.val_corpus = self.train_corpus[-self.num_of_val:]
        random.shuffle(self.train_corpus)

        self.train_nums = len(self.train_corpus)
        print('val strand:',self.val_corpus)
        print('train strand:',self.train_corpus)
        print(f"num of training data: {self.train_nums}")




    def __getitem__(self, index):
        file_name=self.train_corpus[index]

        segments, points = load_strand(file_name,True)

        gt_orientation = get_ground_truth_3D_ori(file_name, False, growInv=self.opt.growInv)
        sample_voxel = np.load(os.path.join(file_name, 'sample_voxel.npy'))

        strands, labels = sample_to_padding_strand1(sample_voxel, segments, points, self.pt_num, self.sd_num,
                                                    growInv=self.opt.growInv)
        strands=torch.from_numpy(strands)
        labels=torch.from_numpy(labels)
        gt_orientation=torch.from_numpy(gt_orientation)

        return_list={
            'gt_ori':gt_orientation,
            'strands':strands,
            'labels':labels
        }
        return return_list


    def generate_test_data(self,growInv=False):
        self.segments=np.array(self.segments)
        index=np.cumsum(self.segments)
        if growInv:
            index=index-1
        else:
            index=index[:-1]

        strands=self.points[index][None]

        self.gt_orientation = torch.from_numpy(self.gt_orientation)
        strands=strands[...,None,:]
        strands=torch.from_numpy(strands)
        return_list={
            'gt_ori': self.gt_orientation,
            'strands':strands,
            'labels': None
        }
        return return_list


    def generate_random_root(self):


        occ=np.linalg.norm(self.gt_orientation,axis=-1)[0]
        occ=(occ>0).astype(np.float32)
        # occ[:,30:,:]=0
        samle_voxel_index =np.where(occ>0)
        samle_voxel_index=np.array(samle_voxel_index)
        samle_voxel_index=samle_voxel_index.transpose(1,0)
        random_points=samle_voxel_index[np.random.randint(0,samle_voxel_index.shape[0]-1,size=self.opt.num_root)]
        random_points=random_points[:,::-1]+np.random.random(random_points.shape[:])[None]
        random_points=random_points[...,None,:]
        self.gt_orientation=torch.from_numpy(self.gt_orientation)
        random_points=torch.from_numpy(random_points)
        random_points=torch.reshape(random_points,(len(self.opt.gpu_ids),-1,1,3))


        return_list={
            'gt_ori': self.gt_orientation,
            'strands': random_points,
            'labels': None
        }

        return return_list


