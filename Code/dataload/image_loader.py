from dataload.base_loader import base_loader
from Tools.utils import *
import torch
import shutil
class image_loader(base_loader):

    def initialize(self,opt):

        self.opt = opt
        self.batch_size = opt.batch_size
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
            self.num_strands=self.opt.num_root

    def generate_corpus(self):
        # exclude the tail, to do improve this
        self.all_data = get_all_the_data(self.root)
        self.train_corpus=self.all_data[:-self.num_of_val]

        self.val_corpus = self.train_corpus[-self.num_of_val:]
        random.shuffle(self.train_corpus)

        self.train_nums = len(self.train_corpus)
        # print('val strand:',self.val_corpus)
        # print('train strand:',self.train_corpus)
        print(f"num of training data: {self.train_nums}")

    def __getitem__(self, index):
        data_list={}
        file_name=self.train_corpus[index]
        flip=False
        image=get_image(file_name,flip,self.opt.image_size,self.opt.Ori_mode,self.opt.blur_ori,no_use_depth=True,use_gt=True,use_conf=self.opt.use_conf)
        image=torch.from_numpy(image)
        image=image.permute(2,0,1)
        Ori2D = image.clone()

        if not self.opt.no_use_bust:
            image = get_Bust(file_name, image, self.opt.image_size)

        data_list['image']=image
        if self.opt.use_HD:
            add_info=get_add_info(file_name,self.opt.strand_size,self.opt.info_mode,use_gt=True)
            # strand2D=get_strand2D(file_name,self.opt.strand_size,self.opt.strand_mode,use_gt)
            if self.opt.info_mode!='L':
                add_info=torch.from_numpy(add_info)
                add_info=add_info.permute(2,0,1)
            data_list['add_info']=add_info


        depth=get_depth(file_name,self.opt.image_size)

        depth=torch.from_numpy(depth)
        depth=depth.permute(2,0,1)
        data_list['depth']=depth


        gt_orientation = get_ground_truth_3D_ori(file_name, flip, growInv=self.opt.growInv)


        # image,gt_orientation,strand2D=self.random_translation(image,gt_orientation,strand2D)
        gt_orientation,data_list=self.random_translation(self.opt.image_size,gt_orientation,data_list)


        mask=np.linalg.norm(gt_orientation,axis=-1)
        gt_occ=(mask>0).astype(np.float32)[...,None]
        gt_orientation=torch.from_numpy(gt_orientation)
        gt_occ=torch.from_numpy(gt_occ)
        gt_orientation=gt_orientation.permute(3,0,1,2)
        gt_occ=gt_occ.permute(3,0,1,2)

        if 'add_info' in data_list:
            add_info=data_list['add_info']
        else:
            add_info=torch.Tensor(0)
        return_list={
            'gt_ori':gt_orientation,
            'image':data_list['image'],
            'gt_occ':gt_occ,
            'Ori2D':Ori2D,
            'add_info':add_info,
            'depth':data_list['depth']

        }
        return return_list


    def generate_test_data(self):
        path = os.path.join(self.root, self.opt.test_file)
        save_path=os.path.join(self.opt.current_path, self.opt.save_root, self.opt.check_name, 'record', self.opt.test_file)
        if not os.path.exists(save_path):
            mkdir(save_path)
        orismooth=cv2.imread(os.path.join(path,'Ori.png'))
        orismooth=cv2.resize(orismooth,(1024,1024))
        padding=np.zeros((4*8,1024,3))
        orismooth=np.concatenate([padding,orismooth],axis=2)[:1024]
        cv2.imwrite(os.path.join(save_path,'OriSmooth2D.png'),orismooth)
        # shutil.copyfile(os.path.join(path,'Ori.png'),os.path.join(save_path,'OriSmooth2D.png'))
        image = get_image(path, False, self.opt.image_size, self.opt.Ori_mode,self.opt.blur_ori,self.opt.no_use_depth,use_gt=False)
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)

        Ori2D=image.clone()
        if not self.opt.no_use_L:
            print('use L map')
            L_map = get_luminance_map(path, False, self.opt.image_size,self.opt.no_use_bust)
            image=torch.cat([image,L_map],0)
        elif not self.opt.no_use_bust:
            image = get_Bust(path, image, self.opt.image_size)
        if os.path.exists(path+'/Ori_gt.mat'):
            gt_orientation = get_ground_truth_3D_ori(path, False, growInv=self.opt.growInv)
            mask = np.linalg.norm(gt_orientation, axis=-1)
            gt_occ = (mask > 0).astype(np.float32)[..., None]
            gt_orientation = torch.from_numpy(gt_orientation)
            gt_occ = torch.from_numpy(gt_occ)
            gt_orientation = gt_orientation.permute(3, 0, 1, 2)
            gt_occ = gt_occ.permute(3, 0, 1, 2)

            gt_occ=torch.unsqueeze(gt_occ,0)
            gt_orientation=torch.unsqueeze(gt_orientation,0)

        else:
            gt_orientation=torch.Tensor(0)
            gt_occ=torch.Tensor(0)
        image = torch.unsqueeze(image, 0)
        Ori2D=torch.unsqueeze(Ori2D,0)

        return_list = {
            'gt_ori': gt_orientation,
            'image': image,
            'gt_occ': gt_occ,
            'Ori2D':Ori2D
        }
        return return_list


    def generate_random_root(self,ori):
        # occ=torch.norm(ori,p=2,dim=1,keepdim=True)
        # occ=(occ>0).type(torch.float32)
        occ=np.linalg.norm(ori,axis=-1)[0]
        occ=(occ>0).astype(np.float32)
        samle_voxel_index =np.where(occ>0)
        samle_voxel_index=np.array(samle_voxel_index)
        samle_voxel_index=samle_voxel_index.transpose(1,0)
        random_points=samle_voxel_index[np.random.randint(0,samle_voxel_index.shape[0]-1,size=self.opt.num_root)]
        random_points=random_points[:,::-1]+np.random.random(random_points.shape[:])[None]
        random_points=random_points[...,None,:]
        random_points=torch.from_numpy(random_points)
        random_points=torch.reshape(random_points,(len(self.opt.gpu_ids),-1,1,3))

        return random_points



    def translation(self,data,offset_x,offset_y,mul,shape,rand_x,rand_y):
        padding_x=torch.zeros(shape[0],offset_x*mul,shape[1])
        padding_y=torch.zeros(shape[0],shape[1],offset_y*mul)
        if rand_x<0.5:
            data=torch.cat([data,padding_x],dim=1)[:,offset_x*mul:,:]
        else:
            data = torch.cat([padding_x, data], dim=1)[:, :-offset_x*mul, :]
        if rand_y<0.5:
            data = torch.cat([data, padding_y], dim=2)[:, :, offset_y*mul:]
        else:
            data = torch.cat([ padding_y,data], dim=2)[:, :, :-offset_y*mul]
        return data

    def random_translation(self, image_size, gt, data_list):
        offset_x = random.randint(1, 10) * 2
        offset_y = random.randint(1, 10) * 2
        rand_x = random.random()
        rand_y = random.random()

        for k,v in data_list.items():
            C,H,W=v.shape[:]
            mul=H//image_size
            data_list[k]=self.translation(v,offset_x,offset_y,mul,[C,H,W],rand_x,rand_y)

        gt_depth, gt_x, gt_y, gt_channel = gt.shape[:]

        padding_gt_x = torch.zeros(gt_depth, offset_x // 2, gt_y, gt_channel)
        padding_gt_y = torch.zeros(gt_depth, gt_x, offset_y // 2, gt_channel)
        if rand_x < 0.5:
            gt = np.concatenate([gt, padding_gt_x], axis=1)[:, offset_x // 2:, :, :]
        else:
            gt = np.concatenate([padding_gt_x, gt], axis=1)[:, :-offset_x // 2, :, :]
        if rand_y < 0.5:
            gt = np.concatenate([gt, padding_gt_y], axis=2)[:, :, offset_y // 2:, :]
        else:
            gt = np.concatenate([padding_gt_y, gt], axis=2)[:, :, :-offset_y // 2, :]
        return gt,data_list







