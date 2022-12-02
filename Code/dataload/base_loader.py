from torch.utils import data
from torchvision.transforms import transforms
from PIL import Image
class base_loader(data.Dataset):

    def __init__(self):
        super(base_loader,self).__init__()

    def initialize(self):
        pass



    def __len__(self):
        return len(self.train_corpus)

    def generate_corpus(self):
        pass


    def get_transform(self,image_size,flip=False,normalize=False):
        transform_list=[]
        transform_list.append(transforms.Resize([image_size,image_size],Image.NEAREST))
        if flip:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, flip)))
        transform_list += [transforms.ToTensor()]
        if normalize:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5))]
        return transforms.Compose(transform_list)
def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img