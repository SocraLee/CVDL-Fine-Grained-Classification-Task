from torch.utils.data import Dataset
import cv2
import numpy as np
import torch

class ImgDataset(Dataset):
    def __init__(self, dataframe, image_dir,num_classes, transforms=None):
        super().__init__()

        self.image_ids = dataframe['ID']
        self.image_labels = dataframe['Category']
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms
        self.num_classes = num_classes

    def __getitem__(self, index: int):

        image_id = self.image_ids[index]
        image_id = format(image_id, '05d')
        try:
            image = cv2.imread(f'{self.image_dir}/'+image_id+'.jpg')
            image = torch.from_numpy(np.transpose(image, (2, 0, 1)))
        except:
            print(f'{self.image_dir}/{image_id}.jpg')

        # # 数据增强
        # if self.transforms:
        #     sample = {'image': image}
        #     sample = self.transforms(**sample)
        #     image = sample['image']

        image = image/255
        image_label_index = int(self.image_labels[index])
        image_label = torch.zeros(self.num_classes)
        image_label[image_label_index] = 1


        return image, image_label,image_label_index

    def __len__(self) -> int:
        return self.image_ids.shape[0]