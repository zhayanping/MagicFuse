import torch
import os
from torch.utils.data.dataset import Dataset
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader,DistributedSampler
import random

class DATASET:
    def __init__(self, path, batchsize, mode='train'):
        self.mode = mode
        self.batchsize = batchsize
        if self.mode=='train':
            self.train_dataset = dataset_gen(path, mode='train')
        elif self.mode == 'test':
            self.test_dataset = dataset_gen(path, mode='test')
    def get_loaders(self):
        if self.mode=='train':
            train_sampler = DistributedSampler(self.train_dataset,
                                       shuffle=True,
                                      )
            train_loader = DataLoader(self.train_dataset, batch_size=self.batchsize,
                                     shuffle=False, num_workers=4,
                                     sampler=train_sampler,
                                     pin_memory=True,
                                      )
            return train_sampler, train_loader
        elif self.mode == 'test':
            test_sampler = DistributedSampler(self.test_dataset,
                            shuffle=True,
                            )
            test_loader = DataLoader(self.test_dataset, batch_size=self.batchsize,
                                     shuffle=False, num_workers=4,
                                     sampler=test_sampler,
                                     pin_memory=True
                                     )
            return test_loader
        return None



class dataset_gen(Dataset):
    def __init__(self, data_path, mode='train'):
        super().__init__()
        self.data_path = data_path
        self.mode = mode
        if mode == 'train':
            self.raw_path = os.path.join(data_path, "vis")
            self.label_path = os.path.join(data_path, "label")
            self.filenames = [
                f for f in os.listdir(self.label_path)
                if f.lower().endswith(('.jpg', '.png'))
            ]
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10, fill=0),
            ])
        elif mode == 'test':
            self.raw_path = os.path.join(data_path, "vis")
            self.label_path = os.path.join(data_path, "vis")
            self.filenames = [
                f for f in os.listdir(self.label_path)
                if f.lower().endswith(('.jpg', '.png'))
                ]
        else:
            raise ValueError("mode should be 'train' or 'test'")

        self.length = len(self.filenames)
        self.target_height, self.target_width = 480,640
        self.crop = transforms.RandomCrop((self.target_height, self.target_width))

    def __len__(self):
        return self.length
    def __getitem__(self, index):
        name = self.filenames[index]
        rng_state = torch.random.get_rng_state()
        seed = torch.randint(0, 1000, (1,)).item()
        if self.mode == 'train':
            img_raw = Image.open(os.path.join(self.raw_path, name)).convert('RGB')
            label = Image.open(os.path.join(self.label_path, name)).convert('L')
            img_raw = self.__img_crop(img_raw, seed=seed)
            label = self.__img_crop(label, seed=seed)
            img_raw = transforms.ToTensor()(img_raw)
            label = transforms.PILToTensor()(label)
            label = label.squeeze(0)
            assert img_raw.shape ==  torch.Size([3, self.target_height, self.target_width])
            torch.random.set_rng_state(rng_state)
            return img_raw, label, name
        else:
            img_raw = Image.open(os.path.join(self.raw_path, name)).convert('RGB')
            label = Image.open(os.path.join(self.label_path, name)).convert('L')
            img_raw = self.__img_crop(img_raw, seed=seed)
            label = self.__img_crop(label, seed=seed)
            img_raw = transforms.ToTensor()(img_raw)
            label = transforms.PILToTensor()(label)
            return img_raw, label, name
    def __img_crop(self, img, seed):
        torch.manual_seed(seed)
        if self.mode == 'train':
            img = self.transform(img)
        original_width, original_height = img.size
        if not original_height == self.target_height or not original_width == self.target_width:
            scale = max(self.target_width / original_width, self.target_height / original_height)
            new_width = max(int(original_width * scale), self.target_width)
            new_height = max(int(original_height * scale), self.target_height)
            new_size = (new_width, new_height)
            img = img.resize(new_size, Image.BILINEAR)
        img = self.crop(img)
        return img