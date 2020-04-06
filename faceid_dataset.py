import os
import random
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class FaceIDDataset(Dataset):
    def __init__(self, data_dir, train=True):
        self.poses_per_person = 51
        mean = [0.5255, 0.5095, 0.4861, 0.7114]
        std = [0.2075, 0.1959, 0.1678, 0.2599]
        if train:
            self.data_dir = os.path.join(data_dir, 'train')
            self.no_people = 26
            self.transform = transforms.Compose([transforms.RandomVerticalFlip(),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.RandomApply([transforms.RandomRotation(30)]),
                                                 transforms.RandomCrop(300),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean, std)])
        else:
            self.data_dir = os.path.join(data_dir, 'val')
            self.no_people = 5
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean, std)])
            
    def __len__(self):
        return self.no_people * self.poses_per_person
        
    def __getitem__(self, index):
        
        # Get 1st photo
        person_id = index // self.poses_per_person
        pose_id = index % self.poses_per_person
        x0_array = np.load(os.path.join(self.data_dir, 'person' + str(person_id) + '_pose' + str(pose_id) + '.npy'))
        x0 = Image.fromarray(x0_array)
                
        # Get 2nd photo
        if random.random() > 0.5:
            # Same person
            pose2_id = random.choice(list(set(range(self.poses_per_person)) - set([pose_id])))
            x1_array = np.load(os.path.join(self.data_dir, 'person' + str(person_id) + '_pose' + str(pose2_id) + '.npy'))
            x1 = Image.fromarray(x1_array)
            y = 1
        else:
            person2_id = random.choice(list(set(range(self.no_people)) - set([person_id])))
            pose2_id = random.randint(0, self.poses_per_person-1)
            x1_array = np.load(os.path.join(self.data_dir, 'person' + str(person2_id) + '_pose' + str(pose2_id) + '.npy'))       
            x1 = Image.fromarray(x1_array)
            y = -1
                
        x0 = self.transform(x0)
        x1 = self.transform(x1)
            
        return (x0, x1, y)