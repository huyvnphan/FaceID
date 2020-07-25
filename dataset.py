import os
import random

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class FaceIDDataset(Dataset):
    def __init__(self, data_dir, train=True):
        self.train = train
        self.poses_per_person = 51

        # True mean and std
        # mean = [0.5255, 0.5095, 0.4861, 0.7114]
        # std = [0.2075, 0.1959, 0.1678, 0.2599]

        # Mean and std so that input in range [-1, 1]
        mean = [0.5, 0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5, 0.5]

        if self.train:
            self.data_dir = os.path.join(data_dir, "train")
            self.no_people = 26
            self.transform = transforms.Compose(
                [
                    transforms.CenterCrop(300),
                    transforms.RandomCrop(256),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        else:
            self.data_dir = os.path.join(data_dir, "val")
            self.no_people = 5
            self.transform = transforms.Compose(
                [
                    transforms.CenterCrop(256),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )

    def __len__(self):
        if self.train:
            return self.no_people * self.poses_per_person
        else:
            return self.no_people * self.poses_per_person * 50  # validate more samples

    def __getitem__(self, index):
        # Get 1st photo
        real_index = index % (self.no_people * self.poses_per_person)
        person_id = real_index // self.poses_per_person
        pose_id = real_index % self.poses_per_person
        x0_array = np.load(
            os.path.join(
                self.data_dir,
                "person" + str(person_id) + "_pose" + str(pose_id) + ".npy",
            )
        )
        x0 = Image.fromarray(x0_array)

        # Get 2nd photo
        if self.train:
            choice = random.randint(0, 3)
        else:
            choice = random.randint(0, 1)

        if choice == 0:
            # Same person - correct RGB, correct D
            pose2_id = random.choice(
                list(set(range(self.poses_per_person)) - set([pose_id]))
            )
            x1_array = np.load(
                os.path.join(
                    self.data_dir,
                    "person" + str(person_id) + "_pose" + str(pose2_id) + ".npy",
                )
            )
            x1 = Image.fromarray(x1_array)
            y = 1
        elif choice == 1:
            # wrong RGB, wrong D
            person2_id = random.choice(
                list(set(range(self.no_people)) - set([person_id]))
            )
            pose2_id = random.randint(0, self.poses_per_person - 1)
            x1_array = np.load(
                os.path.join(
                    self.data_dir,
                    "person" + str(person2_id) + "_pose" + str(pose2_id) + ".npy",
                )
            )
            x1 = Image.fromarray(x1_array)
            y = -1
        elif choice == 2:
            # correct RGB, wrong D
            person2_id = random.choice(
                list(set(range(self.no_people)) - set([person_id]))
            )
            pose2_id = random.randint(0, self.poses_per_person - 1)
            x1_array = np.load(
                os.path.join(
                    self.data_dir,
                    "person" + str(person2_id) + "_pose" + str(pose2_id) + ".npy",
                )
            )
            x1_array[:, :, :3] = x0_array[:, :, :3]
            x1 = Image.fromarray(x1_array)
            y = -1
        else:
            # wrong RGB, correct D
            person2_id = random.choice(
                list(set(range(self.no_people)) - set([person_id]))
            )
            pose2_id = random.randint(0, self.poses_per_person - 1)
            x1_array = np.load(
                os.path.join(
                    self.data_dir,
                    "person" + str(person2_id) + "_pose" + str(pose2_id) + ".npy",
                )
            )
            x1_array[:, :, 3] = x0_array[:, :, 3]
            x1 = Image.fromarray(x1_array)
            y = -1

        x0 = self.transform(x0)
        x0_mask = (x0[3] > -0.999).unsqueeze(0).float()
        x0 = x0 * x0_mask

        x1 = self.transform(x1)
        x1_mask = (x1[3] > -0.999).unsqueeze(0).float()
        x1 = x1 * x1_mask

        return (x0, x1, y)
