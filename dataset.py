import os
import random

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class FaceIDDataset(Dataset):
    def __init__(self, data_dir, split, size):
        self.split = split
        self.size = size

        self.poses_per_person = 51

        # True mean and std
        # mean = [0.5255, 0.5095, 0.4861, 0.7114]
        # std = [0.2075, 0.1959, 0.1678, 0.2599]

        # Mean and std so that input in range [-1, 1]
        mean = [0.5, 0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5, 0.5]

        if self.split == "train":
            self.data_dir = os.path.join(data_dir, "train")
            self.no_people = 26
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(),
                    # A.ShiftScaleRotate(
                    #     shift_limit=0.0625,
                    #     scale_limit=0.1,
                    #     rotate_limit=15,
                    #     p=0.5,
                    #     border_mode=0,
                    # ),
                    A.RandomCrop(256, 256),
                    A.Normalize(mean, std),
                    ToTensorV2(),
                ]
            )

        elif self.split == "val":
            self.data_dir = os.path.join(data_dir, "val")
            self.no_people = 5
            self.transform = A.Compose(
                [A.CenterCrop(256, 256), A.Normalize(mean, std), ToTensorV2()]
            )
        else:
            raise ("Only support train or val")

    def random_apply_mask(self, x):
        new_x = x.clone()
        if self.split == "train":
            if random.random() > 0.5:
                x_mask = (x[3] > -0.999).unsqueeze(0).float()
                new_x[:3] = x[:3] * x_mask
        return new_x

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # First photo is deterministic. Second photo is random.
        # Get 1st photo
        real_index = index % (self.no_people * self.poses_per_person)
        person_id = real_index // self.poses_per_person
        pose_id = real_index % self.poses_per_person
        x_ref = np.load(
            os.path.join(
                self.data_dir,
                "person" + str(person_id) + "_pose" + str(pose_id) + ".npy",
            )
        )

        # Get 2nd photo
        if random.random() > 0.5:
            # Same person - correct RGB, correct D
            pose2_id = random.choice(
                list(set(range(self.poses_per_person)) - set([pose_id]))
            )
            x = np.load(
                os.path.join(
                    self.data_dir,
                    "person" + str(person_id) + "_pose" + str(pose2_id) + ".npy",
                )
            )
            y = 1
        else:
            # Different person - wrong RGB, wrong D
            person2_id = random.choice(
                list(set(range(self.no_people)) - set([person_id]))
            )
            pose2_id = random.randint(0, self.poses_per_person - 1)
            x = np.load(
                os.path.join(
                    self.data_dir,
                    "person" + str(person2_id) + "_pose" + str(pose2_id) + ".npy",
                )
            )
            y = -1

        x_ref = self.transform(image=x_ref)["image"]
        x_ref = self.random_apply_mask(x_ref)

        x = self.transform(image=x)["image"]
        x = self.random_apply_mask(x)

        return x_ref, x, y
