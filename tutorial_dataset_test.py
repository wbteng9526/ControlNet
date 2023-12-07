import json
import cv2
import numpy as np
import os
import random
import math

from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, seq_name):
        self.data = []
        self.seq_name = seq_name

        self.data_dir = '/home/wteng/data/sequence_all/testing/'
        with open(os.path.join(self.data_dir, 'prompts.json'), 'rt') as f:
            for line in f:
                if line.split('/')[0] == self.seq_name:
                    self.data.append(json.loads(line))
        f.close()

        self.camera_dir = self.data_dir + 'target/camera/'
        camera_paths = [self.camera_dir + f for f in sorted(os.listdir(self.camera_dir))]
        cam_cs = []
        for path in camera_paths:
            with open(path, 'r') as f:
                cam = json.load(f)
            f.close()
            cam_ext = np.array(cam['transform_matrix'])
            cam_c = cam_ext[:3, 3]
            cam_cs.append(cam_c)
        self.cam_dcs[1:] = [(cam_cs[i]- cam_cs[i-1]).astype(np.float32) for i in range(1, len(cam_cs))]
        self.cam_dcs[0] = np.zeros_like(self.cam_dcs[0]).astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source_img = cv2.imread(self.data_dir + source_filename + '.png')
        target_img = cv2.imread(self.data_dir + target_filename + '.png')

        # Do not forget that OpenCV read images in BGR order.
        source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source_img = source_img.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target_img = (target_img.astype(np.float32) / 127.5) - 1.0

        dist = self.cam_dcs[idx]
        return dict(jpg=target_img, loc=dist, txt=prompt, cond_jpg=source_img)