import json
import cv2
import numpy as np
import os

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self):
        self.data = []

        self.data_dir = '/home/wteng/data/sequence/training/'
        with open(os.path.join(self.data_dir, 'prompts.json'), 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))
        f.close()

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

        return dict(jpg=target_img, txt=prompt, cond_jpg=source_img)

