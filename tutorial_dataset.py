import json
import cv2
import numpy as np
import os
import random
import math

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
    

class MyMutualDataset(Dataset):
    def __init__(self):
        self.data = []

        self.data_dir = '/home/wteng/data/sequence_all/training/'
        with open(os.path.join(self.data_dir, 'prompts.json'), 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))
        f.close()
        self.frame_range = 12
    
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

        # reference image
        reference_range = self.frame_range
        source_seq_id, _, _, source_frame_id = source_filename.split("/")
        source_seq_length = len(os.listdir(os.path.join(self.data_dir, source_seq_id, "target", "image")))
        if source_seq_length < reference_range:
            reference_range = source_seq_length
        
        frame_id = int(source_frame_id.split("_")[-1])
        half_range = reference_range // 2
        
        start_id = frame_id - half_range
        if start_id < 0:
            start_id = 0
        end_id = frame_id + half_range
        if end_id > source_seq_length:
            end_id = source_seq_length
        
        ref_frame_ids = list(range(start_id, end_id))
        ref_id = random.choice(ref_frame_ids)
        ref_filename = source_seq_id + "/target/image/frame_" + f"{ref_id:06d}"

        ref_img = cv2.imread(self.data_dir + ref_filename + '.png')
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

        ref_img = (ref_img.astype(np.float32) / 127.5) - 1.0
        
        source_camfile = self.data_dir + source_seq_id + "/target/camera/" + source_frame_id + ".json"
        with open(source_camfile, 'r') as f:
            source_cam = json.load(f)
        f.close()
        
        source_ext = np.array(source_cam['transform_matrix'])
        source_center = source_ext[:3, 3]

        ref_camfile = self.data_dir + source_seq_id + '/target/camera/frame_' + f"{ref_id:06d}.json"
        with open(ref_camfile, 'r') as f:
            ref_cam = json.load(f)
        f.close()
        
        ref_ext = np.array(ref_cam['transform_matrix'])
        ref_center = ref_ext[:3, 3]

        dist = (ref_center - source_center).astype(np.float32)

        return dict(jpg=target_img, loc=dist, cond_jpg=source_img, ref_jpg=ref_img, txt=prompt, hint=source_img)



