import os
import einops
import torch
import numpy as np
from PIL import Image
import imageio

from tutorial_dataset_test import MyDataset
from cldm.model import create_model, load_state_dict

if __name__ == "__main__":
    model = create_model('./models/single_ldm_v15.yaml')
    model.load_state_dict(load_state_dict('./models/single_epoch75.ckpt', location='cuda'))

    # TODO: create argparse
    seq_name = 'seq_0001'
    save_image = False
    save_video = True
    save_image_dir = f'test/{seq_name}/image'
    save_video_dir = f'test/{seq_name}/video'
    os.makedirs(save_image_dir, exist_ok=True)
    os.makedirs(save_video_dir, exist_ok=True)

    dataset = MyDataset(seq_name)

    reference = None

    x_samples = []
    for i in range(len(dataset)):
        data_dict = dataset.__getitem__(i)

        cond_img = torch.from_numpy(data_dict['cond_jpg']).float().cuda()
        cond_img = cond_img.unsqueeze(0)
        cond_img = einops.rearrange(cond_img, 'b h w c -> b c h w').clone()
        _, C, H, W = cond_img.shape
        cond = model.get_learned_conditioning(cond_img)
        cond = {"c_t_crossattn": cond}

        shape = (4, H // 8, W // 8)

        x_sample = model.sample(cond=cond, batch_size=1, return_intermediates=False, verbose=False, shape=shape)
        x_sample = model.decode_first_stage(x_sample)
            
        x_sample = (einops.rearrange(x_sample, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        x_samples.append(x_sample)

        if save_image:
            img = Image.fromarray(x_sample)
            img.save(os.path.join(save_image_dir, f'{i:03d}.png'))
    
    if save_video:
        imageio.mimwrite(x_samples, fps=12)
    

            
        