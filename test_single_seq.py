import os
import einops
import torch
import numpy as np
from PIL import Image
import imageio
from tqdm import tqdm

from tutorial_dataset_test import MyDataset
from cldm.model import create_model, load_state_dict


def get_data_info(data_dict):
    cond_txt = data_dict['txt']
    control = torch.from_numpy(data_dict['hint']).float().cuda()
    control = control.unsqueeze(0)
    control = einops.rearrange(control, 'b h w c -> b c h w')
    _, C, H, W = control.shape

    reference_location = torch.from_numpy(data_dict['loc']).float().cuda()
    reference_location = reference_location.unsqueeze(0)

    shape = (1, 4, H // 8, W // 8)
    return cond_txt, control, reference_location, shape

if __name__ == "__main__":
    model = create_model('./models/mutual_cldm_v15.yaml')
    model.load_state_dict(load_state_dict('./models/mutual_finetune_control_epoch40.ckpt', location='cuda'))
    model = model.cuda()

    # TODO: create argparse
    seq_name = 'seq_0001'
    save_image = True
    save_video = True
    save_image_dir = f'exp/carla_control/image/{seq_name}/'
    save_video_dir = f'exp/carla_control/video/{seq_name}/'
    os.makedirs(save_image_dir, exist_ok=True)
    os.makedirs(save_video_dir, exist_ok=True)

    dataset = MyDataset(seq_name)

    x_samples = []
    for i in tqdm(range(len(dataset))):
        data_dict = dataset.__getitem__(i)

        cond_txt, control, _, shape = get_data_info(data_dict)

        x_sample = model.sample(
            cond={"c_t_crossattn":[model.get_learned_conditioning(cond_txt)], "c_t_concat":[control]},
            reference=None,
            reference_location=None, 
            batch_size=1, return_intermediates=False, verbose=True, shape=shape
        )
        x_sample = model.decode_first_stage(x_sample)
            
        x_sample = (einops.rearrange(x_sample, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        x_samples.append(x_sample[0])

        if save_image:
            img = Image.fromarray(x_sample[0])
            img.save(os.path.join(save_image_dir, f'{i:03d}.png'))
    
    if save_video:
        imageio.mimwrite(os.path.join(save_video_dir, 'video.mp4'), x_samples, fps=4)
    

            
        