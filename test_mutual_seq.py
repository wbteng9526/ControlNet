import os
import einops
import torch
import numpy as np
from PIL import Image
import imageio
from tqdm import tqdm

from tutorial_dataset_test import MyDataset
from cldm.model import create_model, load_state_dict

if __name__ == "__main__":
    single_model = create_model('./models/single_ldm_v15.yaml')
    single_model.load_state_dict(load_state_dict('./models/single_epoch75.ckpt', location='cuda'))
    single_model = single_model.cuda()

    mutual_model = create_model('./models/mutual_ldm_v15.yaml')
    mutual_model.load_state_dict(load_state_dict('./lightning_logs/version_42/checkpoints/epoch=7-step=16631.ckpt', location='cuda'))
    mutual_model = mutual_model.cuda()

    # TODO: create argparse
    seq_name = 'seq_0001'
    save_image = True
    save_video = True
    save_image_dir = f'test/{seq_name}/image'
    save_video_dir = f'test/{seq_name}/video'
    os.makedirs(save_image_dir, exist_ok=True)
    os.makedirs(save_video_dir, exist_ok=True)

    dataset = MyDataset(seq_name)

    reference = None

    x_samples = []
    for i in tqdm(range(len(dataset))):
        data_dict = dataset.__getitem__(i)

        cond_img = torch.from_numpy(data_dict['cond_jpg']).float().cuda()
        cond_img = cond_img.unsqueeze(0)
        cond_img = einops.rearrange(cond_img, 'b h w c -> b c h w').clone()
        _, C, H, W = cond_img.shape
        cond = single_model.get_learned_conditioning(cond_img)
        # cond = {"c_t_crossattn": [cond]}

        reference_location = torch.from_numpy(data_dict['loc']).float().cuda()
        reference_location = reference_location.unsqueeze(0)

        shape = (1, 4, H // 8, W // 8)

        
        if i == 0:
            x_sample = single_model.sample(cond={"c_crossattn": [cond]}, batch_size=1, return_intermediates=False, verbose=True, shape=shape)
            x_sample = single_model.decode_first_stage(x_sample)
            
        else:
            x_sample = mutual_model.sample(cond={"c_t_crossattn": [cond]}, reference=reference, reference_location=reference_location, 
                                           batch_size=1, return_intermediates=False, verbose=True, shape=shape)
            x_sample = mutual_model.decode_first_stage(x_sample)
        
        encoder_posterior = mutual_model.encode_first_stage(x_sample)
        reference = mutual_model.get_first_stage_encoding(encoder_posterior)

        x_sample = (einops.rearrange(x_sample, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        x_samples.append(x_sample[0])

        if save_image:
            img = Image.fromarray(x_sample[0])
            img.save(os.path.join(save_image_dir, f'{i:03d}.png'))
    
    if save_video:
        imageio.mimwrite(os.path.join(save_video_dir, 'video.mp4'), x_samples, fps=12)
    

            
        