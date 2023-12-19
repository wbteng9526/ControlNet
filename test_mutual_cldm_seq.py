import os
import einops
import torch
import numpy as np
from PIL import Image
import imageio
from tqdm import tqdm

from tutorial_dataset_test import MyDataset
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

if __name__ == "__main__":
    single_model = create_model('./models/cldm_v15.yaml')
    single_model.load_state_dict(load_state_dict('./models/single_control_epoch12.ckpt', location='cuda'))
    single_model = single_model.cuda()

    mutual_model = create_model('./models/mutual_cldm_v15.yaml')
    mutual_model.load_state_dict(load_state_dict('./models/mutual_fintune_control_epoch19.ckpt', location='cuda'))
    mutual_model = mutual_model.cuda()

    ddim_sampler = DDIMSampler(single_model)

    # TODO: create argparse
    for seq in tqdm(range(1, 10)):
        seq_name = f'seq_{seq:04d}'
        save_image = True
        save_video = True
        save_image_dir = f'test/fintune_mutual_v1/{seq_name}/image'
        save_video_dir = f'test/fintune_mutual_v1/{seq_name}/video'
        os.makedirs(save_image_dir, exist_ok=True)
        os.makedirs(save_video_dir, exist_ok=True)

        dataset = MyDataset(seq_name)

        reference = None

        x_samples = []
        for i in tqdm(range(len(dataset))):
            data_dict = dataset.__getitem__(i)

            cond_txt = data_dict['txt']
            # cond = {"c_t_crossattn": [cond]}

            control = torch.from_numpy(data_dict['hint']).float().cuda()
            control = control.unsqueeze(0)
            control = einops.rearrange(control, 'b h w c -> b c h w')
            _, C, H, W = control.shape

            reference_location = torch.from_numpy(data_dict['loc']).float().cuda()
            reference_location = reference_location.unsqueeze(0)

            shape = (1, 4, H // 8, W // 8)

            
            if i == 0:
                x_sample = single_model.sample(cond={"c_crossattn": [single_model.get_learned_conditioning(cond_txt)], "c_concat": [control]}, batch_size=1, return_intermediates=False, verbose=True, shape=shape)
                x_sample = single_model.decode_first_stage(x_sample)
                
            else:
                x_sample = mutual_model.sample(cond={"c_t_crossattn": [mutual_model.get_learned_conditioning(cond_txt)], "c_t_concat": [control]}, reference=reference, reference_location=reference_location, 
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
            imageio.mimwrite(os.path.join(save_video_dir, 'video.mp4'), x_samples, fps=4)